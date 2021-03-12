use super::flservice::{model_parameters, server_model_data, train_request, train_response};
use super::flservice::{
    Config, CryptoConfig, DataBlock, ModelConfig, ModelParameters, TrainRequest,
    WorkerRegisterMessage,
};
use super::params::{EncModelParamType, EncModelParams, PlainParams};
use super::{flservice::flservice_client::FlserviceClient, logs::TimeState};
use crate::flserver::trainclient::FlTraining;
use crate::flserver::util::DataBlockStorage;
use curve25519_dalek::scalar::Scalar;
use log::info;
use model_parameters::{ModelParametersMeta, ParamMessage};
use prost::Message;
use rofl_crypto::pedersen_ops::zero_scalar_vec;
use tokio::sync::mpsc;
use tokio::sync::mpsc::Sender;
use tonic::Request;

const CHAN_BUFFER_SIZE: usize = 100;
const NUM_PARAM_BYTES_PER_PACKET: usize = 1 << 20;

struct ServerModelDataState {
    config: Vec<Config>,
    data: DataBlockStorage,
}

impl ServerModelDataState {
    fn new() -> Self {
        ServerModelDataState {
            config: Vec::new(),
            data: DataBlockStorage::new(),
        }
    }

    fn update(&mut self, config: Config) {
        self.config.clear();
        self.config.push(config);
    }

    fn get_current_config(&self) -> Option<&Config> {
        self.config.last()
    }
}

fn get_enc_type_from_config(config: &Config) -> Option<EncModelParamType> {
    if let Some(crypto_config) = &config.crypto_config {
        return EncModelParamType::get_type_from_int(&crypto_config.enc_type);
    }
    None
}

fn get_crypto_config(config: &Config) -> Option<&CryptoConfig> {
    if let Some(crypto_config) = &config.crypto_config {
        return Some(crypto_config);
    }
    None
}

fn get_model_config(config: &Config) -> Option<&ModelConfig> {
    if let Some(model_config) = &config.model_config {
        return Some(model_config);
    }
    None
}

fn derive_dummy_blindings(size: usize) -> Vec<Scalar> {
    zero_scalar_vec(size)
}

pub struct FlServiceClient {
    client_id: i32,
    grpc: Box<FlserviceClient<tonic::transport::Channel>>,
    training_client: Box<FlTraining>,
}

impl FlServiceClient {
    pub fn new(
        client_id: i32,
        channel: tonic::transport::Channel,
        training_client: Box<FlTraining>,
    ) -> Self {
        FlServiceClient {
            client_id,
            grpc: Box::new(FlserviceClient::new(channel)),
            training_client,
        }
    }

    async fn train_model_locally(
        &mut self,
        params: PlainParams,
        conifg: &Config,
        round_id: i32,
        model_id: i32,
    ) -> Option<PlainParams> {
        let model_config = get_model_config(conifg).unwrap();
        let trained_params_opt = self
            .training_client
            .train_for_round(model_config.clone(), params.into_vec(), round_id, model_id)
            .await;
        trained_params_opt.map(|trained_params| PlainParams {content: trained_params,})
    }

    pub fn encrypt_data(&self, params: &PlainParams, conifg: &Config) -> Option<EncModelParams> {
        let enc_type_opt = get_enc_type_from_config(conifg);
        let crypto_config = get_crypto_config(conifg).unwrap();
        let blindings = derive_dummy_blindings(params.content.len());
        if let Some(enc_type) = enc_type_opt {
            return Some(
                EncModelParams::encrypt(&enc_type, params, crypto_config, &blindings).unwrap(),
            );
        }
        None
    }

    async fn handle_send_data(
        &self,
        enc_params: EncModelParams,
        outbound: &mut Sender<TrainRequest>,
        model_id: i32,
        round_id: u32,
    ) -> usize {
        let mut data_send = 0;
        let buffer = enc_params.serialize();
        let len_buffer = buffer.len();
        let mut num_packets = len_buffer / NUM_PARAM_BYTES_PER_PACKET;
        if len_buffer % NUM_PARAM_BYTES_PER_PACKET != 0 {
            num_packets += 1;
        }

        let meta_message = TrainRequest {
            param_message: Some(train_request::ParamMessage::Params(ModelParameters {
                param_message: Some(model_parameters::ParamMessage::ParamMeta(
                    ModelParametersMeta {
                        model_id,
                        round_id: round_id as i32,
                        num_blocks: num_packets as i32,
                    },
                )),
            })),
        };
        data_send += meta_message.encoded_len();
        let _res = outbound.send(meta_message).await;
        //TODO : handle error
        for packet_num in 0..num_packets {
            let begin = packet_num * NUM_PARAM_BYTES_PER_PACKET;
            let end = (packet_num + 1) * NUM_PARAM_BYTES_PER_PACKET;
            let end = if end > len_buffer { len_buffer } else { end };

            let data_packet = TrainRequest {
                param_message: Some(train_request::ParamMessage::Params(ModelParameters {
                    param_message: Some(model_parameters::ParamMessage::ParamBlock(DataBlock {
                        block_number: packet_num as u32,
                        data: buffer[begin..end].to_vec(),
                    })),
                })),
            };
            data_send += data_packet.encoded_len();
            let _res = outbound.send(data_packet).await;
            //TODO : handle error
        }
        data_send
    }

    pub async fn train_model(&mut self, model_id: i32, _verbose: bool) {
        info!("Client {} starts training model", self.client_id);
        let (mut outbound, rx) = mpsc::channel(CHAN_BUFFER_SIZE);

        let client_id = self.client_id;
        let outbound_local = outbound.clone();
        tokio::spawn(async move {
            info!("Client {} registers to train model {}", client_id, model_id);
            // Register phase
            let request = TrainRequest {
                param_message: Some(train_request::ParamMessage::StartMessage(
                    WorkerRegisterMessage {
                        model_id,
                        client_id,
                    },
                )),
            };
            let _res = outbound_local.send(request).await;
        });

        let response = self
            .grpc
            .train_model(Request::new(Box::pin(
                tokio_stream::wrappers::ReceiverStream::new(rx),
            )))
            .await
            .unwrap();
        let mut inbound = response.into_inner();

        let mut state = ServerModelDataState::new();
        let time_state = TimeState::new();
        let mut data_received = 0;
        // Protocol loop
        info!(
            "Client {} starts protocol loop for model {}",
            client_id, model_id
        );
        while let Some(response) = inbound.message().await.unwrap() {
            data_received += response.encoded_len();
            match response.param_message.unwrap() {
                train_response::ParamMessage::Params(msg) => match msg.model_message.unwrap() {
                    server_model_data::ModelMessage::Config(config) => {
                        state.update(config);
                    }
                    server_model_data::ModelMessage::ModelBlock(msg) => {
                        match msg.param_message.unwrap() {
                            ParamMessage::ParamMeta(meta) => {
                                info!(
                                    "Client {} receives model parameters for round {}",
                                    client_id, meta.round_id
                                );
                                state.data.init(meta);
                                time_state.record_instant();
                            }
                            ParamMessage::ParamBlock(data_block) => {
                                let _ok = state.data.apply(&data_block);
                                if state.data.done() {
                                    let params = PlainParams::deserialize(state.data.data_ref());
                                    let round_id = state.data.get_round_id();
                                    state.data.reset_mem();
                                    let current_config_ref = state.get_current_config().unwrap();
                                    time_state.record_instant();
                                    let trained_params = self
                                        .train_model_locally(
                                            params,
                                            current_config_ref,
                                            round_id as i32,
                                            model_id,
                                        )
                                        .await;
                                    time_state.record_instant();
                                    let encrypted_params = self
                                        .encrypt_data(&trained_params.unwrap(), current_config_ref);
                                    time_state.record_instant();
                                    let data_sent = self
                                        .handle_send_data(
                                            encrypted_params.unwrap(),
                                            &mut outbound,
                                            model_id,
                                            round_id,
                                        )
                                        .await;
                                    info!(
                                        "Client {} finished training for round {}",
                                        client_id, round_id
                                    );
                                    time_state.record_instant();
                                    time_state.log_bench_times_with_bandwith(
                                        round_id as i32,
                                        data_received,
                                        data_sent,
                                    );
                                    time_state.reset();
                                    data_received = 0;
                                }
                            }
                        }
                    }
                },
                train_response::ParamMessage::ErrorMessage(_) => {
                    todo!("Handle erros");
                }
                train_response::ParamMessage::DoneMessage(_) => {
                    info!(
                        "Client {} terminates, server done message received",
                        client_id
                    );
                    break;
                }
            }
        }
    }
}
