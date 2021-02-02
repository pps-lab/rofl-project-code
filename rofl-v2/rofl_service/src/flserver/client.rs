use tokio::sync::mpsc::Sender;
use model_parameters::{ModelParametersMeta, ParamMessage};
use crate::flserver::util::DataBlockStorage;
use super::flservice::flservice_client::FlserviceClient;
use super::{flservice::flservice_server::Flservice, params::{EncModelParamType, EncModelParams, PlainParams}};
use super::flservice::{DataBlock, Config, ModelConfig, CryptoConfig, ServerModelData, WorkerRegisterMessage, ModelRegisterResponse, StatusMessage, ModelParameters, TrainRequest, TrainResponse, ModelSelection};
use super::flservice::{train_request, train_response, server_model_data, model_parameters};
use std::sync::Arc;
use std::iter::FromIterator;
use tokio::sync::mpsc;
use tonic::{Request};


const CHAN_BUFFER_SIZE: usize = 100;
const NUM_PARAM_BYTES_PER_PACKET: usize = 1 << 20;

struct ServerModelDataState {
    config : Vec<Config>,
    data : DataBlockStorage,
}

impl ServerModelDataState {
    fn new() -> Self {
        ServerModelDataState {
            config : Vec::new(),
            data : DataBlockStorage::new()
        }
    }

    fn update(&mut self, config : Config) {
        self.config.clear();
        self.config.push(config);
    }

    fn get_current_config(&self) -> Option<&Config>  {
        self.config.last()
    }
       
}

fn get_enc_type_from_config(config : &Config) -> Option<EncModelParamType> {
    if let Some(crypto_config) = &config.crypto_config {
        return EncModelParamType::get_type_from_int(&crypto_config.enc_type);
    }
    None
}

fn get_crypto_config(config : &Config) -> Option<&CryptoConfig> {
    if let Some(crypto_config) = &config.crypto_config {
        return  Some(crypto_config);
    }
    None
}

pub struct FlServiceClient {
    client_id : i32,
    grpc : Box<FlserviceClient<tonic::transport::Channel>>,
}

impl FlServiceClient {

    pub fn new(client_id: i32, channel : tonic::transport::Channel) -> Self {
        FlServiceClient {
            client_id : client_id,
            grpc : Box::new(FlserviceClient::new(channel)),
        }
    }

    pub fn train_model_locally(&self, params : PlainParams, conifg : &Config) -> Option<PlainParams> {
        Some(params)
    }

    pub fn encrypt_data(&self, params : &PlainParams, conifg : &Config) -> Option<EncModelParams> {
        let enc_type_opt = get_enc_type_from_config(conifg);
        if let Some(enc_type) = enc_type_opt {
            return Some(EncModelParams::encrypt(&enc_type, params));
        }
        None
    }

    async fn handle_send_data(&self, enc_params : EncModelParams, outbound : &mut Sender<TrainRequest>, model_id : i32, round_id : u32) -> () {
        let buffer = enc_params.serialize();
        let len_buffer = buffer.len();
        let mut num_packets = len_buffer / NUM_PARAM_BYTES_PER_PACKET;
        if len_buffer % NUM_PARAM_BYTES_PER_PACKET != 0 {
            num_packets += 1;
        }
        
        let meta_message = TrainRequest {
            param_message : Some(train_request::ParamMessage::Params(ModelParameters {
                param_message : Some(model_parameters::ParamMessage::ParamMeta(ModelParametersMeta { 
                    model_id : model_id,
                    round_id : round_id as i32,
                    num_blocks : num_packets as i32
                }))
            }))
        };

        let _res = outbound.send(meta_message).await;
        //TODO : handle error

        for packet_num in 0..num_packets {
            let begin = packet_num * NUM_PARAM_BYTES_PER_PACKET;
            let end = (packet_num + 1) * NUM_PARAM_BYTES_PER_PACKET;
            let end = if end > len_buffer { len_buffer } else { end };
            
            let data_packet = TrainRequest {
                param_message : Some(train_request::ParamMessage::Params(ModelParameters {
                    param_message : Some(model_parameters::ParamMessage::ParamBlock(DataBlock { 
                        block_number : packet_num as u32, 
                        data : Vec::from_iter(buffer[begin..end].iter().cloned())
                    }))
                }))
            };
            let _res = outbound.send(data_packet).await;
              //TODO : handle error
        }
    }

    pub async fn train_model(&mut self, model_id : i32, verbose: bool) -> () {
        println!("Client {} starts training model", self.client_id);
        let (mut outbound, rx) = mpsc::channel(CHAN_BUFFER_SIZE);
       
        let client_id =  self.client_id;
        let mut outbound_local = outbound.clone();
        tokio::spawn(async move {
            println!("Client {} registers to train model {}", client_id, model_id);
            // Register phase
            let request = TrainRequest {
                param_message : Some(train_request::ParamMessage::StartMessage(WorkerRegisterMessage {
                    model_id : model_id,
                    client_id : client_id,
                }))
            };
            let _res = outbound_local.send(request).await;
        });

        let response = self.grpc.train_model(Request::new(rx)).await.unwrap();
        let mut inbound = response.into_inner();

        let mut state = ServerModelDataState::new();
        // Protocol loop
        println!("Client {} starts protocol loop for model {}", client_id, model_id);
        while let Some(response) = inbound.message().await.unwrap() {
           
            match response.param_message.unwrap() {
                train_response::ParamMessage::Params(msg) => {
                    match msg.model_message.unwrap() {
                        server_model_data::ModelMessage::Config(config) => {
                            state.update(config);
                        }
                        server_model_data::ModelMessage::ModelBlock(msg) => {
                            match msg.param_message.unwrap() {
                                ParamMessage::ParamMeta(meta) => {
                                    state.data.init(meta);
                                }
                                ParamMessage::ParamBlock(data_block) => {
                                    let ok = state.data.apply(&data_block);
                                    if state.data.done() {
                                        let params = PlainParams::deserialize(state.data.data_ref());
                                        let round_id = state.data.get_round_id();
                                        state.data.reset_mem();
                                        let current_config_ref = state.get_current_config().unwrap();
                                        let trained_params = self.train_model_locally(params, current_config_ref);
                                        let encrypted_params = self.encrypt_data(&trained_params.unwrap(), current_config_ref);
                                        let _res = self.handle_send_data(encrypted_params.unwrap(), &mut outbound, model_id, round_id).await;
                                        println!("Client {} finished training for round {}", client_id, round_id);
                                    }
                                }
                            }
                        }
                        server_model_data::ModelMessage::DoneMessage(_) => {
                            break;
                        }
                    }
                }
                train_response::ParamMessage::ErrorMessage(_) => {
                    todo!("Handle erros");
                }
            }
        }
    }
}