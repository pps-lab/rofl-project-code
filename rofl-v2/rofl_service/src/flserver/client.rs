use model_parameters::ParamMessage;
use crate::flserver::util::DataBlockStorage;
use super::flservice::flservice_client::FlserviceClient;
use super::{flservice::flservice_server::Flservice, params::{EncModelParamType, EncModelParams, PlainParams}};
use super::flservice::{DataBlock, Config, ModelConfig, CryptoConfig, ServerModelData, WorkerRegisterMessage, ModelRegisterResponse, StatusMessage, ModelParameters, TrainRequest, TrainResponse, ModelSelection};
use super::flservice::{train_request, train_response, server_model_data, model_parameters};
use std::sync::Arc;
use tokio::sync::mpsc;
use tonic::{Request};


const CHAN_BUFFER_SIZE: usize = 100;

struct ClientServerModelData {
    config : Config,
    data : Vec<f32>,
}
pub struct FlServiceClient {
    client_id : i32,
    grpc : Box<FlserviceClient<tonic::transport::Channel>>,
}



impl FlServiceClient {

    pub fn train_model_locally(&self, params : PlainParams, conifg : &Config) -> Option<PlainParams> {
        Some(params)
    }

    pub fn encrypt_data(&self, params : PlainParams, conifg : &Config) -> Option<EncModelParams> {
        None
    }

    async fn handle_send_data(&self) {
       
    }

    pub async fn train_model(&mut self, model_id : i32, verbose: bool) -> () {
        let (mut outbound, rx) = mpsc::channel(CHAN_BUFFER_SIZE);
        let response = self.grpc.train_model(Request::new(rx)).await.unwrap();
        let mut inbound = response.into_inner();
        
        
        // Register phase
        let request = TrainRequest {
            param_message : Some(train_request::ParamMessage::StartMessage(WorkerRegisterMessage {
                model_id : model_id,
                client_id : self.client_id,
            }))
        };
        let _res = outbound.send(request).await;

        let mut block_storage = DataBlockStorage::new();
        let mut current_config : Option<Config> =  None;
        // Protocol loop
        while let Some(response) = inbound.message().await.unwrap() {
            match response.param_message.unwrap() {
                train_response::ParamMessage::Params(msg) => {
                    match msg.model_message.unwrap() {
                        server_model_data::ModelMessage::CryptoConfig(config) => {
                            current_config = Some(config);
                        }
                        server_model_data::ModelMessage::ModelBlock(msg) => {
                            match msg.param_message.unwrap() {
                                ParamMessage::ParamMeta(meta) => {
                                    block_storage.init(meta);
                                }
                                ParamMessage::ParamBlock(data_block) => {
                                    let ok = block_storage.apply(&data_block);
                                    if block_storage.done() {
                                        let params = PlainParams::deserialize(block_storage.data_ref());
                                        block_storage.reset_mem();
                                        //let trained_params = self.train_model_locally(params, &current_config.unwrap());
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