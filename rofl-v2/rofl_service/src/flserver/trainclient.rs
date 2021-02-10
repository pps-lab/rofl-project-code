use super::flservice::client_model_message;
use super::flservice::fl_client_train_service_client::FlClientTrainServiceClient;
use super::flservice::{ClientModelMessage, FloatBlock, MetaFloatBlockMessage, ModelConfig};
use tokio::sync::mpsc;
use tonic::Request;

const CHAN_BUFFER_SIZE: usize = 100;
const NUM_FLOATS_PACKET: usize = 10000;

pub enum FlTraining {
    Dummy,
    Grpc(FlTrainClient),
}

impl FlTraining {
    pub async fn train_for_round(
        &mut self,
        config: ModelConfig,
        params: Vec<f32>,
        round_id: i32,
        model_id: i32,
    ) -> Option<Vec<f32>> {
        match self {
            FlTraining::Dummy => {
                return Some(vec![0.001; params.len()]);
            }
            FlTraining::Grpc(client) => {
                return client
                    .train_for_round(config, params, round_id, model_id)
                    .await;
            }
        }
    }
}

pub struct FlTrainClient {
    grpc: Box<FlClientTrainServiceClient<tonic::transport::Channel>>,
}

impl FlTrainClient {
    pub fn new(channel: tonic::transport::Channel) -> Self {
        FlTrainClient {
            grpc: Box::new(FlClientTrainServiceClient::new(channel)),
        }
    }
}

impl FlTrainClient {
    pub async fn train_for_round(
        &mut self,
        config: ModelConfig,
        params: Vec<f32>,
        round_id: i32,
        model_id: i32,
    ) -> Option<Vec<f32>> {
        let (outbound, rx) = mpsc::channel(CHAN_BUFFER_SIZE);
        let mut outbound_local = outbound.clone();
        let num_elems = params.len();
        tokio::spawn(async move {
            let request = ClientModelMessage {
                model_message: Some(client_model_message::ModelMessage::Config(config)),
            };
            let num_blocks = if num_elems % NUM_FLOATS_PACKET == 0 {
                num_elems / NUM_FLOATS_PACKET
            } else {
                num_elems / NUM_FLOATS_PACKET + 1
            };
            let _res = outbound_local.send(request).await;
            let request = ClientModelMessage {
                model_message: Some(client_model_message::ModelMessage::MetaBlockMessage(
                    MetaFloatBlockMessage {
                        model_id: model_id,
                        round_id: round_id,
                        num_blocks: num_blocks as i32,
                        num_floats: num_elems as i32,
                    },
                )),
            };
            let _res = outbound_local.send(request).await;
            for block_id in 0..num_blocks {
                let begin = block_id * NUM_FLOATS_PACKET;
                let end = if (block_id + 1) * NUM_FLOATS_PACKET > num_elems {
                    num_elems
                } else {
                    (block_id + 1) * NUM_FLOATS_PACKET
                };
                let request = ClientModelMessage {
                    model_message: Some(client_model_message::ModelMessage::ModelBlock(
                        FloatBlock {
                            block_number: block_id as u32,
                            floats: params[begin..end].to_vec(),
                        },
                    )),
                };
                let _res = outbound_local.send(request).await;
            }
        });

        let response = self.grpc.train_for_round(Request::new(rx)).await.unwrap();
        let mut inbound = response.into_inner();

        let mut trained_model: Vec<f32> = Vec::with_capacity(num_elems);
        let mut wait_for_blocks = 0;
        while let Some(response) = inbound.message().await.unwrap() {
            match response.model_message.unwrap() {
                client_model_message::ModelMessage::Config(_) => {
                    panic!("Config should not be received");
                }
                client_model_message::ModelMessage::MetaBlockMessage(meta_msg) => {
                    wait_for_blocks = meta_msg.num_blocks;
                }
                client_model_message::ModelMessage::ModelBlock(params) => {
                    trained_model.extend(params.floats.iter());
                    wait_for_blocks -= 1;
                    if wait_for_blocks == 0 {
                        break;
                    }
                }
            }
        }
        return Some(trained_model);
    }
}
