use http::Uri;
use super::flservice::client_model_message;
use super::flservice::fl_client_train_service_client::FlClientTrainServiceClient;
use super::flservice::{ClientModelMessage, FloatBlock, MetaFloatBlockMessage, ModelConfig};
use tokio::sync::mpsc;
use tonic::Request;
use tonic::transport::Endpoint;


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
        trainer_port: i32
    ) -> Option<Vec<f32>> {
        match self {
            FlTraining::Dummy => {
                return Some(vec![0.001; params.len()]);
            }
            FlTraining::Grpc(client) => {
                let mut res = client
                    .train_for_round(config.clone(), params.clone(), round_id, model_id)
                    .await;
                while res.is_none() {
                    println!("Training failed, attempting to reconnect...");
                    // let uri = (client.grpc.inner as Endpoint).uri().clone();
                    // let new_client = client.grpc.
                    let uri = Uri::builder()
                        .scheme("http")
                        .authority(format!("127.0.0.1:{}", trainer_port).as_str())
                        .path_and_query("/")
                        .build()
                        .unwrap();
                    let channel = tonic::transport::Channel::builder(uri)
                        .connect()
                        .await;
                    if channel.is_err() {
                        println!("Reconnection failed, retrying in 5 seconds...");
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    } else {
                        client.grpc = Box::new(FlClientTrainServiceClient::new(channel.unwrap()));
                        res = client
                            .train_for_round(config.clone(), params.clone(), round_id, model_id)
                            .await;
                    }
                }
                return res;
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
        let outbound_local = outbound.clone();
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
                        model_id,
                        round_id,
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


        // let (response, _no_attempts) = FutureRetry::new(
        //     move || {
        //         let request = Request::new(Box::pin(
        //             tokio_stream::wrappers::ReceiverStream::new(rx.clone()),
        //         ));
        //         let response = self.grpc.train_for_round(request);
        //         response
        //     },
        //     |e| {
        //         println!("Error: {:?}", e);
        //         return RetryPolicy::WaitRetry(Duration::from_millis(5000))
        //     },
        // ).await.unwrap();
        // let response = self.grpc.

        let response = self
            .grpc
            .train_for_round(Request::new(Box::pin(
                tokio_stream::wrappers::ReceiverStream::new(rx),
            )))
            .await;
        if response.is_err() {
            println!("Train_for_round failed: {:?}", response.err());
            return None;
        }
        // Handle a timeout here..., can we attempt to reconnect?

        let mut inbound = response.unwrap().into_inner();

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
