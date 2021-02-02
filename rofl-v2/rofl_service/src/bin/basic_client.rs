use rofl_crypto::pedersen_ops::zero_scalar_vec;
use rofl_service::flserver::client::FlServiceClient;
use std::sync::Arc;
use tokio::sync::mpsc;
use tonic::{Request};
use std::time::Duration;

async fn start_dummy_client(client_id : i32, model_id : i32, verbose : bool) {
    let channel = tonic::transport::Channel::from_static("http://[::1]:50051").connect().await.unwrap();
    let blindings = zero_scalar_vec(100);
    let mut client =  FlServiceClient::new(client_id, channel, blindings);
    client.train_model(model_id, verbose).await;
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let num_clients = 10;
    let mut tasks= Vec::with_capacity(num_clients as usize);
    for i in 1..num_clients {
        let client_id = i;
        tasks.push(tokio::spawn(async move {
            start_dummy_client(client_id, 1, true).await;
        }));
    }
    start_dummy_client(0, 1, true).await;
    for task in tasks {
        let _ou = task.await;
    }
    Ok(())
}