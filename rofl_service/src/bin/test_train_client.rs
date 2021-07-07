use rofl_service::flserver::{flservice::ModelConfig, trainclient::FlTrainClient};

async fn create_train_dummy_client() -> FlTrainClient {
    let channel = tonic::transport::Channel::from_static("http://[::1]:50016")
        .connect()
        .await
        .unwrap();
    FlTrainClient::new(channel)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_config = ModelConfig {
        num_of_clients: 10,
        client_batch_size: 10,
        num_local_epochs: 1,
        optimizer: "sgd".to_string(),
        learning_rate: 0.5,
        loss: "crossentropy".to_string(),
        metrics: "accuracy".to_string(),
        image_augmentation: false,
        lr_decay: 0.5,
        model_id: 1,
        probabilistic_quantization: false,
        fp_bits: 32,
        fp_frac: 7,
        range_bits: 8,
    };

    let mut client = create_train_dummy_client().await;
    let before: Vec<f32> = vec![0.01; 19166];
    let res = client.train_for_round(model_config, before, 0, 0).await;
    println!("{:?}", res.unwrap());
    Ok(())
}
