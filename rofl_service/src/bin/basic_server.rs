use clap::{App, Arg};
use params::GlobalModel;
use flexi_logger::{opt_format, LogTarget, Logger};
use rofl_service::flserver::flservice::flservice_server::FlserviceServer;
use rofl_service::flserver::flservice::{CryptoConfig, ModelConfig};
use rofl_service::flserver::params;
use rofl_service::flserver::server::DefaultFlService;
use rofl_service::flserver::server::TrainingState;
use tonic::transport::Server;

fn dummy_training_state(
    num_clients: i32,
    num_params: i32,
    num_in_memory: i32,
    train_until_round: i32,
) -> TrainingState {
    let model_confing = ModelConfig {
        num_of_clients: num_clients,
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
    let crypto_config = CryptoConfig {
        value_range: 8,
        n_partition: 1,
        l2_value_range: 32,
        check_percentage: 0.1,
        enc_type: params::PLAIN_TYPE as i32,
    };
    TrainingState::new(
        model_confing.model_id,
        model_confing,
        crypto_config,
        num_params,
        num_in_memory,
        train_until_round,
        // GlobalModel::new(num_params as usize, 1.0),
        GlobalModel::new_from_file(1.0, "models/mnist_dev_initialized.txt"),
        false,
        true,
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = App::new("FL Server")
        .version("1.0")
        .author("Lukas B. <lubu@inf.ethz.ch>")
        .about("Runs the Fl _server")
        .arg(
            Arg::with_name("address")
                .short("a")
                .long("address")
                .help("the ip addr of the server")
                .default_value("[::1]")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("port")
                .short("p")
                .long("port")
                .help("the port of the server")
                .default_value("50051")
                .takes_value(true),
        )
        .get_matches();
    let ip = matches.value_of("address").unwrap_or("default.conf");
    let port = matches.value_of("port").unwrap_or("default.conf");
    let addr = format!("{}:{}", ip, port).parse().unwrap();
    let service = DefaultFlService::new(8);

    Logger::with_str("info")
        .log_target(LogTarget::StdOut)
        .format_for_stdout(opt_format)
        .start()?;

    service.register_new_trainig_state(dummy_training_state(1, 19166, 5, 10));
    Server::builder()
        .tcp_nodelay(true)
        .add_service(FlserviceServer::new(service))
        .serve(addr)
        .await?;
    Ok(())
}
