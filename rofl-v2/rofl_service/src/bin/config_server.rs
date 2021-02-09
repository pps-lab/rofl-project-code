use clap::{App, Arg};
use params::GlobalModel;
use rofl_service::flserver::flservice::flservice_server::FlserviceServer;
use rofl_service::flserver::flservice::{CryptoConfig, ModelConfig};
use rofl_service::flserver::params;
use rofl_service::flserver::server::DefaultFlService;
use rofl_service::flserver::server::TrainingState;
use std::fs::File;
use std::io::Read;
use tonic::transport::Server;
use yaml_rust::{YamlEmitter, YamlLoader};

fn get_training_state_from_config(path: &str) -> TrainingState {
    let config_str = match File::open(path) {
        Ok(mut file) => {
            let mut content = String::new();
            let err = file.read_to_string(&mut content);
            content
        }
        Err(error) => {
            panic!("{}", error);
        }
    };
    let docs = YamlLoader::load_from_str(&config_str).unwrap();
    let doc = &docs[0];
    let experiment = &doc["base_experiment"];
    // NO ERROR HANDLING !!!
    let num_clients = experiment["environment"]["num_clients"]
        .as_i64()
        .expect("Missing num_clients") as i32;
    let num_params = experiment["client"]["num_params"]
        .as_i64()
        .expect("Missing num_params") as i32;
    let num_in_memory = 2;
    let train_until_round = experiment["server"]["num_rounds"]
        .as_i64()
        .expect("Missing train_until_round") as i32;
    let global_learning_rate = experiment["server"]["global_learning_rate"]
        .as_f64()
        .unwrap_or(1.0) as f32;

    let client_training = &experiment["client"]["benign_training"];
    let num_epochs = client_training["num_epochs"].as_i64().unwrap_or(1) as i32;
    let batch_size = client_training["batch_size"].as_i64().unwrap_or(24) as i32;
    let optimizer = client_training["optimizer"].as_str().unwrap_or("Adam");
    let local_learning_rate = client_training["learning_rate"].as_f64().unwrap_or(0.001) as f32;

    let crypto = &experiment["crypto"];
    let value_range = crypto["value_range"].as_i64().unwrap_or(8) as i32;
    let l2_value_range = crypto["l2_value_range"].as_i64().unwrap_or(32) as i32;
    let n_partition = crypto["n_partition"].as_i64().unwrap_or(1) as i32;
    let check_percentage = crypto["check_percentage"].as_i64().unwrap_or(100) as i32;
    let fp_bits = crypto["fp_bits"].as_i64().unwrap_or(32) as i32;
    let fp_frac = crypto["fp_frac"].as_i64().unwrap_or(32) as i32;

    let enc_type = match crypto["enc_type"].as_str().unwrap_or("Range") {
        "Range" => params::ENC_RANGE_TYPE,
        "l2" => params::ENC_L2_TYPE,
        "Plain" => params::PLAIN_TYPE,
        _ => params::PLAIN_TYPE,
    };

    let model_confing = ModelConfig {
        num_of_clients: num_clients,
        client_batch_size: batch_size,
        num_local_epochs: num_epochs,
        optimizer: optimizer.to_string(),
        learning_rate: local_learning_rate,
        loss: "crossentropy".to_string(),
        metrics: "accuracy".to_string(),
        image_augmentation: false,
        lr_decay: 0.0,
        model_id: 1,
        probabilistic_quantization: false,
        fp_bits: fp_bits,
        fp_frac: fp_frac,
        range_bits: 8,
    };
    let crypto_config = CryptoConfig {
        value_range: value_range,
        n_partition: n_partition,
        l2_value_range: l2_value_range,
        check_percentage: check_percentage,
        enc_type: enc_type as i32,
    };
    TrainingState::new(
        model_confing.model_id,
        model_confing,
        crypto_config,
        num_params,
        num_in_memory,
        train_until_round,
        GlobalModel::new(num_params as usize, global_learning_rate),
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = App::new("ROFL Config Server")
        .version("1.0")
        .author("Lukas B. <lubu@inf.ethz.ch>")
        .about("Runs the RoFl server based on a YAML config")
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
        .arg(
            Arg::with_name("config")
                .short("c")
                .long("config")
                .help("the YAML config file path")
                .default_value("./configs/example_config.yml")
                .takes_value(true),
        )
        .get_matches();
    let ip = matches.value_of("address").unwrap_or("default.conf");
    let port = matches.value_of("port").unwrap_or("default.conf");
    let addr = format!("{}:{}", ip, port).parse().unwrap();
    let config = matches.value_of("config").unwrap_or("default.conf");
    let service = DefaultFlService::new();
    service.register_new_trainig_state(get_training_state_from_config(config));
    Server::builder()
        .tcp_nodelay(true)
        .add_service(FlserviceServer::new(service))
        .serve(addr)
        .await?;
    Ok(())
}
