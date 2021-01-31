
use rofl_service::flserver::server::DefaultFlService;
use rofl_service::flserver::server::TrainingState;
use rofl_service::flserver::params;
use rofl_service::flserver::flservice::{ModelConfig, CryptoConfig};
use tonic::{transport::Server};
use clap::{Arg, App};

fn dummy_training_state(num_clients : i32, num_params : i32) -> TrainingState {
     let model_confing = ModelConfig {
          num_of_clients: num_clients,
          client_batch_size: 10,
          num_local_epochs: 1,
          optimizer: String::form("sgd"),
          learning_rate: 0.5,
          loss:  String::form("crossentropy"),
          metrics: String::form("accuracy"),
          image_augmentation: false,
          lr_decay: 0.5,
          model_id: 1,
          probabilistic_quantization: false,
          fp_bits: 12,
          fp_frac: 12,
          range_bits: 2,
     };
     let crypto_config = CryptoConfig {
          value_range: 12,
          n_partition: 2,
          l2_value_range: 4,
          enc_type: params::PLAIN_TYPE as i32,
     };
     TrainingState::new(model_confing.model_id, model_confing, crypto_config, num_params)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = App::new("FL Server")
                          .version("1.0")
                          .author("Lukas B. <lubu@inf.ethz.ch>")
                          .about("Runs the Fl _server")
                          .arg(Arg::with_name("address")
                               .short("a")
                               .long("address")
                               .help("the ip addr of the server")
                               .default_value("[::1]")
                               .takes_value(true))
                          .arg(Arg::with_name("port")
                               .short("p")
                               .long("port")
                               .help("the port of the server")
                               .default_value("50051")
                               .takes_value(true))
                          .get_matches();
    let ip = matches.value_of("address").unwrap_or("default.conf");
    let port = matches.value_of("port").unwrap_or("default.conf");
    //let addr : &str = format!("{}:{}",ip, port).parse().unwrap();
    let service = DefaultFlService::new();
    Ok(())
}