use clap::{App, Arg};
use flexi_logger::{opt_format, LogTarget, Logger};
use log::info;
use num_cpus;
use params::GlobalModel;
use rofl_service::flserver::flservice::flservice_server::FlserviceServer;
use rofl_service::flserver::flservice::{CryptoConfig, ModelConfig};
use rofl_service::flserver::logs::{bench_logger, BENCH_TAG};
use rofl_service::flserver::params;
use rofl_service::flserver::server::DefaultFlService;
use rofl_service::flserver::server::TrainingState;
use std::fs::File;
use std::io::Read;
use tonic::transport::Server;
use yaml_rust::YamlLoader;

fn get_training_state_from_config(path: &str, lazy_eval: bool, std_init: f32) -> TrainingState {
    let config_str = match File::open(path) {
        Ok(mut file) => {
            let mut content = String::new();
            let _err = file.read_to_string(&mut content);
            content
        }
        Err(error) => {
            panic!("{}", error);
        }
    };
    let docs = YamlLoader::load_from_str(&config_str).unwrap();
    let experiment = &docs[0];
    // NO ERROR HANDLING !!!
    let num_clients = experiment["environment"]["num_clients"]
        .as_i64()
        .expect("Missing num_clients") as i32;
    let init_model_path = experiment["client"]["model_init_path"].as_str();
    let num_params = experiment["client"]["num_params"].as_i64();
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
    let check_percentage = crypto["check_percentage"].as_f64().unwrap_or(1.0) as f32;
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
    let global_model = if let Some(model_path) = init_model_path {
        info!("Global model loaded from file: {}", model_path);
        GlobalModel::new_from_file(
            global_learning_rate,
            model_path,
        )
        
    } else {
        info!("Global model initialized with normal distribution std: {}", std_init);
        GlobalModel::new_from_normal_distribution(
            num_params.unwrap() as usize,
            global_learning_rate,
            std_init,
        )
    };
    info!("Global model has {} parameters and learning rate {}", global_model.get_num_params(), global_model.learning_rate);
    info!("Lazy verification {}, Train until round {}", lazy_eval, train_until_round);
    TrainingState::new(
        model_confing.model_id,
        model_confing,
        crypto_config,
        global_model.get_num_params()as i32,
        num_in_memory,
        train_until_round,
        global_model,
        true,
        lazy_eval,
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
        .arg(
            Arg::with_name("dleval")
                .short("l")
                .long("dleval")
                .help("Disable lazy verification")
        )
        .arg(
            Arg::with_name("vthreads")
                .short("t")
                .long("vthreads")
                .help("The number of verification threads Default: Num CPU's")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("initstd")
                .short("s")
                .long("initstd")
                .help("The standard deviation of the normal distribution to initialize the global model")
                .default_value("0.05")
                .takes_value(true),
        )
        .get_matches();
    let ip = matches.value_of("address").unwrap_or("default.conf");
    let port = matches.value_of("port").unwrap_or("default.conf");
    let addr = format!("{}:{}", ip, port).parse().unwrap();
    let config = matches.value_of("config").unwrap_or("default.conf");
    let std_init = matches
        .value_of("initstd")
        .unwrap_or("default.conf")
        .parse::<f32>()
        .unwrap();
    let num_threads = matches.value_of("vthreads");
    let num_threads = match num_threads {
        Some(str) => str.parse::<i32>().unwrap() as usize,
        None => num_cpus::get() as usize,
    };
    Logger::with_str("info")
        .log_target(LogTarget::StdOut)
        .format_for_stdout(opt_format)
        .add_writer(BENCH_TAG, bench_logger())
        .start()?;
    
    let service = DefaultFlService::new(num_threads);
    service.register_new_trainig_state(get_training_state_from_config(
        config,
        !matches.is_present("dleval"),
        std_init,
    ));
    
    Server::builder()
        .tcp_nodelay(true)
        .add_service(FlserviceServer::new(service))
        .serve(addr)
        .await?;
    Ok(())
}
