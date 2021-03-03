use rofl_service::flserver::logs::bench_logger;
use rofl_service::flserver::logs::BENCH_TAG;
use clap::{App, Arg};
use http::Uri;
use rofl_service::flserver::client::FlServiceClient;
use rofl_service::flserver::trainclient::FlTrainClient;
use rofl_service::flserver::trainclient::FlTraining;
use tonic::transport::Channel;
use flexi_logger::{LogTarget, Logger, opt_format};

async fn start_client(
    channel: Channel,
    client_id: i32,
    model_id: i32,
    verbose: bool,
    trainer_port: i32,
) {
    let trainer = if trainer_port == 0 {
        Box::new(FlTraining::Dummy)
    } else {
        let uri = Uri::builder()
            .scheme("http")
            .authority(format!("127.0.0.1:{}", trainer_port).as_str())
            .path_and_query("/")
            .build()
            .unwrap();
        let channel = tonic::transport::Channel::builder(uri)
            .connect()
            .await
            .expect(format!("Failed to connect to trainer at port {}", trainer_port).as_str());
        Box::new(FlTraining::Grpc(FlTrainClient::new(channel)))
    };
    let mut client = FlServiceClient::new(client_id, channel, trainer);
    client.train_model(model_id, verbose).await;
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = App::new("RoFl Clients")
        .version("1.0")
        .author("Lukas B. <lubu@inf.ethz.ch>")
        .about("Runs a certain number of RoFl clients")
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
            Arg::with_name("nclients")
                .short("n")
                .help("Number of clients to run")
                .default_value("10")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("idstart")
                .short("c")
                .help("The first client id, idstart, idstart + 1, ...")
                .default_value("0")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("modelid")
                .short("m")
                .help("The model  id to train on")
                .default_value("1")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("trainport")
                .short("r")
                .help("The clients connect to the local trainer at the given port")
                .default_value("0")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("numtrainers")
                .short("t")
                .help("The number of local training services")
                .default_value("1")
                .takes_value(true),
        )
        .get_matches();
    let ip = matches.value_of("address").unwrap_or("default.conf");
    let port = matches.value_of("port").unwrap_or("default.conf");
    let addr = &format!("{}:{}", ip, port);
    let num_clients = matches
        .value_of("nclients")
        .unwrap_or("default.conf")
        .parse::<i32>()
        .unwrap();
    let start_id = matches
        .value_of("idstart")
        .unwrap_or("default.conf")
        .parse::<i32>()
        .unwrap();
    let model_id = matches
        .value_of("modelid")
        .unwrap_or("default.conf")
        .parse::<i32>()
        .unwrap();
    let trainer_port = matches
        .value_of("trainport")
        .unwrap_or("default.conf")
        .parse::<i32>()
        .unwrap();
    let num_trainiers = matches
        .value_of("numtrainers")
        .unwrap_or("default.conf")
        .parse::<i32>()
        .unwrap();
    let uri = Uri::builder()
        .scheme("http")
        .authority(addr.as_str())
        .path_and_query("/")
        .build()
        .unwrap();
    
    Logger::with_str("info")
        .log_target(LogTarget::StdOut)  
        .format_for_stdout(opt_format)    
        .add_writer(BENCH_TAG, bench_logger())                      
        .start()?;

    let mut tasks = Vec::with_capacity(num_clients as usize);
    let mut port = trainer_port;
    for i in (start_id + 1)..(num_clients + start_id) {
        let client_id = i;
        let local_uri = uri.clone();
        if i % (num_clients/num_trainiers) == 0 {
            port += 1;
        }
        println!("Client {} connects to trainer at port {}", client_id, port);
        tasks.push(tokio::spawn(async move {
            let channel = tonic::transport::Channel::builder(local_uri)
                .connect()
                .await
                .expect("Connection to FlService failed");
            start_client(channel, client_id, model_id, true, port).await;
        }));
    }
    let channel = tonic::transport::Channel::builder(uri)
        .connect()
        .await
        .expect("Connection to FlService failed");
    println!("Client {} connects to trainer at port {}", start_id, trainer_port);
    start_client(channel, start_id, model_id, true, trainer_port).await;
    for task in tasks {
        let _ou = task.await;
    }
    Ok(())
}
