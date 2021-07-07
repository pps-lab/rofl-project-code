pub mod client;
pub mod logs;
pub mod params;
pub mod server;
pub mod trainclient;
pub mod util;

pub mod flservice {
    tonic::include_proto!("flservice");
}
