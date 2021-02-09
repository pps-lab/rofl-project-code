use log::*;

use flexi_logger::Logger;
use flexi_logger::writers::FileLogWriter;

pub const  BENCH_TAG: &'static str = "Bench";

// Configure a FileLogWriter for bench messages
pub fn bench_logger() -> Box<FileLogWriter> {
    Box::new(FileLogWriter::builder()
        .directory("benchlog")
        .discriminant(BENCH_TAG)
        .suffix("bench")
        .try_build()
        .unwrap())
}

// Define a macro for writing messages to the bench log and to the normal log
#[macro_use]
pub mod macros {
    #[macro_export]
    macro_rules! bench_info {
        ($($arg:tt)*) => (
            info!(target: "{Bench}", $($arg)*);
        )
    }
}