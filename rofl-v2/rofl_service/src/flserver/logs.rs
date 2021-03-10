use log::*;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use flexi_logger::writers::FileLogWriter;
use flexi_logger::DeferredNow;

pub const BENCH_TAG: &'static str = "Bench";

pub fn bench_format(
    w: &mut dyn std::io::Write,
    _now: &mut DeferredNow,
    record: &Record,
) -> Result<(), std::io::Error> {
    write!(w, "{}", &record.args())
}

// Configure a FileLogWriter for bench messages
pub fn bench_logger() -> Box<FileLogWriter> {
    Box::new(
        FileLogWriter::builder()
            .directory("benchlog")
            .format(bench_format)
            .discriminant(BENCH_TAG)
            .suffix("bench")
            .try_build()
            .unwrap(),
    )
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

#[derive(Clone)]
pub struct TimeState {
    instants: Arc<RwLock<Vec<Instant>>>,
}

impl TimeState {
    pub fn new() -> Self {
        TimeState {
            instants: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn record_instant(&self) {
        let ts = Instant::now();
        let ts_list_arc = Arc::clone(&self.instants);
        let mut ts_list_mut = ts_list_arc.write().unwrap();
        ts_list_mut.push(ts);
    }

    pub fn reset(&self) {
        let ts_list_arc = Arc::clone(&self.instants);
        let mut ts_list_mut = ts_list_arc.write().unwrap();
        ts_list_mut.clear();
    }

    pub fn log_bench_times(&self, round_id: i32) {
        let ts_list_arc = Arc::clone(&self.instants);
        let ts_list = ts_list_arc.read().unwrap();
        let mut out = String::new();
        let mut sum = 0;
        &ts_list[0..(ts_list.len() - 1)]
            .iter()
            .zip(&ts_list[1..ts_list.len()])
            .for_each(|(elem1, elem2)| {
                let millis = elem2.duration_since(*elem1).as_millis();
                sum += millis;
                out.push_str(millis.to_string().as_str());
                out.push_str(", ");
            });
        out.push_str(sum.to_string().as_str());
        bench_info!("{}, {}", round_id, out);
    }

    pub fn log_bench_times_with_bandwith(&self, round_id: i32, received: usize, sent: usize) {
        let ts_list_arc = Arc::clone(&self.instants);
        let ts_list = ts_list_arc.read().unwrap();
        let mut out = String::new();
        let mut sum = 0;
        &ts_list[0..(ts_list.len() - 1)]
            .iter()
            .zip(&ts_list[1..ts_list.len()])
            .for_each(|(elem1, elem2)| {
                let millis = elem2.duration_since(*elem1).as_millis();
                sum += millis;
                out.push_str(millis.to_string().as_str());
                out.push_str(", ");
            });
        out.push_str(sum.to_string().as_str());
        bench_info!("{}, {}, {}, {}", round_id, out, received, sent);
    }
}
