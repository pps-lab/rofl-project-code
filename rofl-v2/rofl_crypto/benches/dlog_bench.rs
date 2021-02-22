#![feature(duration_as_u128)]
#[macro_use]
extern crate bencher;
use bencher::Bencher;
use criterion::black_box;

#[macro_use]
extern crate itertools;

extern crate rand;
use rand::Rng;

extern crate chrono;
use chrono::prelude::*;

extern crate curve25519_dalek;
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;

use rofl_crypto::bsgs32::*;
use rofl_crypto::conversion32::*;
use rofl_crypto::fp::N_BITS;
use rofl_crypto::pedersen_ops::*;
use rofl_crypto::range_proof_vec::*;
use rofl_crypto::util::{create_bench_file,get_bench_dir};

use std::io::prelude::*;
use std::time::{Duration, Instant};

use std::thread::sleep;

static DIM: [usize; 4] = [32768, 131072, 262144, 524288];
static TABLE_SIZE: [usize; 1] = [16];
static num_samples: usize = 4;

fn bench_solve_discrete_log2_fn(bench: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let (fp_min, fp_max) = get_clip_bounds(N_BITS);

    let table_size: Vec<&usize> = TABLE_SIZE.iter().filter(|x| **x <= N_BITS).collect();

    for (d, ts) in iproduct!(&DIM, table_size) {
        let label: String = label_solve_discrete_log(*d, *ts);
        let mut bench_file = create_bench_file(&label);

        let x_vec: Vec<f32> = (0..*d)
            .map(|_| rng.gen_range::<f32>(fp_min, fp_max))
            .collect();
        let x_vec_scalar: Vec<Scalar> = f32_to_scalar_vec(&x_vec);
        let x_vec_enc: Vec<RistrettoPoint> = commit_no_blinding_vec(&x_vec_scalar);
        println!("warming up...");
        discrete_log_vec(&x_vec_enc, *ts);
        println!("sampling {} / dim: {} / table_size: {}", num_samples, d, ts);

        for i in 0..num_samples {
            let x_vec: Vec<f32> = (0..*d)
                .map(|_| rng.gen_range::<f32>(fp_min, fp_max))
                .collect();
            let x_vec_scalar: Vec<Scalar> = f32_to_scalar_vec(&x_vec);
            let x_vec_enc: Vec<RistrettoPoint> = commit_no_blinding_vec(&x_vec_scalar);

            println!("sample nr: {}", i);
            let now = Instant::now();
            discrete_log_vec(black_box(&x_vec_enc), black_box(*ts));
            let elapsed = now.elapsed().as_millis();
            println!("elapsed: {}", elapsed.to_string());
            bench_file.write_all(elapsed.to_string().as_bytes());
            bench_file.write_all(b"\n");
            bench_file.flush();
        }
    }

    std::process::exit(0); // Exit because we are actually using the benchmarking library wrong.

}

fn label_solve_discrete_log(dim: usize, table_size: usize) -> String {
    // (fp_bitsize-table_size-dim-(time))
    let t: DateTime<Local> = Local::now();
    format!(
        "bench_paper_dlog2-{:02}-{:02}-{:05}-({})",
        N_BITS,
        table_size,
        dim,
        t.format("%Y-%m-%d-%H-%M-%S").to_string()
    )
}

benchmark_group!(dlog_bench, bench_solve_discrete_log2_fn);
benchmark_main!(dlog_bench);
