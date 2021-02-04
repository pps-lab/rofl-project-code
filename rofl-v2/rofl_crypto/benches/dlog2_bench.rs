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

use rust_crypto::bsgs32::*;
use rust_crypto::conversion32::*;
use rust_crypto::fp::N_BITS;
use rust_crypto::pedersen_ops::*;
use rust_crypto::range_proof_vec::*;
use std::env;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use std::thread::sleep;

static DIM: [usize; 4] = [8192, 4096, 2048, 1024];
//static DIM: [usize; 6] = [32768, 16384, 8192, 4096, 2048, 1024];
//static DIM: [usize; 6] = [1024, 2048, 4096, 8192, 16384, 32768];
//static DIM: [usize; 1] = [32768];
static TABLE_SIZE: [usize; 9] = [8, 9, 10, 11, 12, 13, 14, 15, 16];
//static TABLE_SIZE: [usize; 9] = [16, 15, 14, 13, 12, 11, 10, 9, 8];
//static TABLE_SIZE: [usize; 5] = [12, 11, 10, 9, 8];
static num_samples: usize = 10;

fn bench_solve_discrete_log2(bench: &mut Bencher) {
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
}

fn label_solve_discrete_log(dim: usize, table_size: usize) -> String {
    // (fp_bitsize-table_size-dim-(time))
    let t: DateTime<Local> = Local::now();
    format!(
        "bench_dlog2-{:02}-{:02}-{:05}-({})",
        N_BITS,
        table_size,
        dim,
        t.format("%Y-%m-%d-%H-%M-%S").to_string()
    )
}

fn get_bench_dir() -> PathBuf {
    let mut cwd = env::current_exe().unwrap();
    cwd.pop();
    cwd.pop();
    cwd.pop();
    cwd.push("criterion");
    cwd
}

fn create_bench_file(label: &String) -> File {
    let mut bench_file = get_bench_dir();
    //bench_file.push("asdf");
    bench_file.push(label);
    bench_file.set_extension("bench");
    println!("bench file: {}", bench_file.display());
    let file = match OpenOptions::new()
        .append(true)
        .create(true)
        .open(bench_file)
    {
        Err(err) => panic!("Could not find {}", err),
        Ok(f) => f,
    };
    return file;
}

benchmark_group!(dlog2_bench, bench_solve_discrete_log2);
benchmark_main!(dlog2_bench);
