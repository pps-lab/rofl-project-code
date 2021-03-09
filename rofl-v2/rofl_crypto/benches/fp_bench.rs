#![feature(duration_as_u128)]
#[macro_use]
extern crate criterion;
use criterion::Criterion;

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
use rofl_crypto::util::{create_bench_file, get_bench_dir};

use std::io::prelude::*;
use std::time::{Duration, Instant};

use std::thread::sleep;

// static DIM: [usize; 6] = [32768, 16384, 8192, 4096, 2048, 1024];
static DIM: [usize; 6] = [1024, 2048, 4096, 8192, 16384, 32768];
//static DIM: [usize; 1] = [16384];
//static TABLE_SIZE: [usize; 9] = [8 , 9, 10, 11, 12, 13, 14, 15, 16];
//static TABLE_SIZE: [usize; 9] = [16, 15, 14, 13, 12, 11, 10, 9, 8];
//static TABLE_SIZE: [usize; 5] = [12, 11, 10, 9, 8];
static num_samples: usize = 100;

fn bench_convert_from_float_to_fixed(c: &mut Criterion) {
    let mut bench_file = create_bench_file(&"hello".to_string());
    // let now = Instant::now();

    // // we sleep for 2 seconds
    // sleep(Duration::new(2, 0));
    // // it prints '2'
    // let elapsed = now.elapsed().as_nanos();
    // println!("{}", elapsed);
    // bench_file.write_all(elapsed.to_string().as_bytes());
    // bench_file.write_all(b"\n");
    // bench_file.flush();

    // bench_file.write_all(b"hellloworldajds");
    // bench_file.write_all(b"\n");
    // bench_file.flush();

    let mut rng = rand::thread_rng();
    // for d in DIM.iter() {
    //     let vector<
    // }
    // let (fp_min, fp_max) = get_clip_bounds(N_BITS);

    // let table_size: Vec<&usize> = TABLE_SIZE.iter().filter(|x| **x <= N_BITS).collect();

    // for (ts, d) in iproduct!(table_size, &DIM) {
    //     let x_vec: Vec<f32> = (0..*d).map(|_| rng.gen_range::<f32>(fp_min, fp_max)).collect();
    //     let x_vec_scalar: Vec<Scalar> = f32_to_scalar_vec(&x_vec);
    //     let x_vec_enc: Vec<RistrettoPoint> = commit_no_blinding_vec(&x_vec_scalar);
    //     let label: String = label_solve_discrete_log(*d, *ts);
    //     c.bench_function(
    //         &label, move |b| b.iter(||
    //             discrete_log_vec(&x_vec_enc, *ts)));
    // }
}

fn bench_convert_from_fixed_to_float(c: &mut Criterion) {
    // let mut rng = rand::thread_rng();
    // let (fp_min, fp_max) = get_clip_bounds(N_BITS);

    // let table_size: Vec<&usize> = TABLE_SIZE.iter().filter(|x| **x <= N_BITS).collect();

    // for (ts, d) in iproduct!(table_size, &DIM) {
    //     let x_vec: Vec<f32> = (0..*d).map(|_| rng.gen_range::<f32>(fp_min, fp_max)).collect();
    //     let x_vec_scalar: Vec<Scalar> = f32_to_scalar_vec(&x_vec);
    //     let x_vec_enc: Vec<RistrettoPoint> = commit_no_blinding_vec(&x_vec_scalar);
    //     let label: String = label_solve_discrete_log(*d, *ts);
    //     c.bench_function(
    //         &label, move |b| b.iter(||
    //             discrete_log_vec(&x_vec_enc, *ts)));
    // }
}

fn label_solve_discrete_log(dim: usize, table_size: usize) -> String {
    // (fp_bitsize-table_size-dim-(time))
    let t: DateTime<Local> = Local::now();
    format!(
        "bench_discrete_log-{}-{}-{}-({})",
        N_BITS,
        table_size,
        dim,
        t.format("%Y-%m-%d/%H:%M:%S").to_string()
    )
}

criterion_group! {
    name = fp_conversion;
    config = Criterion::default().sample_size(2);
    targets =
    bench_convert_from_float_to_fixed,
    bench_convert_from_fixed_to_float,
}

criterion_main!(fp_conversion);
