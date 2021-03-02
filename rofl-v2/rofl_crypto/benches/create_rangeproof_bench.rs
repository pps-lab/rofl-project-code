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
extern crate bulletproofs;
use bulletproofs::RangeProof;

use rofl_crypto::bsgs32::*;
use rofl_crypto::conversion32::*;
use rofl_crypto::fp::N_BITS;
use rofl_crypto::pedersen_ops::*;
use rofl_crypto::range_proof_vec::*;
use rofl_crypto::util::{create_bench_file,get_bench_dir};

use std::io::prelude::*;
use std::time::{Duration, Instant};

use std::thread::sleep;

// static DIM: [usize; 1] = [100];

static DIM: [usize; 4] = [32768, 131072, 262144, 524288];
// static DIM: [usize; 4] = [25000, 100000, 250000, 500000];
static RANGE: [usize; 1] = [8];
static N_PARTITION: usize = 2;
static num_samples: usize = 4;

fn create_rangeproof_bench_fn(bench: &mut Bencher) {
    let mut rng = rand::thread_rng();

    let range: Vec<&usize> = RANGE.into_iter().filter(|x| **x <= N_BITS).collect();
    for (r, d) in iproduct!(range, &DIM) {
        let (fp_min, fp_max) = get_clip_bounds(*r);
        let createproof_label: String = createproof_label(*d, *r);
        let mut createproof_file = create_bench_file(&createproof_label);

        // let verifyproof_label: String = verifyproof_label(*d, *r);
        // let mut verifyproof_file = create_bench_file(&verifyproof_label);

        let x_vec: Vec<f32> = (0..*d)
            .map(|_| rng.gen_range::<f32>(fp_min, fp_max))
            .collect();
        let x_vec_scalar: Vec<Scalar> = f32_to_scalar_vec(&x_vec);
        let x_vec_enc: Vec<RistrettoPoint> = commit_no_blinding_vec(&x_vec_scalar);
        println!("warming up...");
        let value_vec: Vec<f32> = (0..*d)
            .map(|_| rng.gen_range::<f32>(fp_min, fp_max))
            .collect();
        let blinding_vec: Vec<Scalar> = rnd_scalar_vec(*d);
        let (rangeproof_vec, commit_vec_vec): (Vec<RangeProof>, Vec<RistrettoPoint>) =
            create_rangeproof(&value_vec, &blinding_vec, black_box(*r), N_PARTITION).unwrap();
        // verify_rangeproof(&rangeproof_vec, &commit_vec_vec, black_box(*r)).unwrap();
        println!("sampling {} / dim: {} / range: {}", num_samples, d, r);

        for i in 0..num_samples {
            let value_vec: Vec<f32> = (0..*d)
                .map(|_| rng.gen_range::<f32>(fp_min, fp_max))
                .collect();
            let blinding_vec: Vec<Scalar> = rnd_scalar_vec(*d);

            println!("sample nr: {}", i);
            let createproof_now = Instant::now();
            let (rangeproof_vec, commit_vec_vec): (Vec<RangeProof>, Vec<RistrettoPoint>) =
                create_rangeproof(&value_vec, &blinding_vec, black_box(*r), N_PARTITION).unwrap();
            let create_elapsed = createproof_now.elapsed().as_millis();
            println!("createproof elapsed: {}", create_elapsed.to_string());
            createproof_file.write_all(create_elapsed.to_string().as_bytes());
            createproof_file.write_all(b"\n");
            createproof_file.flush();
        }
    }
    std::process::exit(0); // Exit because we are actually using the benchmarking library wrong.
}

fn createproof_label(dim: usize, range: usize) -> String {
    let t: DateTime<Local> = Local::now();
    format!(
        "create-rangeproof-{:02}-{:02}-{:05}-({})",
        N_BITS,
        range,
        dim,
        t.format("%Y-%m-%d-%H-%M-%S").to_string()
    )
}

fn verifyproof_label(dim: usize, range: usize) -> String {
    let t: DateTime<Local> = Local::now();
    format!(
        "verify-rangeproof-{:02}-{:02}-{:05}-({})",
        N_BITS,
        range,
        dim,
        t.format("%Y-%m-%d-%H-%M-%S").to_string()
    )
}

benchmark_group!(
    create_rangeproof_bench,
    create_rangeproof_bench_fn
);
benchmark_main!(create_rangeproof_bench);