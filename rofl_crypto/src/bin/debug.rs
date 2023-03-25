

use criterion::black_box;

#[macro_use]
extern crate itertools;

extern crate rand;
use rand::Rng;

extern crate chrono;
use chrono::prelude::*;

extern crate curve25519_dalek;
use curve25519_dalek_ng::ristretto::RistrettoPoint;
use curve25519_dalek_ng::scalar::Scalar;
extern crate bulletproofs;
use bulletproofs::RangeProof;

use rofl_crypto::bsgs32::*;
use rofl_crypto::conversion32::*;
use rofl_crypto::fp::N_BITS;
use rofl_crypto::pedersen_ops::*;
use rofl_crypto::range_proof_vec::*;
use rofl_crypto::util::{create_bench_file, get_bench_dir};

use std::io::prelude::*;
use std::time::{Duration, Instant};

use std::thread::sleep;

// static DIM: [usize; 1] = [100];

use rofl_crypto::bench_constants::{DIM_RANGEPROOF, num_samples, RANGE, N_PARTITION_SMALL};


fn create_rangeproof_bench_fn() {
    let mut rng = rand::thread_rng();

    let range: Vec<&usize> = RANGE.into_iter().filter(|x| **x <= N_BITS).collect();
    for (r, d) in iproduct!(range, &DIM_RANGEPROOF) {
        let (fp_min, fp_max) = get_clip_bounds(*r);
        let createproof_label: String = createproof_label(*d, *r);
        let mut createproof_file = create_bench_file(&createproof_label);

        println!("warming up...");
        let mut value_vec: Vec<f32> = (0..*d)
            .map(|_| rng.gen_range(fp_min..fp_max))
            .collect();
        let mut blinding_vec: Vec<Scalar> = rnd_scalar_vec(*d);
        let (rangeproof_vec, commit_vec_vec): (Vec<RangeProof>, Vec<RistrettoPoint>) =
            create_rangeproof(&value_vec, &blinding_vec, black_box(*r), N_PARTITION_SMALL).unwrap();
        black_box(rangeproof_vec);
        black_box(commit_vec_vec);

        // verify_rangeproof(&rangeproof_vec, &commit_vec_vec, black_box(*r)).unwrap();
        println!("sampling {} / dim: {} / range: {}", num_samples, d, r);

        for i in 0..num_samples {
            value_vec = (0..*d)
                .map(|_| rng.gen_range(fp_min..fp_max))
                .collect();
            blinding_vec = rnd_scalar_vec(*d);

            println!("sample nr: {}", i);
            let createproof_now = Instant::now();
            let (rangeproof_vec, commit_vec_vec) =
                create_rangeproof(&value_vec, &blinding_vec, black_box(*r), N_PARTITION_SMALL).unwrap();
            black_box(rangeproof_vec);
            black_box(commit_vec_vec);
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


fn main() {
    create_rangeproof_bench_fn();

}

