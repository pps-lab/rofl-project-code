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

extern crate curve25519_dalek_ng;
use curve25519_dalek_ng::ristretto::RistrettoPoint;
use curve25519_dalek_ng::scalar::Scalar;
extern crate bulletproofs;
use bulletproofs::{PedersenGens, RangeProof};

use rofl_crypto::bsgs32::*;
use rofl_crypto::conversion32::*;
use rofl_crypto::fp::N_BITS;
use rofl_crypto::pedersen_ops::*;
use rofl_crypto::square_proof_vec::*;
use rofl_crypto::util::{create_bench_file, get_bench_dir};
use std::io::prelude::*;
use std::time::{Duration, Instant};

use rofl_crypto::square_proof::pedersen::SquareProofCommitments;
use rofl_crypto::square_proof::SquareProof;
use std::thread::sleep;
use rayon::prelude::*;

use rofl_crypto::bench_constants::{DIM, num_samples};


fn create_squareproof_bench_fn(bench: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let (fp_min, fp_max) = get_clip_bounds(N_BITS);
    let ped_gens: PedersenGens = PedersenGens::default();

    for d in DIM.iter() {
        let createproof_label: String = createproof_label(*d);
        let mut createproof_file = create_bench_file(&createproof_label);

        let value_vec: Vec<f32> = (0..*d)
            .map(|_| rng.gen_range(fp_min..fp_max))
            .collect();
        let blinding_vec: Vec<Scalar> = rnd_scalar_vec(*d);
        let random_sq_vec: Vec<Scalar> = rnd_scalar_vec(*d);
        let value_com_vec = value_vec.par_iter().zip(&blinding_vec)
            .map(|(m, r)| ped_gens.commit(f32_to_scalar(m), r.clone())).collect();

        println!("warming up...");
        let (randproof_vec, commit_vec_vec): (
            Vec<SquareProof>,
            Vec<SquareProofCommitments>,
        ) = create_l2rangeproof_vec_existing(&value_vec, value_com_vec, &blinding_vec, &random_sq_vec).unwrap();
        black_box(randproof_vec);
        black_box(commit_vec_vec);
        println!("sampling {} / dim: {}", num_samples, d);

        for i in 0..num_samples {
            let value_vec: Vec<f32> = (0..*d)
                .map(|_| rng.gen_range(fp_min..fp_max))
                .collect();
            let blinding_vec: Vec<Scalar> = rnd_scalar_vec(*d);
            let value_vec_com: Vec<RistrettoPoint> = value_vec.par_iter().zip(&blinding_vec)
                .map(|(m, r)| ped_gens.commit(f32_to_scalar(m), r.clone())).collect();

            println!("sample nr: {}", i);
            let createproof_now = Instant::now();
            let (randproof_vec, commit_vec_vec): (
                Vec<SquareProof>,
                Vec<SquareProofCommitments>,
            ) = create_l2rangeproof_vec_existing(&value_vec, value_vec_com, &blinding_vec, &random_sq_vec).unwrap();
            black_box(randproof_vec);
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

fn createproof_label(dim: usize) -> String {
    let t: DateTime<Local> = Local::now();
    format!(
        "create-squareproof-{:02}-{:05}-({})",
        N_BITS,
        dim,
        t.format("%Y-%m-%d-%H-%M-%S").to_string()
    )
}

fn verifyproof_label(dim: usize) -> String {
    let t: DateTime<Local> = Local::now();
    format!(
        "verify-squareproof-{:02}-{:05}-({})",
        N_BITS,
        dim,
        t.format("%Y-%m-%d-%H-%M-%S").to_string()
    )
}

benchmark_group!(
    create_squareproof_bench,
    create_squareproof_bench_fn
);
benchmark_main!(create_squareproof_bench);
