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
use bulletproofs::{PedersenGens, RangeProof};

use rofl_crypto::bsgs32::*;
use rofl_crypto::conversion32::*;
use rofl_crypto::fp::N_BITS;
use rofl_crypto::pedersen_ops::*;
use rofl_crypto::rand_proof::{ElGamalPair, RandProof};
use rofl_crypto::rand_proof_vec::*;
use rofl_crypto::range_proof_vec::*;
use rofl_crypto::util::{create_bench_file, get_bench_dir};

use std::io::prelude::*;
use std::time::{Duration, Instant};

use std::thread::sleep;
use merlin::Transcript;
use rayon::prelude::*;
use rofl_crypto::compressed_rand_proof::{CompressedRandProof, ElGamalGens};
use rofl_crypto::compressed_rand_proof::types::CompressedRandProofCommitments;

// static DIM: [usize; 1] = [100];

use rofl_crypto::bench_constants::{DIM, num_samples};


fn create_compressedrandproof_bench_fn(bench: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let (fp_min, fp_max) = get_clip_bounds(N_BITS);
    let eg_gens = ElGamalGens::default();
    let ped_gens = PedersenGens::default();

    for d in DIM.iter() {
        let createproof_label: String = createproof_label(*d);
        let mut createproof_file = create_bench_file(&createproof_label);

        let value_vec: Vec<f32> = (0..*d)
            .map(|_| rng.gen_range(fp_min..fp_max))
            .collect();
        let blinding_vec: Vec<Scalar> = rnd_scalar_vec(*d);
        println!("warming up...");
        let mut prove_transcript = Transcript::new(b"CompressedRandProof");
        let value_scalar_vec = f32_to_scalar_vec(&value_vec);
        let value_vec_com: Vec<RistrettoPoint> = value_scalar_vec.par_iter().zip(&blinding_vec)
            .map(|(m, r)| ped_gens.commit(m.clone(), r.clone())).collect();
        let (randproof, commit_vec): (CompressedRandProof, CompressedRandProofCommitments) =
            CompressedRandProof::prove_existing(&eg_gens, &mut prove_transcript, value_scalar_vec, value_vec_com, blinding_vec).unwrap();
        black_box(randproof);
        black_box(commit_vec);
        println!("sampling {} / dim: {}", num_samples, d);

        for i in 0..num_samples {
            let value_vec: Vec<f32> = (0..*d)
                .map(|_| rng.gen_range(fp_min..fp_max))
                .collect();
            let blinding_vec: Vec<Scalar> = rnd_scalar_vec(*d);
            let value_scalar_vec = f32_to_scalar_vec(&value_vec);
            let value_vec_com: Vec<RistrettoPoint> = value_scalar_vec.par_iter().zip(&blinding_vec)
                .map(|(m, r)| ped_gens.commit(m.clone(), r.clone())).collect();

            println!("sample nr: {}", i);
            let createproof_now = Instant::now();

            let mut prove_transcript = Transcript::new(b"CompressedRandProof");
            let (randproof, commit_vec): (CompressedRandProof, CompressedRandProofCommitments) =
                CompressedRandProof::prove_existing(&eg_gens, &mut prove_transcript, value_scalar_vec, value_vec_com, blinding_vec).unwrap();
            black_box(randproof);
            black_box(commit_vec);
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
        "create-compressedrandproof-{:02}-{:05}-({})",
        N_BITS,
        dim,
        t.format("%Y-%m-%d-%H-%M-%S").to_string()
    )
}

fn verifyproof_label(dim: usize) -> String {
    let t: DateTime<Local> = Local::now();
    format!(
        "verify-compressedrandproof-{:02}-{:05}-({})",
        N_BITS,
        dim,
        t.format("%Y-%m-%d-%H-%M-%S").to_string()
    )
}

benchmark_group!(create_compressedrandproof_bench, create_compressedrandproof_bench_fn);
benchmark_main!(create_compressedrandproof_bench);
