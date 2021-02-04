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

use rust_crypto::bsgs32::*;
use rust_crypto::conversion32::*;
use rust_crypto::fp::N_BITS;
use rust_crypto::pedersen_ops::*;
use rust_crypto::rand_proof::{ElGamalPair, RandProof};
use rust_crypto::rand_proof_vec::*;
use rust_crypto::range_proof_vec::*;
use std::env;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use std::thread::sleep;

static DIM: [usize; 4] = [32768, 131072, 262144, 524288];
static num_samples: usize = 4;

fn bench_paper_randproof_fn(bench: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let (fp_min, fp_max) = get_clip_bounds(N_BITS);

    for d in DIM.iter() {
        let createproof_label: String = createproof_label(*d);
        let mut createproof_file = create_bench_file(&createproof_label);

        let verifyproof_label: String = verifyproof_label(*d);
        let mut verifyproof_file = create_bench_file(&verifyproof_label);

        let value_vec: Vec<f32> = (0..*d)
            .map(|_| rng.gen_range::<f32>(fp_min, fp_max))
            .collect();
        let blinding_vec: Vec<Scalar> = rnd_scalar_vec(*d);
        println!("warming up...");
        let (randproof_vec, commit_vec_vec): (Vec<RandProof>, Vec<ElGamalPair>) =
            create_randproof_vec(&value_vec, &blinding_vec).unwrap();
        verify_randproof_vec(&randproof_vec, &commit_vec_vec).unwrap();
        println!("sampling {} / dim: {}", num_samples, d);

        for i in 0..num_samples {
            let value_vec: Vec<f32> = (0..*d)
                .map(|_| rng.gen_range::<f32>(fp_min, fp_max))
                .collect();
            let blinding_vec: Vec<Scalar> = rnd_scalar_vec(*d);

            println!("sample nr: {}", i);
            let createproof_now = Instant::now();
            let (randproof_vec, commit_vec_vec): (Vec<RandProof>, Vec<ElGamalPair>) =
                create_randproof_vec(&value_vec, &blinding_vec).unwrap();
            let create_elapsed = createproof_now.elapsed().as_millis();
            println!("createproof elapsed: {}", create_elapsed.to_string());
            createproof_file.write_all(create_elapsed.to_string().as_bytes());
            createproof_file.write_all(b"\n");
            createproof_file.flush();
            let verify_now = Instant::now();
            verify_randproof_vec(&randproof_vec, &commit_vec_vec).unwrap();
            let verify_elapsed = verify_now.elapsed().as_millis();
            println!("verifyproof elapsed: {}", verify_elapsed.to_string());
            verifyproof_file.write_all(verify_elapsed.to_string().as_bytes());
            verifyproof_file.write_all(b"\n");
            verifyproof_file.flush();
        }
    }
}

fn createproof_label(dim: usize) -> String {
    let t: DateTime<Local> = Local::now();
    format!(
        "create-paper-randproof-{:02}-{:05}-({})",
        N_BITS,
        dim,
        t.format("%Y-%m-%d-%H-%M-%S").to_string()
    )
}

fn verifyproof_label(dim: usize) -> String {
    let t: DateTime<Local> = Local::now();
    format!(
        "verify-paper-randproof-{:02}-{:05}-({})",
        N_BITS,
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

benchmark_group!(paper_randproof_bench, bench_paper_randproof_fn);
benchmark_main!(paper_randproof_bench);
