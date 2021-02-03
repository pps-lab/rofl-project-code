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
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::ristretto::RistrettoPoint;
extern crate bulletproofs;
use bulletproofs::RangeProof;


use rust_crypto::fp::N_BITS;
use rust_crypto::pedersen_ops::*;
use rust_crypto::l2_range_proof_vec::*;
use rust_crypto::fp::Fix;
use rust_crypto::bsgs32::*;
use rust_crypto::conversion32::*;
use std::fs::OpenOptions;
use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;
use std::env;
use std::time::{Duration, Instant};

use std::thread::sleep;
use reduce::Reduce;


static DIM: [usize; 4] = [32768, 131072, 262144, 524288];
static RANGE: [usize; 1] = [8];
static N_PARTITION: usize = 1;
static num_samples: usize = 4;

fn bench_paper_rangeproof_l2(bench: &mut Bencher) {

    let mut rng = rand::thread_rng();
    

    let range: Vec<&usize> = RANGE.into_iter().filter(|x| **x <= N_BITS).collect();
    for (r, d) in iproduct!(range, &DIM) {

        let mul_factor = (i32::pow(2, Fix::frac_nbits()) as f32);
        let l2_bound = get_l2_clip_bounds(*r);
        let elem = ((l2_bound) / (*d as f32)).sqrt() / mul_factor / 10.0;
        // let (fp_min, fp_max) = get_clip_bounds(*r);

        let (fp_min, fp_max) = (-elem, elem);
        println!("{:?} {:?} {:?}, {:?}", l2_bound, N_BITS, fp_min, fp_max);
        // let (fp_min, fp_max) = get_clip_bounds(*r);
        let createproof_label: String = createproof_label(*d, *r);
        let mut createproof_file = create_bench_file(&createproof_label);

        let verifyproof_label: String = verifyproof_label(*d, *r);
        let mut verifyproof_file = create_bench_file(&verifyproof_label);

        let x_vec: Vec<f32> = (0..*d).map(|_| rng.gen_range::<f32>(fp_min, fp_max)).collect();
        let x_vec_scalar: Vec<Scalar> = f32_to_scalar_vec(&x_vec);
        let x_vec_enc: Vec<RistrettoPoint> = commit_no_blinding_vec(&x_vec_scalar);
        println!("warming up...");
        let value_vec: Vec<f32> = (0..*d).map(|_| scalar_to_f32(&f32_to_scalar(&rng.gen_range::<f32>(fp_min, fp_max)))).collect();
        let value_clipped = clip_f32_to_range_vec(&value_vec, *r);
        if value_clipped != value_vec {
            println!("they are not equal!")
        }
        // let norm = value_vec.iter().map(|x| scalar_to_f32(&f32_to_scalar(x)))
        //     .map(|x| x * x * (i32::pow(2, Fix::frac_nbits()) as f32))
        //     .reduce(|a, b| a + b).unwrap() / l2_bound;
        // let value_clipped = value_vec.iter().map(|x| scalar_to_f32(&f32_to_scalar(&(x / norm)))).collect();
        let blinding_vec: Vec<Scalar> = rnd_scalar_vec(*d);
        let (rangeproof, commit_vec): (RangeProof, RistrettoPoint) =
        create_rangeproof_l2(&value_vec, &blinding_vec, black_box(*r), N_PARTITION).unwrap();
        verify_rangeproof_l2(rangeproof, commit_vec, black_box(*r)).unwrap();
        println!("sampling {} / dim: {} / range: {}", num_samples, d, r);

        for i in 0..num_samples {
            let value_vec: Vec<f32> = (0..*d).map(|_| rng.gen_range::<f32>(fp_min, fp_max)).collect();
            let blinding_vec: Vec<Scalar> = rnd_scalar_vec(*d);

            println!("sample nr: {}", i);
            let createproof_now = Instant::now();
            let (rangeproof, commit): (RangeProof, RistrettoPoint) =
            create_rangeproof_l2(&value_vec, &blinding_vec, black_box(*r), N_PARTITION).unwrap();
            let create_elapsed = createproof_now.elapsed().as_millis();
            println!("createproof elapsed: {}", create_elapsed.to_string());
            createproof_file.write_all(create_elapsed.to_string().as_bytes());
            createproof_file.write_all(b"\n");
            createproof_file.flush();
            let verify_now = Instant::now();
            verify_rangeproof_l2(rangeproof, commit, black_box(*r)).unwrap();
            let verify_elapsed = verify_now.elapsed().as_millis();
            println!("verifyproof elapsed: {}", verify_elapsed.to_string());
            verifyproof_file.write_all(verify_elapsed.to_string().as_bytes());
            verifyproof_file.write_all(b"\n");
            verifyproof_file.flush();
        }

    }
}

fn createproof_label(dim: usize, range: usize) -> String{
    let t: DateTime<Local> = Local::now();
    format!("create-paper-rangeproof-l2-{:02}-{:02}-{:05}-({})",
        N_BITS,
        range,
        dim,
        t.format("%Y-%m-%d-%H-%M-%S").to_string())
}

fn verifyproof_label(dim: usize, range: usize) -> String{
    let t: DateTime<Local> = Local::now();
    format!("verify-paper-rangeproof-l2-{:02}-{:02}-{:05}-({})",
        N_BITS,
        range,
        dim,
        t.format("%Y-%m-%d-%H-%M-%S").to_string())
}

fn get_bench_dir() -> PathBuf {
    let mut cwd = env::current_exe().unwrap();
    cwd.pop(); cwd.pop(); cwd.pop();
    cwd.push("criterion");
    cwd
}

fn create_bench_file(label: &String) -> File {
    let mut bench_file = get_bench_dir();
    //bench_file.push("asdf");
    bench_file.push(label);
    bench_file.set_extension("bench");
    println!("bench file: {}", bench_file.display());
    let file = match OpenOptions::new().append(true).create(true).open(bench_file) {
        Err(err) => panic!("Could not find {}", err),
        Ok(f) => f
    };
    return file
}


benchmark_group!(paper_l2rangeproof_bench, bench_paper_rangeproof_l2);
benchmark_main!(paper_l2rangeproof_bench);