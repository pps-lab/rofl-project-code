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

use rust_crypto::fp::N_BITS;
use rust_crypto::pedersen_ops::*;
use rust_crypto::range_proof_vec::*;
use rust_crypto::bsgs32::*;
use rust_crypto::conversion32::*;
use std::fs::OpenOptions;
use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;
use std::env;
use std::time::{Duration, Instant};

use std::thread::sleep;


static DIM: [usize; 4] = [25000, 100000, 250000, 500000];
static num_samples: usize = 8;

fn bench_paper_addelgamalfunction(bench: &mut Bencher) {
    
    for d in DIM.iter() {

        let label: String = label_addelgamal(*d);
        let mut bench_file = create_bench_file(&label);

        let rnd_scalar_vec_vec: Vec<Vec<Scalar>> = (0..2).map(|_| rnd_scalar_vec(*d)).collect();
        let rnd_rp_vec_vec: Vec<Vec<RistrettoPoint>> = rnd_scalar_vec_vec.iter().map(|x| commit_no_blinding_vec(x)).collect();
        println!("warming up...");
        add_rp_vec_vec(&rnd_rp_vec_vec);
        println!("sampling {} / dim: {}", num_samples, *d);

        for i in 0..num_samples {
            let rnd_scalar_vec_vec: Vec<Vec<Scalar>> = (0..2).map(|_| rnd_scalar_vec(*d)).collect();
            let rnd_rp_vec_vec: Vec<Vec<RistrettoPoint>> = rnd_scalar_vec_vec.iter().map(|x| commit_no_blinding_vec(x)).collect();
            println!("sample nr: {}", i);
            let now = Instant::now();
            add_rp_vec_vec(&rnd_rp_vec_vec);
            let elapsed = now.elapsed().as_millis();
            println!("elapsed: {}", elapsed.to_string());
            bench_file.write_all(elapsed.to_string().as_bytes());
            bench_file.write_all(b"\n");
            bench_file.flush();
        }

    }
}

fn label_addelgamal(dim: usize) -> String{
    // (fp_bitsize-table_size-dim-(time))
    let t: DateTime<Local> = Local::now();
    format!("bench_paper_addelgamal-{:05}-({})",
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
    std::fs::create_dir_all(&bench_file);
    bench_file.push(label);
    bench_file.set_extension("bench");
    println!("bench file: {}", bench_file.display());
    let file = match OpenOptions::new().append(true).create(true).open(bench_file) {
        Err(err) => panic!("Could not find {}", err),
        Ok(f) => f
    };
    return file
}


benchmark_group!(bench_paper_addelgamal, bench_paper_addelgamalfunction);
benchmark_main!(bench_paper_addelgamal);