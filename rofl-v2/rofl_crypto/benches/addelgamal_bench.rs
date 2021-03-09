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
use rofl_crypto::util::{create_bench_file, get_bench_dir};

use std::io::prelude::*;
use std::time::{Duration, Instant};

use std::thread::sleep;

static DIM: [usize; 4] = [32768, 131072, 262144, 524288];
static num_samples: usize = 4;

fn bench_addelgamal_fn(bench: &mut Bencher) {
    for d in DIM.iter() {
        let label: String = label_addelgamal(*d);
        let mut bench_file = create_bench_file(&label);

        let rnd_scalar_vec_vec: Vec<Vec<Scalar>> = (0..2).map(|_| rnd_scalar_vec(*d)).collect();
        let rnd_rp_vec_vec: Vec<Vec<RistrettoPoint>> = rnd_scalar_vec_vec
            .iter()
            .map(|x| commit_no_blinding_vec(x))
            .collect();
        println!("warming up...");
        add_rp_vec_vec(&rnd_rp_vec_vec);
        println!("sampling {} / dim: {}", num_samples, *d);

        for i in 0..num_samples {
            let rnd_scalar_vec_vec: Vec<Vec<Scalar>> = (0..2).map(|_| rnd_scalar_vec(*d)).collect();
            let rnd_rp_vec_vec: Vec<Vec<RistrettoPoint>> = rnd_scalar_vec_vec
                .iter()
                .map(|x| commit_no_blinding_vec(x))
                .collect();
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

    std::process::exit(0); // Exit because we are actually using the benchmarking library wrong.
}

fn label_addelgamal(dim: usize) -> String {
    // (fp_bitsize-table_size-dim-(time))
    let t: DateTime<Local> = Local::now();
    format!(
        "bench_paper_addelgamal-{:05}-({})",
        dim,
        t.format("%Y-%m-%d-%H-%M-%S").to_string()
    )
}

benchmark_group!(addelgamal_bench, bench_addelgamal_fn);
benchmark_main!(addelgamal_bench);
