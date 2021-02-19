#[macro_use]
extern crate criterion;
use criterion::Criterion;

extern crate rand;
use rand::Rng;

extern crate chrono;
use chrono::prelude::*;

extern crate curve25519_dalek;
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;

use rust_crypto::conversion32::get_clip_bounds;
use rust_crypto::fp::N_BITS;
use rust_crypto::pedersen_ops::rnd_scalar_vec;
use rust_crypto::rand_proof::{ElGamalPair, RandProof};
use rust_crypto::rand_proof_vec::{create_randproof_vec, verify_randproof_vec};

static DIM: [usize; 6] = [1024, 2048, 4096, 8192, 16384, 32768];

fn bench_randproof_create_proof(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let (fp_min, fp_max) = get_clip_bounds(N_BITS);
    for d in DIM.iter() {
        let value_vec: Vec<f32> = (0..*d)
            .map(|_| rng.gen_range::<f32>(fp_min, fp_max))
            .collect();
        let blinding_vec: Vec<Scalar> = rnd_scalar_vec(*d);
        let label: String = label_randproof_create_proof(*d);

        c.bench_function(&label, move |b| {
            b.iter(|| {
                let (randproof_vec, commit_vec_vec): (Vec<RandProof>, Vec<ElGamalPair>) =
                    create_randproof_vec(&value_vec, &blinding_vec).unwrap();
            })
        });
    }
}

fn bench_randproof_verify_proof(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let (fp_min, fp_max) = get_clip_bounds(N_BITS);
    for d in DIM.iter() {
        let value_vec: Vec<f32> = (0..*d)
            .map(|_| rng.gen_range::<f32>(fp_min, fp_max))
            .collect();
        let blinding_vec: Vec<Scalar> = rnd_scalar_vec(*d);
        let label: String = label_randproof_verify_proof(*d);
        let (randproof_vec, commit_vec_vec): (Vec<RandProof>, Vec<ElGamalPair>) =
            create_randproof_vec(&value_vec, &blinding_vec).unwrap();
        c.bench_function(&label, move |b| {
            b.iter(|| {
                verify_randproof_vec(&randproof_vec, &commit_vec_vec).unwrap();
            })
        });
    }
}

fn label_randproof_create_proof(dim: usize) -> String {
    // (fp_bitsize-dim-(time))
    let t: DateTime<Local> = Local::now();
    format!(
        "bench_randproof_create_proof-{}-{}-({})",
        N_BITS,
        dim,
        t.format("%Y-%m-%d/%H:%M:%S").to_string()
    )
}

fn label_randproof_verify_proof(dim: usize) -> String {
    // (fp_bitsize-dim-(time))
    let t: DateTime<Local> = Local::now();
    format!(
        "bench_randproof_verify_proof-{}-{}-({})",
        N_BITS,
        dim,
        t.format("%Y-%m-%d/%H:%M:%S").to_string()
    )
}

criterion_group! {
    name = rand_proof;
    config = Criterion::default().sample_size(10);
    targets =
    bench_randproof_create_proof,
    bench_randproof_verify_proof
}
criterion_main!(rand_proof);
