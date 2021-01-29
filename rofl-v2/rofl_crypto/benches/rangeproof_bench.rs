#[macro_use]
extern crate criterion;
use criterion::Criterion;

#[macro_use]
extern crate itertools;

extern crate rand;
use rand::Rng;

extern crate curve25519_dalek;
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::ristretto::RistrettoPoint;

extern crate bulletproofs;
use bulletproofs::RangeProof;

extern crate chrono;
use chrono::prelude::*;

use rust_crypto::fp::N_BITS;
use rust_crypto::pedersen_ops::*;
use rust_crypto::range_proof_vec::*;
use rust_crypto::conversion32::get_clip_bounds;

// static DIM: [usize; 1] = [1024];
// static RANGE: [usize; 1] = [8];
//static DIM: [usize; 6] = [1024, 2048, 4096, 8192, 16384, 32768];
//static RANGE: [usize; 3] = [8, 16, 32];
static DIM: [usize; 1] = [32768];
static RANGE: [usize; 1] = [16];
static N_PARTITION: usize = 16;

fn bench_rangeproof_create_proof(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    
    let range: Vec<&usize> = RANGE.into_iter().filter(|x| **x <= N_BITS).collect();
    for (r, d) in iproduct!(range, &DIM) {
        let (fp_min, fp_max) = get_clip_bounds(*r);
        let value_vec: Vec<f32> = (0..*d).map(|_| rng.gen_range::<f32>(fp_min, fp_max)).collect();
        let blinding_vec: Vec<Scalar> = rnd_scalar_vec(*d);
        let label: String = label_rangeproof_create_proof(*r, *d);
        c.bench_function(
            &label, 
            move |b| b.iter(|| {
                let (rangeproof_vec, commit_vec_vec): (Vec<RangeProof>, Vec<RistrettoPoint>) = 
                    create_rangeproof(&value_vec, &blinding_vec, *r, N_PARTITION).unwrap();
            })
        );
    }
}

fn bench_rangeproof_verify_proof(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    
    let range: Vec<&usize> = RANGE.into_iter().filter(|x| **x <= N_BITS).collect();
    for (r, d) in iproduct!(range, &DIM) {
        let (fp_min, fp_max) = get_clip_bounds(*r);
        let value_vec: Vec<f32> = (0..*d).map(|_| rng.gen_range::<f32>(fp_min, fp_max)).collect();
        let blinding_vec: Vec<Scalar> = rnd_scalar_vec(*d);
        let label: String = label_rangeproof_verify_proof(*r, *d);
        let (rangeproof_vec, commit_vec_vec): (Vec<RangeProof>, Vec<RistrettoPoint>) = 
            create_rangeproof(&value_vec, &blinding_vec, *r, N_PARTITION).unwrap();
        c.bench_function(
            &label, 
            move |b| b.iter(|| {
                verify_rangeproof(&rangeproof_vec, &commit_vec_vec, *r).unwrap();
            })
        );
    }
}


fn label_rangeproof_create_proof(range: usize, dim: usize, ) -> String {
    // (range-fp_bitsize-dim-(time))
    let t: DateTime<Local> = Local::now();
    format!("bench_rangeproof_create_proof-{}-{}-{}-({})",
        range,
        N_BITS,
        dim,
        t.format("%Y-%m-%d/%H:%M:%S").to_string())
}

fn label_rangeproof_verify_proof(range: usize, dim: usize) -> String {
    // (range-fp_bitsize-dim-(time))
    let t: DateTime<Local> = Local::now();
    format!("bench_rangeproof_verify_proof-{}-{}-{}-({})",
        range,
        N_BITS,
        dim,
        t.format("%Y-%m-%d/%H:%M:%S").to_string())
}

criterion_group!{
    name = range_proof;
    config = Criterion::default().sample_size(10);
    targets = 
    bench_rangeproof_create_proof,
    //bench_rangeproof_verify_proof
}
criterion_main!(range_proof);