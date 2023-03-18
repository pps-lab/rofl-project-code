use bulletproofs::ProofError;
use bulletproofs::RangeProof;
use bulletproofs::{BulletproofGens, PedersenGens};
use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoPoint};
use curve25519_dalek::scalar::Scalar;
use merlin::Transcript;
use rayon::prelude::*;

pub mod errors;
use self::errors::RangeProofError;
use crate::conversion32::{f32_to_scalar, get_clip_bounds};
use crate::fp::{read_from_bytes, Fix, URawFix};
use crate::pedersen_ops::compute_shifted_values_rp;

/// prove that value x is element of [-2^((prove_range-1)/n_frac), 2^((prove_range-1)/n_frac)]
pub fn create_rangeproof(
    value_vec_clipped: &Vec<f32>,
    blinding_vec: &Vec<Scalar>,
    prove_range: usize,
    n_partition: usize,
) -> Result<(Vec<RangeProof>, Vec<RistrettoPoint>), RangeProofError> {
    if value_vec_clipped.len() != blinding_vec.len() {
        return Err(ProofError::WrongNumBlindingFactors.into());
    }

    // TODO mlei: checkbounds
    if is_out_of_range(&value_vec_clipped, prove_range) {
        return Err(RangeProofError::ValueOutOfRangeError);
    }

    // upshift values
    // let offset_value_fp: Fix = Fix::from_bits((1 as URawFix) << (prove_range-1));
    // let value_vec_clipped_shifted: Vec<f32> = compute_shifted_values_vec(&value_vec_clipped, &offset_value_fp.to_float());
    // let value_fp_vec_clipped_shifted: Vec<URawFix> = f32_to_fp_vec(&value_vec_clipped_shifted);

    let offset_value_scalar: Scalar = Scalar::from((1 as URawFix) << (prove_range - 1));
    // I LOVE ITERATORS!!!
    let value_fp_vec_clipped_shifted: Vec<URawFix> = value_vec_clipped
        .par_iter()
        .map(|x| f32_to_scalar(x) + &offset_value_scalar)
        .map(|x| x.to_bytes())
        .map(|x| read_from_bytes(&x))
        .collect();

    // extend vector length to pow2
    let value_fp_vec_shifted_clipped_ext: Vec<u64> =
        extend_vec_to_pow2(&value_fp_vec_clipped_shifted, 0)
            .iter()
            .map(|x| *x as u64)
            .collect();
    let blinding_vec_ext: Vec<Scalar> = extend_vec_to_pow2(&blinding_vec, Scalar::zero());

    // chunk the vector for par_iter
    let n_chunks: usize = std::cmp::min(value_fp_vec_shifted_clipped_ext.len(), n_partition);
    let chunk_size: usize = value_fp_vec_shifted_clipped_ext.len() / n_chunks;
    let value_fp_vec_chunks: Vec<Vec<u64>> = value_fp_vec_shifted_clipped_ext
        .chunks(chunk_size)
        .map(|x| x.to_vec())
        .collect();
    let blinding_fp_vec_chunks: Vec<Vec<Scalar>> = blinding_vec_ext
        .chunks(chunk_size)
        .map(|x| x.to_vec())
        .collect();
    let proof_args: Vec<(&Vec<u64>, &Vec<Scalar>)> = value_fp_vec_chunks
        .iter()
        .zip(&blinding_fp_vec_chunks)
        .collect();

    let pc_gens = PedersenGens::default();
    let res_vec: Vec<Result<(RangeProof, Vec<CompressedRistretto>), ProofError>> = proof_args
        .par_iter()
        .map(|(v, b)| create_rangeproof_helper(v, b, prove_range, &pc_gens))
        .collect();

    // bundle up results
    let mut res_range_proof_vec: Vec<RangeProof> = Vec::with_capacity(n_chunks);
    let mut res_commit_vec: Vec<CompressedRistretto> =
        Vec::with_capacity(value_fp_vec_shifted_clipped_ext.len());
    for r in res_vec {
        match r {
            Ok((rp, cpr_vec)) => {
                res_range_proof_vec.push(rp);
                res_commit_vec.extend(cpr_vec.iter());
            }
            Err(e) => return Err(e.into()),
        }
    }

    // downshift values
    let rp_vec_shifted = crp_to_rp_vec(&res_commit_vec);
    let inv_offset_rp: RistrettoPoint = pc_gens.commit(-offset_value_scalar, Scalar::zero());
    let rp_vec =
        compute_shifted_values_rp(&rp_vec_shifted[0..value_vec_clipped.len()], &inv_offset_rp);

    Ok((res_range_proof_vec, rp_vec))
}

pub fn clip_f32_to_range_vec(value_vec: &Vec<f32>, prove_range: usize) -> Vec<f32> {
    let (min_val, max_val) = get_clip_bounds(prove_range);
    let value_vec_clipped: Vec<f32> = value_vec
        .iter()
        .map(|x| f32::min(max_val, f32::max(min_val, *x)))
        .collect();
    value_vec_clipped
}

fn is_out_of_range(value_vec: &Vec<f32>, prove_range: usize) -> bool {
    let (min, max) = get_clip_bounds(prove_range);
    value_vec.iter().any(|x| (&min > x) || (x > &max))
}

fn create_rangeproof_helper(
    value_vec: &[u64],
    blinding_vec: &[Scalar],
    prove_range: usize,
    pc_gens: &PedersenGens,
) -> Result<(RangeProof, Vec<CompressedRistretto>), ProofError> {
    let bp_gens = BulletproofGens::new(64, value_vec.len());
    let mut transcript = Transcript::new(b"RangeProof");
    match RangeProof::prove_multiple(
        &bp_gens,
        &pc_gens,
        &mut transcript,
        &value_vec,
        &blinding_vec,
        prove_range,
    ) {
        Ok((range_proof, commit_vec)) => Ok((range_proof, commit_vec)),
        Err(e) => match e {
            ProofError::InvalidBitsize => Err(e),
            _ => panic!("Should not get here: {}", e),
        },
    }
}

/// converts sym range given by client into corresponding range for bulletproof
pub fn convert_sym_range_to_asym_fp_range(range_exp: usize) -> usize {
    range_exp + (Fix::frac_nbits() as usize)
}

pub fn verify_rangeproof(
    range_proof_vec: &Vec<RangeProof>,
    commit_vec: &Vec<RistrettoPoint>,
    prove_range: usize,
) -> Result<bool, ProofError> {
    // shift up
    let pc_gens = PedersenGens::default();
    let offset_value_fp: Fix = Fix::from_bits((1 as URawFix) << (prove_range - 1));
    let offset_rp: RistrettoPoint =
        pc_gens.commit(Scalar::from(offset_value_fp.to_bits()), Scalar::zero());
    let commit_vec_shifted: Vec<RistrettoPoint> =
        compute_shifted_values_rp(&commit_vec, &offset_rp);

    // extend to pow2
    let rp_zero: RistrettoPoint = pc_gens.commit(Scalar::zero(), Scalar::zero());
    let commit_vec_shifted_ext: Vec<RistrettoPoint> =
        extend_vec_to_pow2(&commit_vec_shifted, rp_zero);

    let crp_vec_shifted_ext: Vec<CompressedRistretto> = rp_to_crp_vec(&commit_vec_shifted_ext);
    let chunk_size: usize = crp_vec_shifted_ext.len() / range_proof_vec.len();
    let commit_vec_vec_shifted: Vec<Vec<CompressedRistretto>> = crp_vec_shifted_ext
        .chunks(chunk_size)
        .map(|x| x.to_vec())
        .collect();
    let verify_args: Vec<(&RangeProof, &Vec<CompressedRistretto>)> = range_proof_vec
        .iter()
        .zip(&commit_vec_vec_shifted)
        .collect();

    let res_vec: Vec<Result<bool, ProofError>> = verify_args
        .par_iter()
        .map(|(rp, crp_chunk)| verify_rangeproof_helper(rp, crp_chunk, prove_range, &pc_gens))
        .collect();

    let mut res: bool = true;
    for r in res_vec {
        match r {
            Ok(v) => res &= v,
            Err(e) => return Err(e),
        }
    }
    Ok(res)
}

pub fn verify_rangeproof_helper(
    range_proof: &RangeProof,
    commit_vec: &Vec<CompressedRistretto>,
    range_exp: usize,
    pc_gens: &PedersenGens,
) -> Result<bool, ProofError> {
    let bp_gens = BulletproofGens::new(64, commit_vec.len());
    let verify_range: usize = range_exp;
    let mut transcript = Transcript::new(b"RangeProof");

    match range_proof.verify_multiple(
        &bp_gens,
        &pc_gens,
        &mut transcript,
        &commit_vec,
        verify_range,
    ) {
        Ok(_) => Ok(true),
        Err(e) => match e {
            ProofError::VerificationError => Ok(false),
            _ => Err(e), //Err(_) => panic!("Should not get here")
        },
    }
}

fn extend_vec_to_pow2<T: Clone>(value_vec: &Vec<T>, fill_value: T) -> Vec<T> {
    let ext_len: usize = next_pow2(value_vec.len());
    let mut value_vec_ext: Vec<T> = vec![fill_value; ext_len];
    value_vec_ext[0..(value_vec.len())].clone_from_slice(&value_vec);
    value_vec_ext
}

fn next_pow2(val: usize) -> usize {
    // bit magic
    if val == 1 {
        return 1;
    }
    let mut n: usize = val - 1;
    while (n & n - 1) != 0 {
        n &= n - 1;
    }
    return n << 1;
}

fn rp_to_crp_vec(rp_vec: &Vec<RistrettoPoint>) -> Vec<CompressedRistretto> {
    rp_vec.par_iter().map(|x| x.compress()).collect()
}

fn crp_to_rp_vec(crp_vec: &Vec<CompressedRistretto>) -> Vec<RistrettoPoint> {
    crp_vec
        .par_iter()
        .map(|x| x.decompress().unwrap())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversion32::scalar_to_f32_vec;
    use crate::fp::N_BITS;
    use crate::pedersen_ops::*;
    use crate::range_proof_vec::clip_f32_to_range_vec;
    use rand::Rng;
    use std::cmp;
    use std::ops::Range;

    #[test]
    fn test_next_pow2() {
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(127), 128);
        assert_eq!(next_pow2(1usize << 31), 1usize << 31);
    }

    #[test]
    fn test_rangeproof_roundtrip() {
        let prove_range: usize = 4;
        let values: Vec<f32> = clip_f32_to_range_vec(
            &vec![-1.25, 0.5, -(Fix::max_value().to_float::<f32>())],
            prove_range,
        );
        let blindings: Vec<Scalar> = rnd_scalar_vec(values.len());
        let (range_proof_vec, commit_vec) =
            create_rangeproof(&values, &blindings, N_BITS, prove_range).unwrap();
        assert!(verify_rangeproof(&range_proof_vec, &commit_vec, N_BITS).unwrap());
    }

    #[test]
    fn test_fake_proof() {
        let prove_range: usize = 4;
        let values: Vec<f32> = clip_f32_to_range_vec(&vec![0.5], prove_range);
        let blindings: Vec<Scalar> = rnd_scalar_vec(values.len());
        let (range_proof_vec, _) =
            create_rangeproof(&values, &blindings, N_BITS, prove_range).unwrap();

        let fake_value: u64 = 1u64 << N_BITS + 1;
        let mut rng = rand::thread_rng();
        let fake_blinding: Scalar = Scalar::random(&mut rng);
        let pcs = PedersenGens::default();
        let fake_commit: RistrettoPoint = pcs.commit(Scalar::from(fake_value), fake_blinding);
        assert!(!verify_rangeproof(&range_proof_vec, &vec![fake_commit], N_BITS).unwrap());
    }

    #[test]
    fn test_rangeproof_par_roundtrip() {
        let n_values: usize = 100;
        let n_partition: usize = 4;

        let mut rng = rand::thread_rng();
        let prove_range: usize = N_BITS;
        let (fp_min, fp_max) = get_clip_bounds(prove_range);

        let value_vec: Vec<f32> = clip_f32_to_range_vec(
            &(0..n_values)
                .map(|_| rng.gen_range::<f32, Range<f32>>(fp_min..fp_max))
                .collect(),
            prove_range,
        );
        let blinding_vec: Vec<Scalar> = rnd_scalar_vec(n_values);

        let (rangeproof_vec, commit_vec): (Vec<RangeProof>, Vec<RistrettoPoint>) =
            create_rangeproof(&value_vec, &blinding_vec, prove_range, n_partition).unwrap();
        assert!(verify_rangeproof(&rangeproof_vec, &commit_vec, prove_range).unwrap());
    }

    #[test]
    fn test_create_rangeproof_correct_shift() {
        let n_partition: usize = 4;
        let x_vec: Vec<f32> = clip_f32_to_range_vec(&vec![0.25, 1.25, -1.5], N_BITS);
        let blinding_vec: Vec<Scalar> = (0..x_vec.len()).map(|_| Scalar::zero()).collect();

        let (_, commit_vec): (Vec<RangeProof>, Vec<RistrettoPoint>) =
            create_rangeproof(&x_vec, &blinding_vec, N_BITS, n_partition).unwrap();
        let extracted_val_scalar: Vec<Scalar> = default_discrete_log_vec(&commit_vec);
        let extracted_val: Vec<f32> = scalar_to_f32_vec(&extracted_val_scalar);
        //println!("extracted_val {:?}", extracted_val);
        assert_eq!(x_vec.len(), extracted_val.len());
        for (s, t) in x_vec.iter().zip(&extracted_val) {
            assert_eq!(s, t);
        }
    }

    #[test]
    fn test_fake_par_proof() {
        let n_values: usize = 100;
        let n_partition: usize = 4;

        let mut rng = rand::thread_rng();
        // prove_range as to be at least 8 bit
        let prove_range: usize = cmp::max(N_BITS / 2, 8);
        //let float_range: f32 = 2f32.powi((range_exp-1) as i32) - 1f32/(Fix::frac_nbits() as f32);
        let float_range: f32 =
            Fix::from_bits(((1u128 << prove_range) - 1u128) as URawFix).to_float();
        let value_vec: Vec<f32> = clip_f32_to_range_vec(
            &(0..n_values)
                .map(|_| rng.gen_range::<f32, Range<f32>>(-float_range..float_range))
                .collect(),
            prove_range,
        );
        let blinding_vec: Vec<Scalar> = rnd_scalar_vec(n_values);

        let (rangeproof_vec, commit_vec_vec): (Vec<RangeProof>, Vec<RistrettoPoint>) =
            create_rangeproof(&value_vec, &blinding_vec, prove_range, n_partition).unwrap();

        let fake_value: Scalar = Scalar::from(1u64 << prove_range + 1);
        let mut rng = rand::thread_rng();
        let fake_blinding: Scalar = Scalar::random(&mut rng);

        let pcs = PedersenGens::default();
        let fake_commit: RistrettoPoint = pcs.commit(fake_value, fake_blinding);
        let mut fake_commit_vec = commit_vec_vec.clone();
        fake_commit_vec[0] = fake_commit;

        assert!(!verify_rangeproof(&rangeproof_vec, &fake_commit_vec, prove_range).unwrap());
    }

    #[test]
    fn test_rangeproof_with_cancelling_blindings() {
        let n_partition: usize = 4;
        let range_exp: usize = N_BITS;
        let x_vec: Vec<f32> = vec![0.25, 1.25, -1.5];
        let y_vec: Vec<f32> = vec![-0.75, 1.25, -2.0];
        let z_vec: Vec<f32> = vec![0.5, 1.25, -3.0];
        let target_vec: Vec<f32> = vec![0.0, 3.75, -6.5];

        let blindings: Vec<Vec<Scalar>> = generate_cancelling_scalar_vec(3, x_vec.len());
        let (range_proof_x, commitments_x) =
            create_rangeproof(&x_vec, &blindings[0], range_exp, n_partition).unwrap();
        let (range_proof_y, commitments_y) =
            create_rangeproof(&y_vec, &blindings[1], range_exp, n_partition).unwrap();
        let (range_proof_z, commitments_z) =
            create_rangeproof(&z_vec, &blindings[2], range_exp, n_partition).unwrap();

        assert!(verify_rangeproof(&range_proof_x, &commitments_x, range_exp).unwrap());
        assert!(verify_rangeproof(&range_proof_y, &commitments_y, range_exp).unwrap());
        assert!(verify_rangeproof(&range_proof_z, &commitments_z, range_exp).unwrap());
        let sum_vec_rp: Vec<RistrettoPoint> = add_rp_vec_vec(&vec![
            commitments_x.to_vec(),
            commitments_y.to_vec(),
            commitments_z.to_vec(),
        ]);
        let sum_vec_scalar = default_discrete_log_vec(&sum_vec_rp);
        let sum_vec = scalar_to_f32_vec(&sum_vec_scalar);

        for (s, t) in sum_vec.iter().zip(&target_vec) {
            assert_eq!(s, t);
        }
    }

    #[test]
    fn test_rangeproof_clipped() {
        let n_partition: usize = 2;
        let prove_range: usize = N_BITS;
        let (min_val, max_val) = get_clip_bounds(prove_range);
        let target = vec![min_val, max_val];
        let x_vec: Vec<f32> = vec![min_val - 2f32, max_val + 3f32];
        let blindings: Vec<Scalar> = (0..x_vec.len()).map(|_| Scalar::zero()).collect();
        let x_vec_clipped: Vec<f32> = clip_f32_to_range_vec(&x_vec, prove_range);
        let (_, commitments_x) =
            create_rangeproof(&x_vec_clipped, &blindings, prove_range, n_partition).unwrap();
        let x_clipped_scalar: Vec<Scalar> = default_discrete_log_vec(&commitments_x);
        let x_clipped_dlog: Vec<f32> = scalar_to_f32_vec(&x_clipped_scalar);
        for (t, xc) in target.iter().zip(&x_clipped_dlog) {
            assert_eq!(t, xc)
        }
    }
}
