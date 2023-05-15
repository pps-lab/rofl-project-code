use bulletproofs::ProofError;
use bulletproofs::RangeProof;
use bulletproofs::{BulletproofGens, PedersenGens};
use curve25519_dalek_ng::ristretto::{CompressedRistretto, RistrettoPoint};
use curve25519_dalek_ng::scalar::Scalar;
use merlin::Transcript;
use rayon::prelude::*;

pub mod errors;
use self::errors::L2RangeProofError;
use crate::conversion32::{f32_to_scalar, get_clip_bounds, get_l2_clip_bounds, scalar_to_f32};
use crate::fp::{read_from_bytes, Fix, URawFix, N_BITS};

/// prove that value x is element of [-2^((prove_range-1)/n_frac), 2^((prove_range-1)/n_frac)]
pub fn create_rangeproof_l2(
    value_vec_clipped: &Vec<f32>,
    blinding_vec: &Vec<Scalar>,
    prove_range: usize,
    n_partition: usize,
) -> Result<(RangeProof, RistrettoPoint), L2RangeProofError> {
    if value_vec_clipped.len() != blinding_vec.len() {
        return Err(ProofError::WrongNumBlindingFactors.into());
    }

    // TODO mlei: checkbounds
    if is_out_of_range(&value_vec_clipped, prove_range) {
        return Err(L2RangeProofError::ValueOutOfRangeError);
    }

    // upshift values
    // let offset_value_fp: Fix = Fix::from_bits((1 as URawFix) << (prove_range-1));
    // let value_vec_clipped_shifted: Vec<f32> = compute_shifted_values_vec(&value_vec_clipped, &offset_value_fp.to_float());
    // let value_fp_vec_clipped_shifted: Vec<URawFix> = f32_to_fp_vec(&value_vec_clipped_shifted);

    // println!("Offset scalar {:?} {:?}", Scalar::from((1 as URawFix)), offset_value_scalar);
    // TODO: Remove iterator in the code below for speed
    let val = value_vec_clipped
        .iter()
        .map(|x| f32_to_scalar(x))
        .map(|x| x * x)
        .reduce(|a, b| a + b)
        .unwrap(); // do we shift?

    let shift = i32::pow(2, Fix::frac_nbits()) as f32;
    let val_float = value_vec_clipped
        .iter()
        .map(|x| scalar_to_f32(&f32_to_scalar(x)))
        .map(|x| x * x * shift)
        .reduce(|a, b| a + b)
        .unwrap();

    // Check for overflows, strictly not required in proving but otherwise we prove bogus
    if (scalar_to_f32(&val) - val_float).abs() > f32::EPSILON {
        return Err(L2RangeProofError::OverflowError(
            scalar_to_f32(&val).to_string(),
            val_float.to_string(),
        ));
    }

    if scalar_to_f32(&val) > get_l2_clip_bounds(prove_range) {
        return Err(L2RangeProofError::NormOutOfRangeError(
            scalar_to_f32(&val).to_string(),
        ));
    }

    let value_shifted: Vec<Scalar> = vec![val];
    // let value_shifted: Vec<Scalar> = val.collect();

    let value_fp_vec_clipped_shifted: Vec<URawFix> = value_shifted
        .iter()
        .map(|x| x.to_bytes())
        .map(|x| read_from_bytes(&x))
        .collect();

    let blinding_sum = vec![blinding_vec
        .iter()
        .map(|x| x.clone())
        .reduce(|a, b| a + b)
        .unwrap()];

    assert_eq!(value_shifted.len(), blinding_sum.len());

    // extend vector length to pow2
    let value_fp_vec_shifted_clipped_ext: Vec<u64> =
        extend_vec_to_pow2(&value_fp_vec_clipped_shifted, 0)
            .iter()
            .map(|x| *x as u64)
            .collect();
    let blinding_vec_ext: Vec<Scalar> = extend_vec_to_pow2(&blinding_sum, Scalar::zero());

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
    // let rp_vec_shifted = crp_to_rp_vec(&res_commit_vec);
    // let inv_offset_rp: RistrettoPoint = pc_gens.commit(-offset_value_scalar, Scalar::zero());
    // let rp_vec = compute_shifted_values_rp(&rp_vec_shifted[0..value_shifted.len()], &inv_offset_rp);

    // TODO lhidde: Look into efficiency
    assert_eq!(res_range_proof_vec.len(), 1);
    assert_eq!(res_commit_vec.len(), 1);

    Ok((
        res_range_proof_vec[0].clone(),
        res_commit_vec[0].decompress().unwrap(),
    ))
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
    let mut transcript = Transcript::new(b"L2RangeProof");
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

pub fn verify_rangeproof_l2(
    range_proof: &RangeProof,
    commit: &RistrettoPoint,
    prove_range: usize,
) -> Result<bool, ProofError> {
    // TODO lhidde: Make efficient
    let range_proof_vec = vec![range_proof.clone()];
    let commit_vec = vec![commit.clone()];

    // shift up
    let pc_gens = PedersenGens::default();
    // let offset_value_fp: Fix = Fix::from_bits((1 as URawFix) << (prove_range-1));
    // let offset_rp: RistrettoPoint = pc_gens.commit(Scalar::from(offset_value_fp.to_bits()), Scalar::zero());
    // let commit_vec_shifted: Vec<RistrettoPoint> = compute_shifted_values_rp(&commit_vec, &offset_rp);

    // extend to pow2
    let rp_zero: RistrettoPoint = pc_gens.commit(Scalar::zero(), Scalar::zero());
    let commit_vec_shifted_ext: Vec<RistrettoPoint> = extend_vec_to_pow2(&commit_vec, rp_zero);

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
    let mut transcript = Transcript::new(b"L2RangeProof");

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
    use crate::square_rand_proof_vec;
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
    fn test_rangeproof_simple() {
        let prove_range: usize = 4;
        // let values: Vec<f32> =  clip_f32_to_range_vec(&vec![-1.25, 0.5, -(Fix::max_value().to_float::<f32>())], prove_range);
        let values: Vec<f32> = clip_f32_to_range_vec(&vec![1.25], prove_range);
        let blindings: Vec<Scalar> = rnd_scalar_vec(values.len());
        let (range_proof, commit) =
            create_rangeproof_l2(&values, &blindings, N_BITS, prove_range).unwrap();
        let res = verify_rangeproof_l2(&range_proof, &commit, N_BITS);
        assert!(res.unwrap());
    }

    #[test]
    fn test_rangeproof_roundtrip() {
        let prove_range: usize = 4;
        // let values: Vec<f32> =  clip_f32_to_range_vec(&vec![-1.25, 0.5, -(Fix::max_value().to_float::<f32>())], prove_range);
        let values: Vec<f32> = clip_f32_to_range_vec(&vec![1.25, 0.5, 0.25], prove_range);
        let blindings: Vec<Scalar> = rnd_scalar_vec(values.len());
        let (range_proof, commit) =
            create_rangeproof_l2(&values, &blindings, N_BITS, prove_range).unwrap();
        let res = verify_rangeproof_l2(&range_proof, &commit, N_BITS);
        assert!(res.unwrap());
    }

    #[test]
    fn test_rangeproof_bound_test() {
        // let num_frac = 5
        // let range: f32 = 1023.96875 * 2.0;
        // let
        let prove_range: usize = 32;
        // let values: Vec<f32> =  clip_f32_to_range_vec(&vec![-1.25, 0.5, -(Fix::max_value().to_float::<f32>())], prove_range);
        let values: Vec<f32> = vec![7.9];
        // let bb = get_l2_clip_bounds(prove_range);
        // let mul = f32_to_scalar(&values[0]) * f32_to_scalar(&values[0]);
        // println!("bb {:?} {:?}", bb, scalar_to_f32(&mul));
        let blindings: Vec<Scalar> = rnd_scalar_vec(values.len());
        let (range_proof, commit) =
            create_rangeproof_l2(&values, &blindings, prove_range, prove_range).unwrap();
        let res = verify_rangeproof_l2(&range_proof, &commit, prove_range);
        assert!(res.unwrap());
    }

    #[test]
    fn test_rangeproof_bound_negative() {
        let prove_range: usize = 32;
        let values: Vec<f32> = vec![-7.9];
        let blindings: Vec<Scalar> = rnd_scalar_vec(values.len());
        let (range_proof, commit) =
            create_rangeproof_l2(&values, &blindings, prove_range, prove_range).unwrap();
        let res = verify_rangeproof_l2(&range_proof, &commit, prove_range);
        assert!(res.unwrap());
    }

    #[test]
    fn test_rangeproof_bound_test_fail() {
        let prove_range: usize = 16;
        let values: Vec<f32> = vec![8.0];
        let blindings: Vec<Scalar> = rnd_scalar_vec(values.len());
        let res = create_rangeproof_l2(&values, &blindings, prove_range, prove_range);
        assert!(res.is_err());
    }

    #[test]
    fn test_rangeproof_bound_test_fail_two() {
        let prove_range: usize = 16;
        let values: Vec<f32> = vec![6.0, 6.0];
        let blindings: Vec<Scalar> = rnd_scalar_vec(values.len());
        let res = create_rangeproof_l2(&values, &blindings, prove_range, prove_range);
        assert!(res.is_err());
    }

    #[test]
    fn test_fake_proof() {
        let prove_range: usize = 4;
        let values: Vec<f32> = clip_f32_to_range_vec(&vec![0.5], prove_range);
        let blindings: Vec<Scalar> = rnd_scalar_vec(values.len());
        let (range_proof_vec, _) =
            create_rangeproof_l2(&values, &blindings, N_BITS, prove_range).unwrap();

        let fake_value: u64 = 1u64 << N_BITS + 1;
        let mut rng = rand::thread_rng();
        let fake_blinding: Scalar = Scalar::random(&mut rng);
        let pcs = PedersenGens::default();
        let fake_commit: RistrettoPoint = pcs.commit(Scalar::from(fake_value), fake_blinding);
        assert!(!verify_rangeproof_l2(&range_proof_vec, &fake_commit, N_BITS).unwrap());
    }

    #[test]
    fn test_rangeproof_par_roundtrip() {
        let n_values: usize = 100;
        let n_partition: usize = 4;

        let mut rng = rand::thread_rng();
        let prove_range: usize = N_BITS; // Because of the square
        let (fp_min, fp_max) = get_clip_bounds(8); // Otw out of bounds?
        println!("{} {} {}", prove_range, fp_min, fp_max);
        let value_vec: Vec<f32> = clip_f32_to_range_vec(
            &(0..n_values)
                .map(|_| rng.gen_range::<f32, Range<f32>>(fp_min..fp_max))
                .collect(),
            prove_range,
        );
        let blinding_vec: Vec<Scalar> = rnd_scalar_vec(n_values);

        let (rangeproof_vec, commit_vec) =
            create_rangeproof_l2(&value_vec, &blinding_vec, prove_range, n_partition).unwrap();
        assert!(verify_rangeproof_l2(&rangeproof_vec, &commit_vec, prove_range).unwrap());
    }

    #[test]
    fn test_create_rangeproof_correct_shift() {
        // Cant work, need cancelling blindings

        let n_partition: usize = 4;
        let x_vec: Vec<f32> = clip_f32_to_range_vec(&vec![0.25, 1.25, -1.5], N_BITS);
        let blinding_vec: Vec<Scalar> = (0..x_vec.len()).map(|_| Scalar::zero()).collect();

        let (_, commit_vec) =
            create_rangeproof_l2(&x_vec, &blinding_vec, N_BITS, n_partition).unwrap();
        let extracted_val_scalar: Vec<Scalar> = default_discrete_log_vec(&vec![commit_vec]);
        let extracted_val: Vec<f32> = scalar_to_f32_vec(&extracted_val_scalar);
        //println!("extracted_val {:?}", extracted_val);

        assert_eq!(
            extracted_val[0],
            3.875 * (i32::pow(2, Fix::frac_nbits()) as f32)
        );
    }

    #[test]
    fn test_fake_par_proof() {
        let n_values: usize = 100;
        let n_partition: usize = 4;

        let mut rng = rand::thread_rng();
        // prove_range as to be at least 8 bit
        let prove_range: usize = N_BITS;
        //let float_range: f32 = 2f32.powi((range_exp-1) as i32) - 1f32/(Fix::frac_nbits() as f32);
        let float_range: f32 =
            Fix::from_bits(((1u128 << prove_range) - 1u128) as URawFix).to_float();
        println!("float range {:?}", float_range);
        let value_vec: Vec<f32> = clip_f32_to_range_vec(
            &(0..n_values)
                .map(|_| rng.gen_range::<f32, Range<f32>>(-float_range..float_range))
                .collect(),
            prove_range,
        );
        let blinding_vec: Vec<Scalar> = rnd_scalar_vec(n_values);

        let (rangeproof_vec, commit_vec_vec): (RangeProof, RistrettoPoint) =
            create_rangeproof_l2(&value_vec, &blinding_vec, prove_range, n_partition).unwrap();

        let fake_value: Scalar = Scalar::from(1u64 << prove_range + 1);
        let mut rng = rand::thread_rng();
        let fake_blinding: Scalar = Scalar::random(&mut rng);

        let pcs = PedersenGens::default();
        let _fake_commit: RistrettoPoint = pcs.commit(fake_value, fake_blinding);
        let fake_commit_vec = commit_vec_vec.clone();

        assert!(!verify_rangeproof_l2(&rangeproof_vec, &fake_commit_vec, prove_range).unwrap());
    }

    #[test]
    fn test_rangeproof_with_cancelling_blindings() {
        let n_partition: usize = 4;
        let range_exp: usize = N_BITS;
        let x_vec: Vec<f32> = vec![0.25, 1.25, -1.5];
        let y_vec: Vec<f32> = vec![-0.75, 1.25, -2.0];
        let z_vec: Vec<f32> = vec![0.5, 1.25, -3.0];
        let correction = i32::pow(2, Fix::frac_nbits()) as f32;
        let target_vec: Vec<f32> =
            vec![3.875 * correction, 6.125 * correction, 10.8125 * correction];

        let zero: f32 = 0.0;

        // Swap so the vectors cancel out per rangeproof, just so we can compare later
        let blindings_tt: Vec<Vec<Scalar>> = generate_cancelling_scalar_vec(3, x_vec.len());
        let mut blindings: Vec<Vec<Scalar>> = vec![vec![f32_to_scalar(&zero); 3]; x_vec.len()];
        for (i, v) in blindings_tt.iter().enumerate() {
            for (j, s) in v.iter().enumerate() {
                blindings[j][i] = s.clone();
            }
        }

        let (range_proof_x, commitments_x) =
            create_rangeproof_l2(&x_vec, &blindings[0], range_exp, n_partition).unwrap();
        let (range_proof_y, commitments_y) =
            create_rangeproof_l2(&y_vec, &blindings[1], range_exp, n_partition).unwrap();
        let (range_proof_z, commitments_z) =
            create_rangeproof_l2(&z_vec, &blindings[2], range_exp, n_partition).unwrap();

        assert!(verify_rangeproof_l2(&range_proof_x, &commitments_x, range_exp).unwrap());
        assert!(verify_rangeproof_l2(&range_proof_y, &commitments_y, range_exp).unwrap());
        assert!(verify_rangeproof_l2(&range_proof_z, &commitments_z, range_exp).unwrap());
        let sum_vec_rp: Vec<RistrettoPoint> = vec![commitments_x, commitments_y, commitments_z];
        let sum_vec_scalar = default_discrete_log_vec(&sum_vec_rp);
        let sum_vec = scalar_to_f32_vec(&sum_vec_scalar);

        for (s, t) in sum_vec.iter().zip(&target_vec) {
            // println!("Expected: {:?}, actual {:?}", s, t);
            assert_eq!(s, t);
        }
    }

    #[test]
    fn test_rangeproof_clipped() {
        let n_partition: usize = 2;
        let prove_range: usize = N_BITS / 2;
        let (min_val, max_val) = get_clip_bounds(prove_range);
        let _target = vec![min_val, max_val];
        let x_vec: Vec<f32> = vec![min_val - 2f32, max_val + 3f32];
        let f0 = 0f32;
        assert_eq!(Scalar::zero(), f32_to_scalar(&f0));
        let blindings: Vec<Scalar> = (0..x_vec.len()).map(|_| Scalar::zero()).collect();
        let x_vec_clipped: Vec<f32> = clip_f32_to_range_vec(&x_vec, prove_range);
        // let x_vec_clipped_small: Vec<f32> = x_vec_clipped.iter().collect();
        println!("{:?} {:?} {:?} {:?}", x_vec_clipped, prove_range, min_val, max_val);

        let (_, commitments_x) =
            create_rangeproof_l2(&x_vec_clipped, &blindings, N_BITS, n_partition).unwrap();
        let x_clipped_scalar: Vec<Scalar> = default_discrete_log_vec(&vec![commitments_x]);
        let x_clipped_dlog: Vec<f32> = scalar_to_f32_vec(&x_clipped_scalar);

        // Incl correction
        let summed = x_vec_clipped
            .iter()
            .map(|x| x * x)
            .reduce(|a, b| a + b)
            .unwrap()
            * (i32::pow(2, Fix::frac_nbits()) as f32);
        assert_eq!(x_clipped_dlog[0], summed);
    }

    #[test]
    fn test_rangeproof_compare_randproof_sum() {
        let prove_range: usize = 4;
        // let values: Vec<f32> =  clip_f32_to_range_vec(&vec![-1.25, 0.5, -(Fix::max_value().to_float::<f32>())], prove_range);
        let values: Vec<f32> = clip_f32_to_range_vec(&vec![1.25, -0.5, 0.25], prove_range);
        let blindings_1: Vec<Scalar> = vec![Scalar::zero(); values.len()];
        let blindings_2: Vec<Scalar> = vec![Scalar::zero(); values.len()];
        let (_range_proof, commit) =
            create_rangeproof_l2(&values, &blindings_2, N_BITS, prove_range).unwrap();
        let (_rand_proof_vec, eg_pair_vec) =
            square_rand_proof_vec::create_l2rangeproof_vec(&values, &blindings_1, &blindings_2)
                .unwrap();

        let sum = eg_pair_vec
            .iter()
            .map(|x| x.c_sq)
            .reduce(|a, b| a + b)
            .unwrap();

        let x_clipped_scalar: Vec<Scalar> = default_discrete_log_vec(&vec![sum, commit]);
        let _x_clipped_dlog: Vec<f32> = scalar_to_f32_vec(&x_clipped_scalar);
        // println!("Values: {:?} {:?}", x_clipped_dlog[0], x_clipped_dlog[1]);
        assert_eq!(sum, commit);
    }

    #[test]
    fn test_clip_max_bounds() {
        let n_partition: usize = 2;
        let prove_range: usize = N_BITS / 2;
        println!("{:?} ", N_BITS);
        let (min_val, max_val) = get_clip_bounds(prove_range);

        let val = scalar_to_f32(&f32_to_scalar(&2000.0));
        let _target = vec![min_val, max_val];
        let x_vec: Vec<f32> = vec![min_val - 2f32, max_val + 3f32];
        let f0 = 0f32;
        assert_eq!(Scalar::zero(), f32_to_scalar(&f0));
        let blindings: Vec<Scalar> = (0..x_vec.len()).map(|_| Scalar::zero()).collect();
        let x_vec_clipped: Vec<f32> = clip_f32_to_range_vec(&x_vec, prove_range);
        // let x_vec_clipped_small: Vec<f32> = x_vec_clipped.iter().collect();
        println!("{:?} {:?} {:?} {:?}", val, prove_range, min_val, max_val);

    }
}
