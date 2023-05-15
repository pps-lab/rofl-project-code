use bulletproofs::PedersenGens;
use curve25519_dalek_ng::scalar::Scalar;
use merlin::Transcript;
use rayon::prelude::*;

use crate::conversion32::f32_to_scalar_vec;

use crate::rand_proof::ElGamalGens;
use crate::square_proof::ProofError;
use crate::square_proof::SquareProof;
pub mod errors;
pub use self::errors::L2RangeProofError;

use crate::square_proof::pedersen::SquareProofCommitments;

use curve25519_dalek_ng::ristretto::RistrettoPoint;
use itertools::Itertools;

pub fn create_l2rangeproof_vec_existing(
    value_vec: &Vec<f32>,
    value_com_vec: Vec<RistrettoPoint>,
    random_vec: &Vec<Scalar>,
    random_vec_2: &Vec<Scalar>,
) -> Result<(Vec<SquareProof>, Vec<SquareProofCommitments>), L2RangeProofError> {
    if value_vec.len() != random_vec.len() {
        return Err(L2RangeProofError::WrongNumBlindingFactors);
    }

    let value_scalar_vec: Vec<Scalar> = f32_to_scalar_vec(&value_vec);

    let ped_gens: PedersenGens = PedersenGens::default();

    // Concatenate togeter
    let randproof_args_1: Vec<(((&Scalar, &Scalar), &Scalar), &RistrettoPoint)> = value_scalar_vec
        .iter()
        .zip(random_vec)
        .zip(random_vec_2)
        .zip(&value_com_vec)
        .collect();

    // Add random vector
    // let rand_2_vec = rnd_scalar_vec(random_vec.len());
    //let randproof_args_2: Vec<((&Scalar, &Scalar), &Scalar)> = randproof_args_1.into_iter().zip_eq(random_vec_2).collect();

    // Actual randproof, item-wise
    let res_vec: Vec<Result<(SquareProof, SquareProofCommitments), ProofError>> =
        randproof_args_1
            .par_iter()
            .map(|(((x, r_1), r_2), m_com)| {
                SquareProof::prove_existing(
                    &ped_gens,
                    &mut Transcript::new(b"SquareProof"),
                    **x,
                    **m_com,
                    **r_1,
                    **r_2,
                )
            })
            .collect();

    // Add to vector and concatenate
    let mut randproof_vec: Vec<SquareProof> = Vec::with_capacity(value_vec.len());
    let mut commitments_pair_vec: Vec<SquareProofCommitments> =
        Vec::with_capacity(value_vec.len());
    for r in res_vec {
        match r {
            Ok((rp, eg_par)) => {
                randproof_vec.push(rp);
                commitments_pair_vec.push(eg_par);
            }
            Err(e) => return Err(e.into()),
        }
    }
    Ok((randproof_vec, commitments_pair_vec))
}

pub fn create_l2rangeproof_vec(
    value_vec: &Vec<f32>,
    random_vec: &Vec<Scalar>,
    random_vec_2: &Vec<Scalar>,
) -> Result<(Vec<SquareProof>, Vec<SquareProofCommitments>), L2RangeProofError> {
    if value_vec.len() != random_vec.len() {
        return Err(L2RangeProofError::WrongNumBlindingFactors);
    }

    let value_scalar_vec: Vec<Scalar> = f32_to_scalar_vec(&value_vec);

    let ped_gens: PedersenGens = PedersenGens::default();

    // Concatenate togeter
    let randproof_args_1: Vec<(&Scalar, &Scalar)> =
        value_scalar_vec.iter().zip(random_vec).collect();

    // Add random vector
    // let rand_2_vec = rnd_scalar_vec(random_vec.len());
    let randproof_args_2: Vec<((&Scalar, &Scalar), &Scalar)> =
        randproof_args_1.into_iter().zip_eq(random_vec_2).collect();

    // Actual randproof, item-wise
    let res_vec: Vec<Result<(SquareProof, SquareProofCommitments), ProofError>> =
        randproof_args_2
            .par_iter()
            .map(|((x, r_1), r_2)| {
                SquareProof::prove(
                    &ped_gens,
                    &mut Transcript::new(b"SquareProof"),
                    **x,
                    **r_1,
                    **r_2,
                )
            })
            .collect();

    // Add to vector and concatenate
    let mut randproof_vec: Vec<SquareProof> = Vec::with_capacity(value_vec.len());
    let mut commitments_pair_vec: Vec<SquareProofCommitments> =
        Vec::with_capacity(value_vec.len());
    for r in res_vec {
        match r {
            Ok((rp, eg_par)) => {
                randproof_vec.push(rp);
                commitments_pair_vec.push(eg_par);
            }
            Err(e) => return Err(e.into()),
        }
    }
    Ok((randproof_vec, commitments_pair_vec))
}

pub fn verify_l2rangeproof_vec(
    randproof_vec: &Vec<SquareProof>,
    commit_vec: &Vec<SquareProofCommitments>,
) -> Result<bool, L2RangeProofError> {
    if randproof_vec.len() != commit_vec.len() {
        return Err(L2RangeProofError::WrongNumberOfElGamalPairs);
    }

    let ped_gens: PedersenGens = PedersenGens::default();
    let verify_args: Vec<(&SquareProof, &SquareProofCommitments)> =
        randproof_vec.iter().zip(commit_vec).collect();
    let res_vec: Vec<Result<(), ProofError>> = verify_args
        .par_iter()
        .map(|(rp, c)| rp.verify(&ped_gens, &mut Transcript::new(b"SquareProof"), **c))
        .collect();

    // check if error occurred in verification
    let proof_error_opt: Option<&Result<(), ProofError>> = res_vec
        .iter()
        .find(|&r| r.is_err() && r.clone().unwrap_err() != ProofError::VerificationError);

    if proof_error_opt.is_some() {
        return Err(proof_error_opt.unwrap().clone().unwrap_err().into());
    }

    // check if verification succeeded
    return Ok(res_vec
        .iter()
        .find(|&r| r.is_err() && r.clone().unwrap_err() == ProofError::VerificationError)
        .is_none());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pedersen_ops::rnd_scalar_vec;
    use curve25519_dalek_ng::ristretto::RistrettoPoint;
    use curve25519_dalek_ng::scalar::Scalar;
    use crate::range_proof_vec::create_rangeproof;
    use crate::conversion32::get_clip_bounds;

    #[test]
    fn test_l2rangeproof_roundtrip() {
        let values: Vec<f32> = vec![0.5, -3.0, -1.23];
        let blindings: Vec<Scalar> = rnd_scalar_vec(values.len());
        let blindings_2: Vec<Scalar> = rnd_scalar_vec(values.len());
        let (rand_proof_vec, eg_pair_vec) =
            create_l2rangeproof_vec(&values, &blindings, &blindings_2).unwrap();
        assert!(verify_l2rangeproof_vec(&rand_proof_vec, &eg_pair_vec).unwrap());
    }

    #[test]
    fn test_fake_l2rangeproof_roundtrip() {
        let values: Vec<f32> = vec![0.5, -3.0, -1.23];
        let blindings: Vec<Scalar> = rnd_scalar_vec(values.len());
        let blindings_2: Vec<Scalar> = rnd_scalar_vec(values.len());
        let (rand_proof_vec, mut eg_pair_vec) =
            create_l2rangeproof_vec(&values, &blindings, &blindings_2).unwrap();

        let eg_gens: ElGamalGens = ElGamalGens::default();
        let one: RistrettoPoint = eg_gens.B;
        eg_pair_vec[0].c_l += &one;
        assert!(!verify_l2rangeproof_vec(&rand_proof_vec, &eg_pair_vec).unwrap());
    }

    #[test]
    fn test_l2rangeproof_existing_roundtrip() {
        let (min, max) = get_clip_bounds(8);
        println!("{:?} {:?}", min, max);
        // let values: Vec<f32> = vec![1.2265625, -1.0546875];
        let values: Vec<f32> = vec![0.5, -0.025];
        let blindings: Vec<Scalar> = rnd_scalar_vec(values.len());
        let (_, values_com) = create_rangeproof(&values, &blindings, 8, 1)
            .unwrap();

        let blindings_2: Vec<Scalar> = rnd_scalar_vec(values.len());
        let (rand_proof_vec, eg_pair_vec) =
            create_l2rangeproof_vec_existing(&values, values_com, &blindings, &blindings_2)
                .unwrap();
        assert!(verify_l2rangeproof_vec(&rand_proof_vec, &eg_pair_vec).unwrap());
    }

}
