use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::ristretto::RistrettoPoint;
use merlin::Transcript;
use rayon::prelude::*;

use crate::rand_proof::RandProof;
use crate::rand_proof::{ElGamalPair, ElGamalGens};
use crate::rand_proof::ProofError;
use crate::conversion32::{f32_to_scalar_vec, f32_to_fp_vec};
use crate::fp::URawFix;
mod errors;
pub use self::errors::RandProofError;
use rayon::prelude::*;


pub fn create_randproof_vec(
    value_vec: &Vec<f32>,
    random_vec: &Vec<Scalar>,
    ) -> Result<(Vec<RandProof>, Vec<ElGamalPair>), RandProofError> {
    
    if value_vec.len() != random_vec.len() {
        return Err(RandProofError::WrongNumBlindingFactors);
    }

    let value_scalar_vec: Vec<Scalar> = f32_to_scalar_vec(&value_vec);
    // let value_fp_vec: Vec<URawFix> = f32_to_fp_vec(&value_vec);
    // let value_scalar_vec: Vec<Scalar> = value_fp_vec.iter().map(|x| Scalar::from(*x as u64)).collect();

    let eg_gens: ElGamalGens = ElGamalGens::default();

    let randproof_args: Vec<(&Scalar, &Scalar)> = value_scalar_vec.iter().zip(random_vec).collect();
    
    let res_vec: Vec<Result<(RandProof, ElGamalPair), ProofError>> = 
    randproof_args.par_iter().map(|(x, r)| RandProof::prove(&eg_gens, &mut Transcript::new(b"RandProof"), **x, **r)).collect();
    
    let mut randproof_vec: Vec<RandProof> = Vec::with_capacity(value_vec.len());
    let mut eg_pair_vec: Vec<ElGamalPair> = Vec::with_capacity(value_vec.len());
    for r in res_vec {
        match r {
            Ok((rp, eg_par)) => {
                randproof_vec.push(rp);
                eg_pair_vec.push(eg_par);
            }
            Err(e) => return Err(e.into())
        }
    }
    Ok((randproof_vec, eg_pair_vec))
}

pub fn create_randproof_vec_existing(
    value_vec: &Vec<f32>,
    existing_value_com_vec: Vec<RistrettoPoint>,
    random_vec: &Vec<Scalar>,
    ) -> Result<(Vec<RandProof>, Vec<ElGamalPair>), RandProofError> {
    
    if value_vec.len() != random_vec.len() {
        return Err(RandProofError::WrongNumBlindingFactors);
    }

    let value_scalar_vec: Vec<Scalar> = f32_to_scalar_vec(&value_vec);
    // let value_fp_vec: Vec<URawFix> = f32_to_fp_vec(&value_vec);
    // let value_scalar_vec: Vec<Scalar> = value_fp_vec.iter().map(|x| Scalar::from(*x as u64)).collect();

    let eg_gens: ElGamalGens = ElGamalGens::default();

    let randproof_args: Vec<((&Scalar, &Scalar), &RistrettoPoint)> = value_scalar_vec.iter().zip(random_vec).zip(&existing_value_com_vec).collect();
    
    let res_vec: Vec<Result<(RandProof, ElGamalPair), ProofError>> = 
    randproof_args.par_iter().map(|((x, r), p)| RandProof::prove_existing(&eg_gens, &mut Transcript::new(b"RandProof"), **x, **p, **r)).collect();
    
    let mut randproof_vec: Vec<RandProof> = Vec::with_capacity(value_vec.len());
    let mut eg_pair_vec: Vec<ElGamalPair> = Vec::with_capacity(value_vec.len());
    for r in res_vec {
        match r {
            Ok((rp, eg_par)) => {
                randproof_vec.push(rp);
                eg_pair_vec.push(eg_par);
            }
            Err(e) => return Err(e.into())
        }
    }
    Ok((randproof_vec, eg_pair_vec))
}

pub fn verify_randproof_vec(
    randproof_vec: &Vec<RandProof>,
    commit_vec: &Vec<ElGamalPair>)
    -> Result<bool, RandProofError> {

    
    if randproof_vec.len() != commit_vec.len() {
        return Err(RandProofError::WrongNumberOfElGamalPairs);
    }

    let eg_gens: ElGamalGens = ElGamalGens::default();
    let verify_args: Vec<(&RandProof, &ElGamalPair)> = randproof_vec.par_iter().zip(commit_vec).collect();
    let res_vec: Vec<Result<(), ProofError>> = verify_args.par_iter().map(|(rp, c)| rp.verify(&eg_gens, &mut Transcript::new(b"RandProof"), **c)).collect();
    // check if error occurred in verification
    let proof_error_opt: Option<&Result<(), ProofError>> = 
    res_vec.iter().find(|&r| r.is_err() && r.clone().unwrap_err() != ProofError::VerificationError);
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
    use curve25519_dalek::scalar::Scalar;
    use curve25519_dalek::ristretto::RistrettoPoint;

    #[test]
    fn test_randproof_roundtrip() {
        let values: Vec<f32> = vec![0.5, -3.0, -1.23];
        let blindings: Vec<Scalar> = rnd_scalar_vec(values.len());
        let (rand_proof_vec, eg_pair_vec) = create_randproof_vec(&values, &blindings).unwrap();
        assert!(verify_randproof_vec(&rand_proof_vec, &eg_pair_vec).unwrap());
    }

    #[test]
    fn test_fake_randproof_roundtrip() {
        let values: Vec<f32> = vec![0.5, -3.0, -1.23];
        let blindings: Vec<Scalar> = rnd_scalar_vec(values.len());
        let (rand_proof_vec, mut eg_pair_vec) = create_randproof_vec(&values, &blindings).unwrap();
        
        let eg_gens: ElGamalGens = ElGamalGens::default();
        let one: RistrettoPoint = eg_gens.B;
        eg_pair_vec[0].R += one; 
        assert!(!verify_randproof_vec(&rand_proof_vec, &eg_pair_vec).unwrap());
    }
}
