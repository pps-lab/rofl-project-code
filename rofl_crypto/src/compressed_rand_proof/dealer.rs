#![allow(non_snake_case)]
#![allow(dead_code)]

use curve25519_dalek::scalar::Scalar;
use merlin::Transcript;

use super::super::rand_proof::el_gamal::{ElGamalGens, ElGamalPair};
use super::constants::*;
use super::errors::*;
use super::CompressedRandProof;
use super::types::CompressedRandProofCommitments;
use crate::rand_proof::transcript::TranscriptProtocol;
use curve25519_dalek::ristretto::RistrettoPoint;
use rayon::prelude::*;
use crate::conversion32::{exponentiate, precompute_exponentiate};

pub struct Dealer {}

impl Dealer {
    pub fn new<'a, 'b, 'c>(
        eg_gens: &'a ElGamalGens,
        transcript: &'b mut Transcript,
        c_vec: &'c CompressedRandProofCommitments,
    ) -> DealerAwaitingCommitment<'a, 'b, 'c> {
        transcript.rand_proof_domain_sep();
        for (i, c) in c_vec.c_vec.iter().enumerate() {
            transcript.commit_eg_point(label_commit_real_elgamal(i), c);
        }

        DealerAwaitingCommitment {
            eg_gens: eg_gens,
            transcript: transcript,
            C_vec: c_vec,
        }
    }
}

pub struct DealerAwaitingCommitment<'a, 'b, 'c> {
    eg_gens: &'a ElGamalGens,
    transcript: &'b mut Transcript,
    C_vec: &'c CompressedRandProofCommitments,
}

impl<'a, 'b, 'c> DealerAwaitingCommitment<'a, 'b, 'c> {
    pub fn receive_commitment(
        self,
        c_prime: ElGamalPair,
    ) -> Result<(DealerAwaitingChallengeResponse<'a, 'b, 'c>, Scalar), ProofError> {
        self.transcript
            .commit_eg_point(LABEL_COMMIT_PRIME_ELGAMAL, &c_prime);

        let challenge: Scalar = self.transcript.challenge_scalar(LABEL_CHALLENGE_SCALAR);

        Ok((
            DealerAwaitingChallengeResponse {
                eg_gens: self.eg_gens,
                transcript: self.transcript,
                C_vec: self.C_vec,
                C_prime: c_prime,
                challenge: challenge,
            },
            challenge,
        ))
    }
}

pub struct DealerAwaitingChallengeResponse<'a, 'b, 'c> {
    eg_gens: &'a ElGamalGens,
    transcript: &'b mut Transcript,
    C_vec: &'c CompressedRandProofCommitments,
    C_prime: ElGamalPair,
    challenge: Scalar,
}

impl<'a, 'b, 'c> DealerAwaitingChallengeResponse<'a, 'b, 'c> {
    pub fn receive_challenge_response(
        self,
        z_m: Scalar,
        z_r: Scalar
    ) -> Result<CompressedRandProof, ProofError> {
        self.transcript.commit_scalar(LABEL_RESPONSE_Z_M, &z_m);
        self.transcript.commit_scalar(LABEL_RESPONSE_R, &z_r);

        // // Implicit: ElGamal pairs are being verified in two parts
        // let precomputation_table: Vec<Scalar> = precompute_exponentiate(&self.challenge, self.C_vec.c_vec.len()+1);
        // let dst_eg_pair: ElGamalPair = self.eg_gens.commit(z_m, z_r);
        // let src_eg_pair = self.C_prime + self.C_vec.c_vec.
        //     par_iter().enumerate()
        //     .map(|(i, m)| m * &precomputation_table[i+1]).sum();
        // if dst_eg_pair != src_eg_pair {
        //     return Err(ProofError::ProvingErrorRandomness);
        // }

        Ok(CompressedRandProof {
            C_prime: self.C_prime,
            Z_m: z_m,
            Z_r: z_r,
        })
    }
}
