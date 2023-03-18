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
use crate::square_rand_proof::pedersen::{PedersenCommitment, SquareRandProofCommitments};
use curve25519_dalek::ristretto::RistrettoPoint;

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

        // Implicit: ElGamal pairs are being verified in two parts
        // Normal rand proof
        // self.m_prime + self.m.iter().enumerate().map(|i, m| m * c.pow(i+1)).sum();

        // TODO: Is this part strictly necessary? Maybe this is an additional check on the prover?
        let dst_eg_pair: ElGamalPair = self.eg_gens.commit(z_m, z_r);
        // TODO: POW
        let src_eg_pair = self.C_vec.c_vec.iter().enumerate().map(|(i, m)| m * &self.challenge).sum();
        // let src_eg_pair: ElGamalPair = &self.C_prime + self.C_vec.c_vec.iter().enumerate().map(|(i, m)| m * &self.challenge).sum();
        if dst_eg_pair != src_eg_pair {
            // If you get this error, it could be that the parameters are outside of the parameter-wise
            // range-proof range. (for L2)
            return Err(ProofError::ProvingErrorRandomness);
        }

        Ok(CompressedRandProof {
            C_prime: self.C_prime,
            Z_m: z_m,
            Z_r: z_r,
        })
    }
}
