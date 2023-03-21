#![allow(non_snake_case)]
#![allow(dead_code)]

use bulletproofs::PedersenGens;
use curve25519_dalek::scalar::Scalar;
use merlin::Transcript;

use super::super::rand_proof::el_gamal::{ElGamalGens, ElGamalPair};
use super::constants::*;
use super::errors::*;
use super::SquareProof;
use crate::rand_proof::transcript::TranscriptProtocol;
use crate::square_proof::pedersen::{SquareProofCommitments};
use crate::square_rand_proof::pedersen::{PedersenCommitment};
use curve25519_dalek::ristretto::RistrettoPoint;

pub struct Dealer {}

impl Dealer {
    pub fn new<'a, 'b>(
        ped_gens: &'a PedersenGens,
        transcript: &'b mut Transcript,
        c: SquareProofCommitments,
    ) -> DealerAwaitingCommitment<'a, 'b> {
        transcript.rand_proof_domain_sep();
        transcript.commit_ped_point(LABEL_COMMIT_REAL_ELGAMAL, &c.c_l);
        transcript.commit_ped_point(LABEL_COMMIT_REAL_PEDERSEN, &c.c_sq);

        DealerAwaitingCommitment {
            ped_gens: ped_gens,
            transcript: transcript,
            C: c,
        }
    }
}

pub struct DealerAwaitingCommitment<'a, 'b> {
    ped_gens: &'a PedersenGens,
    transcript: &'b mut Transcript,
    C: SquareProofCommitments,
}

impl<'a, 'b> DealerAwaitingCommitment<'a, 'b> {
    pub fn receive_commitment(
        self,
        c_prime: SquareProofCommitments,
    ) -> Result<(DealerAwaitingChallengeResponse<'a, 'b>, Scalar), ProofError> {
        self.transcript
            .commit_ped_point(LABEL_COMMIT_PRIME_ELGAMAL, &c_prime.c_l);
        self.transcript
            .commit_ped_point(LABEL_COMMIT_PRIME_PEDERSEN, &c_prime.c_sq);

        let challenge: Scalar = self.transcript.challenge_scalar(LABEL_CHALLENGE_SCALAR);

        Ok((
            DealerAwaitingChallengeResponse {
                ped_gens: self.ped_gens,
                transcript: self.transcript,
                C: self.C,
                C_prime: c_prime,
                challenge: challenge,
            },
            challenge,
        ))
    }
}

pub struct DealerAwaitingChallengeResponse<'a, 'b> {
    ped_gens: &'a PedersenGens,
    transcript: &'b mut Transcript,
    C: SquareProofCommitments,
    C_prime: SquareProofCommitments,
    challenge: Scalar,
}

impl<'a, 'b> DealerAwaitingChallengeResponse<'a, 'b> {
    pub fn receive_challenge_response(
        self,
        z_m: Scalar,
        z_r_1: Scalar,
        z_r_2: Scalar,
    ) -> Result<SquareProof, ProofError> {
        self.transcript.commit_scalar(LABEL_RESPONSE_Z_M, &z_m);
        self.transcript.commit_scalar(LABEL_RESPONSE_R_1, &z_r_1);
        self.transcript.commit_scalar(LABEL_RESPONSE_R_2, &z_r_2);

        Ok(SquareProof {
            C_prime: self.C_prime,
            Z_m: z_m,
            Z_r_1: z_r_1,
            Z_r_2: z_r_2,
        })
    }
}
