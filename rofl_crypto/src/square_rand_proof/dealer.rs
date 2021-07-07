#![allow(non_snake_case)]
#![allow(dead_code)]

use curve25519_dalek::scalar::Scalar;
use merlin::Transcript;

use super::super::rand_proof::el_gamal::{ElGamalGens, ElGamalPair};
use super::constants::*;
use super::errors::*;
use super::SquareRandProof;
use crate::rand_proof::transcript::TranscriptProtocol;
use crate::square_rand_proof::pedersen::{PedersenCommitment, SquareRandProofCommitments};
use curve25519_dalek::ristretto::RistrettoPoint;

pub struct Dealer {}

impl Dealer {
    pub fn new<'a, 'b>(
        eg_gens: &'a ElGamalGens,
        transcript: &'b mut Transcript,
        c: SquareRandProofCommitments,
    ) -> DealerAwaitingCommitment<'a, 'b> {
        transcript.rand_proof_domain_sep();
        transcript.commit_eg_point(LABEL_COMMIT_REAL_ELGAMAL, &c.c);
        transcript.commit_ped_point(LABEL_COMMIT_REAL_PEDERSEN, &c.c_sq);

        DealerAwaitingCommitment {
            eg_gens: eg_gens,
            transcript: transcript,
            C: c,
        }
    }
}

pub struct DealerAwaitingCommitment<'a, 'b> {
    eg_gens: &'a ElGamalGens,
    transcript: &'b mut Transcript,
    C: SquareRandProofCommitments,
}

impl<'a, 'b> DealerAwaitingCommitment<'a, 'b> {
    pub fn receive_commitment(
        self,
        c_prime: SquareRandProofCommitments,
    ) -> Result<(DealerAwaitingChallengeResponse<'a, 'b>, Scalar), ProofError> {
        self.transcript
            .commit_eg_point(LABEL_COMMIT_PRIME_ELGAMAL, &c_prime.c);
        self.transcript
            .commit_ped_point(LABEL_COMMIT_PRIME_PEDERSEN, &c_prime.c_sq);

        let challenge: Scalar = self.transcript.challenge_scalar(LABEL_CHALLENGE_SCALAR);

        Ok((
            DealerAwaitingChallengeResponse {
                eg_gens: self.eg_gens,
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
    eg_gens: &'a ElGamalGens,
    transcript: &'b mut Transcript,
    C: SquareRandProofCommitments,
    C_prime: SquareRandProofCommitments,
    challenge: Scalar,
}

impl<'a, 'b> DealerAwaitingChallengeResponse<'a, 'b> {
    pub fn receive_challenge_response(
        self,
        z_m: Scalar,
        z_r_1: Scalar,
        z_r_2: Scalar,
    ) -> Result<SquareRandProof, ProofError> {
        self.transcript.commit_scalar(LABEL_RESPONSE_Z_M, &z_m);
        self.transcript.commit_scalar(LABEL_RESPONSE_R_1, &z_r_1);
        self.transcript.commit_scalar(LABEL_RESPONSE_R_2, &z_r_2);

        // Implicit: ElGamal pairs are being verified in two parts
        // Normal rand proof
        let dst_eg_pair: ElGamalPair = self.eg_gens.commit(z_m, z_r_1);
        let src_eg_pair: ElGamalPair = &self.C_prime.c + &(&self.challenge * &self.C.c);
        if dst_eg_pair != src_eg_pair {
            // If you get this error, it could be that the parameters are outside of the parameter-wise
            // range-proof range. (for L2)
            return Err(ProofError::ProvingErrorRandomness);
        }

        // Now for the extension:
        let eg_base: PedersenCommitment = self.C.c.L;
        let ped_blinding: RistrettoPoint = self.eg_gens.B_blinding;
        let lhs_ped_pair: PedersenCommitment = (eg_base * z_m) + (ped_blinding * z_r_2);
        let rhs_ped_pair: PedersenCommitment =
            &self.C_prime.c_sq + &(self.challenge * &self.C.c_sq);

        if lhs_ped_pair != rhs_ped_pair {
            return Err(ProofError::ProvingErrorSquare);
        }

        Ok(SquareRandProof {
            C_prime: self.C_prime,
            Z_m: z_m,
            Z_r_1: z_r_1,
            Z_r_2: z_r_2,
        })
    }
}
