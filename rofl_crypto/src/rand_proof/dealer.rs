#![allow(non_snake_case)]
#![allow(dead_code)]

use curve25519_dalek::scalar::Scalar;
use merlin::Transcript;

use super::el_gamal::{ElGamalGens, ElGamalPair};
use super::errors::*;
use super::transcript::TranscriptProtocol;
use super::RandProof;

pub struct Dealer {}

impl Dealer {
    pub fn new<'a, 'b>(
        eg_gens: &'a ElGamalGens,
        transcript: &'b mut Transcript,
        c: ElGamalPair,
    ) -> DealerAwaitingCommitment<'a, 'b> {
        transcript.rand_proof_domain_sep();
        transcript.commit_eg_point(b"C", &c);

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
    C: ElGamalPair,
}

impl<'a, 'b> DealerAwaitingCommitment<'a, 'b> {
    pub fn receive_commitment(
        self,
        c_prime: ElGamalPair,
    ) -> Result<(DealerAwaitingChallengeResponse<'a, 'b>, Scalar), ProofError> {
        self.transcript.commit_eg_point(b"C_prime", &c_prime);
        let challenge: Scalar = self.transcript.challenge_scalar(b"c");

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
    C: ElGamalPair,
    C_prime: ElGamalPair,
    challenge: Scalar,
}

impl<'a, 'b> DealerAwaitingChallengeResponse<'a, 'b> {
    pub fn receive_challenge_response(
        self,
        z_m: Scalar,
        z_r: Scalar,
    ) -> Result<RandProof, ProofError> {
        self.transcript.commit_scalar(b"Z_m", &z_m);
        self.transcript.commit_scalar(b"Z_r", &z_r);

        // This is redundant, but it's a good sanity check.
        // let dst_eg_pair: ElGamalPair = self.eg_gens.commit(z_m, z_r);
        // let src_eg_pair: ElGamalPair = &self.C_prime + &(&self.challenge * &self.C);
        // if dst_eg_pair != src_eg_pair {
        //     return Err(ProofError::ProvingError);
        // }
        Ok(RandProof {
            C_prime: self.C_prime,
            Z_m: z_m,
            Z_r: z_r,
        })
    }
}
