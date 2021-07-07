#![allow(non_snake_case)]

use curve25519_dalek::ristretto::CompressedRistretto;
use curve25519_dalek::scalar::Scalar;
use merlin::Transcript;

use super::super::square_rand_proof::pedersen::PedersenCommitment;
use super::el_gamal::ElGamalPair;

pub trait TranscriptProtocol {
    fn rand_proof_domain_sep(&mut self);
    fn commit_scalar(&mut self, label: &'static [u8], scalar: &Scalar);
    fn commit_point(&mut self, label: &'static [u8], point: &CompressedRistretto);
    fn challenge_scalar(&mut self, label: &'static [u8]) -> Scalar;
    fn commit_eg_point(&mut self, label: &'static [u8], eg_pair: &ElGamalPair);
    fn commit_ped_point(&mut self, label: &'static [u8], ped_pair: &PedersenCommitment); // Sugar
}

impl TranscriptProtocol for Transcript {
    fn rand_proof_domain_sep(&mut self) {
        self.append_message(b"dom-sep", b"randomness proof v1")
    }

    fn commit_scalar(&mut self, label: &'static [u8], scalar: &Scalar) {
        self.append_message(label, scalar.as_bytes());
    }

    fn commit_point(&mut self, label: &'static [u8], point: &CompressedRistretto) {
        self.append_message(label, point.as_bytes());
    }

    fn commit_eg_point(&mut self, label: &'static [u8], eg_pair: &ElGamalPair) {
        self.append_message(label, &eg_pair.to_bytes());
    }

    fn commit_ped_point(&mut self, label: &'static [u8], ped_pair: &PedersenCommitment) {
        self.append_message(label, ped_pair.compress().as_bytes());
    }

    fn challenge_scalar(&mut self, label: &'static [u8]) -> Scalar {
        let mut buf = [0u8; 64];
        self.challenge_bytes(label, &mut buf);
        Scalar::from_bytes_mod_order_wide(&buf)
    }
}
