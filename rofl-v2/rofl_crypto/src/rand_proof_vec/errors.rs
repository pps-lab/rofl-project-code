use crate::rand_proof::ProofError;

/// Represents an error in proof creation, verification, or parsing.
#[derive(Fail, Clone, Debug, Eq, PartialEq)]
pub enum RandProofError {
    /// This error occurs during proving if the number of blinding
    /// factors does not match the number of values.
    #[fail(display = "Wrong number of blinding factors supplied.")]
    WrongNumBlindingFactors,
    ///
    #[fail(display = "Number of ElGamal pairs does not match number of supplied RandProofs")]
    WrongNumberOfElGamalPairs,
    /// Error originating from the Bulletproof lib
    #[fail(display = "Internal error during proof creation: {}", _0)]
    ProvingError(ProofError),
}

impl From<ProofError> for RandProofError {
    fn from(e: ProofError) -> RandProofError {
        RandProofError::ProvingError(e)
    }
}
