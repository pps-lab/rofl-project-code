use bulletproofs::ProofError;

/// Represents an error in proof creation, verification, or parsing.
#[derive(Fail, Clone, Debug, Eq, PartialEq)]
pub enum RangeProofError {
    /// This error occurs when a proof failed to verify.
    #[fail(display = "Fixed precision representation does exceed prove range bounds")]
    ValueOutOfRangeError,

    // Error originating from the Bulletproof lib
    #[fail(display = "Internal error during proof creation: {}", _0)]
    ProvingError(ProofError),
}

impl From<ProofError> for RangeProofError {
    fn from(e: ProofError) -> RangeProofError {
        RangeProofError::ProvingError(e)
    }
}