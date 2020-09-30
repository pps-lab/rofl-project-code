use bulletproofs::ProofError;

/// Represents an error in proof creation, verification, or parsing.
#[derive(Fail, Clone, Debug, Eq, PartialEq)]
pub enum L2RangeProofError {
    /// This error occurs when a proof failed to verify.
    #[fail(display = "Fixed precision representation does exceed prove range bounds")]
    ValueOutOfRangeError,

    #[fail(display = "The scalar L2 norm exceeds prove range bounds: {}", _0)]
    NormOutOfRangeError(String),

    #[fail(display = "The scalar calculation does not match the floating point calculation: {} != {}", _0, _1)]
    OverflowError(String, String),

    // Error originating from the Bulletproof lib
    #[fail(display = "Internal error during proof creation: {}", _0)]
    ProvingError(ProofError),

    #[fail(display = "Commitments do not sum to square commitment")]
    SumError,
}

impl From<ProofError> for L2RangeProofError {
    fn from(e: ProofError) -> L2RangeProofError {
        L2RangeProofError::ProvingError(e)
    }
}