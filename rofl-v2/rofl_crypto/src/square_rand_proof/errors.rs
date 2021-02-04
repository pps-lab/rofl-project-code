/// Represents an error in ElGamal creation, verification, or parsing.
#[derive(Fail, Clone, Debug, Eq, PartialEq)]
pub enum SquareRandProofCommitmentsError {
    /// This error occurs when the El Gamal encoding is malformed.
    #[fail(display = "SquareRandProofCommitments data could not be parsed.")]
    FormatError,
    /// This error occurs when the generators are of the wrong length.
    #[fail(display = "Invalid generators length, must be equal to n.")]
    InvalidGeneratorsLength,
}

#[derive(Fail, Clone, Debug, Eq, PartialEq)]
pub enum ProofError {
    /// This error occurs when the El Gamal encoding is malformed.
    #[fail(display = "Randproof data could not be parsed.")]
    FormatError,
    #[fail(display = "Randomness prove generation failed.")]
    ProvingErrorRandomness,
    #[fail(display = "Randomness prove generation failed for the square.")]
    ProvingErrorSquare,
    /// This error occurs when a proof failed to verify.
    #[fail(display = "Randomness verification failed.")]
    VerificationError,
}
