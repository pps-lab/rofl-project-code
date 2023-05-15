use std::fmt;

/// Represents an error in proof creation, verification, or parsing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClientTrainerError {
    /// This error occurs when a proof failed to verify.
    // #[fail(display = "TrainForRound Failed")]
    TrainForRoundError
}

impl fmt::Display for ClientTrainerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "error with client trainer.")
    }
}
