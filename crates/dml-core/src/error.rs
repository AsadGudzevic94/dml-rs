use thiserror::Error;

/// Errors returned by dml-core.
#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DmlError {
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("matrix is singular or not positive definite")]
    Singular,
    #[error("learner has not been fitted")]
    NotFitted,
    #[error("did not converge: {0}")]
    Convergence(String),
    /// An external learner (e.g. a Python estimator) raised an error.
    #[error("learner failed: {0}")]
    Learner(String),
}

pub type Result<T> = std::result::Result<T, DmlError>;
