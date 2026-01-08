use thiserror::Error;
use zarrs::{
    array::{data_type::DataTypeFillValueMetadataError, ArrayCreateError, ArrayError},
    storage::StorageError,
};

#[derive(Debug, Error)]
pub enum FilterError {
    #[error(transparent)]
    StorageError(#[from] StorageError),
    #[error(transparent)]
    ArrayError(#[from] ArrayError),
    #[error(transparent)]
    ArrayCreateError(#[from] ArrayCreateError),
    #[error("Invalid parameters: {_0}")]
    InvalidParameters(String),
    #[error(transparent)]
    JSONError(#[from] serde_json::Error),
    #[error("Unsupported data type {_0}")]
    UnsupportedDataType(#[from] crate::UnsupportedDataTypeError),
    #[error(transparent)]
    IncompatibleFillValue(#[from] DataTypeFillValueMetadataError),
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    #[error("{_0}")]
    Other(String),
}
