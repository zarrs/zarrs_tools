mod array_subset_overlap;
mod filter_arguments;
mod filter_command;
mod filter_common_arguments;
mod filter_error;
mod filter_input_output_arguments;
mod filter_traits;
mod kernel;
mod path_or_identifier;
mod path_or_temp_path;
// mod chunk_cache;

pub mod filters {
    pub mod clamp;
    pub mod crop;
    pub mod downsample;
    pub mod equal;
    pub mod gaussian;
    pub mod gradient_magnitude;
    pub mod guided_filter;
    pub mod reencode;
    pub mod replace_value;
    pub mod rescale;
    pub mod summed_area_table;
}

pub use array_subset_overlap::ArraySubsetOverlap;
pub use filter_arguments::FilterArguments;
pub use filter_command::{FilterCommand, FilterCommandTraits};
pub use filter_common_arguments::FilterCommonArguments;
pub use filter_error::FilterError;
pub use filter_input_output_arguments::FilterInputOutputArguments;
pub use filter_traits::FilterTraits;
pub use path_or_identifier::PathOrIdentifier;
pub use path_or_temp_path::PathOrTempPath;
// pub use chunk_cache::{ChunkCache, retrieve_array_subset_ndarray_cached};

use sysinfo::{MemoryRefreshKind, RefreshKind, System};

/// Calculates the chunk limit based on the amount of available memory.
pub fn calculate_chunk_limit(memory_per_chunk: usize) -> Result<usize, FilterError> {
    let system = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram()),
    );
    let available_memory = usize::try_from(system.available_memory()).unwrap();
    let available_memory_target = available_memory * 8 / 10; // 80%
    let chunk_limit = available_memory_target / memory_per_chunk;
    if chunk_limit == 0 {
        Err(FilterError::Other(
            "There is not enough available memory to process a single output chunk. Consider reducing the chunk shape (or shard shape if sharding)".to_string(),
        ))
    } else {
        Ok(chunk_limit)
    }
}
