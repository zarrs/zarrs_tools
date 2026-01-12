use clap::Parser;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{
        data_type, Array, ArrayBytes, ArrayIndicesTinyVec, ArraySubset, DataType, DataTypeExt,
    },
    filesystem::FilesystemStore,
};

use crate::{
    filter::{
        calculate_chunk_limit,
        filter_error::FilterError,
        filter_traits::{ChunkInfo, FilterTraits},
        FilterArguments, FilterCommonArguments,
    },
    progress::{Progress, ProgressCallback},
    type_dispatch::retrieve_and_store_converting,
    UnsupportedDataTypeError,
};

#[derive(Debug, Clone, Parser, Serialize, Deserialize)]
pub struct ReencodeArguments {}

impl FilterArguments for ReencodeArguments {
    fn name(&self) -> String {
        "reencode".to_string()
    }

    fn init(
        &self,
        common_args: &FilterCommonArguments,
    ) -> Result<Box<dyn FilterTraits>, FilterError> {
        Ok(Box::new(Reencode::new(*common_args.chunk_limit())))
    }
}

pub struct Reencode {
    chunk_limit: Option<usize>,
}

impl Reencode {
    pub fn new(chunk_limit: Option<usize>) -> Self {
        Self { chunk_limit }
    }

    pub fn apply_chunk(
        &self,
        input: &Array<FilesystemStore>,
        output: &Array<FilesystemStore>,
        chunk_indices: &[u64],
        progress: &Progress,
    ) -> Result<(), FilterError> {
        let input_output_subset = output.chunk_subset_bounded(chunk_indices).unwrap();

        if input.data_type() == output.data_type() {
            // No conversion needed, use raw bytes
            let subset_bytes: ArrayBytes =
                progress.read(|| input.retrieve_array_subset(&input_output_subset))?;
            progress.write(|| output.store_array_subset(&input_output_subset, subset_bytes))?;
        } else {
            // Convert via intermediate type
            let timing = retrieve_and_store_converting(input, output, &input_output_subset)?;
            progress.add_conversion_timing(timing);
        }

        progress.next();
        Ok(())
    }
}

fn is_supported_type(dt: &DataType) -> bool {
    dt.is::<data_type::BoolDataType>()
        || dt.is::<data_type::Int8DataType>()
        || dt.is::<data_type::Int16DataType>()
        || dt.is::<data_type::Int32DataType>()
        || dt.is::<data_type::Int64DataType>()
        || dt.is::<data_type::UInt8DataType>()
        || dt.is::<data_type::UInt16DataType>()
        || dt.is::<data_type::UInt32DataType>()
        || dt.is::<data_type::UInt64DataType>()
        || dt.is::<data_type::Float16DataType>()
        || dt.is::<data_type::Float32DataType>()
        || dt.is::<data_type::Float64DataType>()
        || dt.is::<data_type::BFloat16DataType>()
}

impl FilterTraits for Reencode {
    fn is_compatible(
        &self,
        chunk_input: ChunkInfo,
        chunk_output: ChunkInfo,
    ) -> Result<(), FilterError> {
        for dt in [chunk_input.1, chunk_output.1] {
            if !is_supported_type(dt) {
                Err(FilterError::UnsupportedDataType(
                    UnsupportedDataTypeError::from(format!("{:?}", dt)),
                ))?;
            }
        }
        Ok(())
    }

    fn memory_per_chunk(&self, _chunk_input: ChunkInfo, chunk_output: ChunkInfo) -> usize {
        chunk_output.1.fixed_size().unwrap()
    }

    fn apply(
        &self,
        input: &Array<FilesystemStore>,
        output: &mut Array<FilesystemStore>,
        progress_callback: &ProgressCallback,
    ) -> Result<(), FilterError> {
        assert_eq!(input.shape(), output.shape());

        let chunks = ArraySubset::new_with_shape(output.chunk_grid_shape().to_vec());
        let progress = Progress::new(chunks.num_elements_usize(), progress_callback);

        let chunk_limit = if let Some(chunk_limit) = self.chunk_limit {
            chunk_limit
        } else {
            let input_chunk_shape = input.chunk_shape(&vec![0; input.dimensionality()])?;
            let output_chunk_shape = output.chunk_shape(&vec![0; input.dimensionality()])?;
            calculate_chunk_limit(self.memory_per_chunk(
                (&input_chunk_shape, input.data_type(), input.fill_value()),
                (&output_chunk_shape, output.data_type(), output.fill_value()),
            ))?
        };

        let indices = chunks.indices();
        rayon_iter_concurrent_limit::iter_concurrent_limit!(
            chunk_limit,
            indices,
            try_for_each,
            |chunk_indices: ArrayIndicesTinyVec| {
                self.apply_chunk(input, output, &chunk_indices, &progress)
            }
        )?;

        Ok(())
    }
}
