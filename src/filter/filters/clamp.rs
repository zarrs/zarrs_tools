use clap::Parser;
use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{data_type, Array, ArrayIndicesTinyVec, ArraySubset, DataTypeExt},
    filesystem::FilesystemStore,
    plugin::ExtensionIdentifier,
};

use crate::{
    filter::{
        calculate_chunk_limit,
        filter_error::FilterError,
        filter_traits::{ChunkInfo, FilterTraits},
        FilterArguments, FilterCommonArguments,
    },
    progress::{Progress, ProgressCallback},
    type_dispatch::{retrieve_as, store_from, IntermediateType},
    UnsupportedDataTypeError,
};

#[derive(Debug, Clone, Parser, Serialize, Deserialize)]
pub struct ClampArguments {
    /// Minimum.
    pub min: f64,
    /// Maximum.
    pub max: f64,
}

impl FilterArguments for ClampArguments {
    fn name(&self) -> String {
        "clamp".to_string()
    }

    fn init(
        &self,
        common_args: &FilterCommonArguments,
    ) -> Result<Box<dyn FilterTraits>, FilterError> {
        Ok(Box::new(Clamp::new(
            self.min,
            self.max,
            *common_args.chunk_limit(),
        )))
    }
}

pub struct Clamp {
    min: f64,
    max: f64,
    chunk_limit: Option<usize>,
}

impl Clamp {
    pub fn new(min: f64, max: f64, chunk_limit: Option<usize>) -> Self {
        Self {
            min,
            max,
            chunk_limit,
        }
    }

    /// Apply the clamp operation to elements of type T.
    fn apply_elements<T>(&self, elements: Vec<T>) -> Vec<T>
    where
        T: Copy + Send + Sync + PartialOrd + 'static,
        f64: AsPrimitive<T>,
    {
        let min: T = self.min.as_();
        let max: T = self.max.as_();
        elements
            .into_par_iter()
            .map(|v| num_traits::clamp(v, min, max))
            .collect()
    }

    pub fn apply_chunk(
        &self,
        input: &Array<FilesystemStore>,
        output: &Array<FilesystemStore>,
        chunk_indices: &[u64],
        progress: &Progress,
    ) -> Result<(), FilterError> {
        let input_output_subset = output.chunk_subset_bounded(chunk_indices).unwrap();

        // Choose intermediate type based on input data type
        // Use f64 for f64 and large integer types to preserve precision
        let use_f64 = matches!(
            IntermediateType::for_data_type(input.data_type()),
            IntermediateType::F64 | IntermediateType::I64 | IntermediateType::U64
        );

        if use_f64 {
            let (elements_in, retrieve_timing) =
                retrieve_as::<f64, _>(input, &input_output_subset)?;
            progress.add_retrieve_timing(retrieve_timing);
            let elements_out = progress.process(|| self.apply_elements(elements_in));
            let store_timing = store_from(output, &input_output_subset, elements_out)?;
            progress.add_store_timing(store_timing);
        } else {
            let (elements_in, retrieve_timing) =
                retrieve_as::<f32, _>(input, &input_output_subset)?;
            progress.add_retrieve_timing(retrieve_timing);
            let elements_out = progress.process(|| self.apply_elements(elements_in));
            let store_timing = store_from(output, &input_output_subset, elements_out)?;
            progress.add_store_timing(store_timing);
        }

        progress.next();
        Ok(())
    }
}

impl FilterTraits for Clamp {
    fn is_compatible(
        &self,
        chunk_input: ChunkInfo,
        chunk_output: ChunkInfo,
    ) -> Result<(), FilterError> {
        const SUPPORTED_TYPES: &[&str] = &[
            data_type::BoolDataType::IDENTIFIER,
            data_type::Int8DataType::IDENTIFIER,
            data_type::Int16DataType::IDENTIFIER,
            data_type::Int32DataType::IDENTIFIER,
            data_type::Int64DataType::IDENTIFIER,
            data_type::UInt8DataType::IDENTIFIER,
            data_type::UInt16DataType::IDENTIFIER,
            data_type::UInt32DataType::IDENTIFIER,
            data_type::UInt64DataType::IDENTIFIER,
            data_type::Float16DataType::IDENTIFIER,
            data_type::Float32DataType::IDENTIFIER,
            data_type::Float64DataType::IDENTIFIER,
            data_type::BFloat16DataType::IDENTIFIER,
        ];
        for data_type in [chunk_input.1, chunk_output.1] {
            if !SUPPORTED_TYPES.contains(&data_type.identifier()) {
                Err(UnsupportedDataTypeError::from(
                    data_type.identifier().to_string(),
                ))?;
            }
        }
        Ok(())
    }

    fn memory_per_chunk(&self, chunk_input: ChunkInfo, chunk_output: ChunkInfo) -> usize {
        chunk_input.1.fixed_size().unwrap() + chunk_output.1.fixed_size().unwrap()
    }

    fn apply(
        &self,
        input: &Array<FilesystemStore>,
        output: &mut Array<FilesystemStore>,
        progress_callback: &ProgressCallback,
    ) -> Result<(), FilterError> {
        assert_eq!(output.shape(), input.shape());

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
