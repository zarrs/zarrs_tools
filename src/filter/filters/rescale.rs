use clap::Parser;
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
pub struct RescaleArguments {
    /// Multiplier term.
    #[arg(allow_hyphen_values(true))]
    pub multiply: f64,
    /// Addition term.
    #[arg(allow_hyphen_values(true))]
    pub add: f64,
    /// Perform the addition before multiplication.
    #[arg(long)]
    #[serde(default)]
    pub add_first: bool,
}

impl FilterArguments for RescaleArguments {
    fn name(&self) -> String {
        "rescale".to_string()
    }

    fn init(
        &self,
        common_args: &FilterCommonArguments,
    ) -> Result<Box<dyn FilterTraits>, FilterError> {
        Ok(Box::new(Rescale::new(
            self.multiply,
            self.add,
            self.add_first,
            *common_args.chunk_limit(),
        )))
    }
}

pub struct Rescale {
    multiply: f64,
    add: f64,
    add_first: bool,
    chunk_limit: Option<usize>,
}

impl Rescale {
    pub fn new(multiply: f64, add: f64, add_first: bool, chunk_limit: Option<usize>) -> Self {
        Self {
            multiply,
            add,
            add_first,
            chunk_limit,
        }
    }

    /// Apply the rescale operation to f32 elements.
    fn apply_elements_f32(&self, elements: Vec<f32>) -> Vec<f32> {
        let multiply = self.multiply as f32;
        let add = self.add as f32;
        if self.add_first {
            elements
                .into_par_iter()
                .map(|v| (v + add) * multiply)
                .collect()
        } else {
            elements
                .into_par_iter()
                .map(|v| v.mul_add(multiply, add))
                .collect()
        }
    }

    /// Apply the rescale operation to f64 elements.
    fn apply_elements_f64(&self, elements: Vec<f64>) -> Vec<f64> {
        if self.add_first {
            elements
                .into_par_iter()
                .map(|v| (v + self.add) * self.multiply)
                .collect()
        } else {
            elements
                .into_par_iter()
                .map(|v| v.mul_add(self.multiply, self.add))
                .collect()
        }
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
            let elements_out = progress.process(|| self.apply_elements_f64(elements_in));
            let store_timing = store_from(output, &input_output_subset, elements_out)?;
            progress.add_store_timing(store_timing);
        } else {
            let (elements_in, retrieve_timing) =
                retrieve_as::<f32, _>(input, &input_output_subset)?;
            progress.add_retrieve_timing(retrieve_timing);
            let elements_out = progress.process(|| self.apply_elements_f32(elements_in));
            let store_timing = store_from(output, &input_output_subset, elements_out)?;
            progress.add_store_timing(store_timing);
        }

        progress.next();
        Ok(())
    }
}

impl FilterTraits for Rescale {
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
