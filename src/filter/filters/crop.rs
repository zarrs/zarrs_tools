use clap::Parser;
use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{data_type, Array, ArrayBytes, ArrayIndicesTinyVec, DataTypeExt},
    array_subset::ArraySubset,
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
pub struct CropArguments {
    /// Crop offset, comma delimited.
    #[arg(required = true, value_delimiter = ',')]
    pub offset: Vec<u64>,
    /// Crop shape, comma delimited.
    #[arg(required = true, value_delimiter = ',')]
    pub shape: Vec<u64>,
}

impl FilterArguments for CropArguments {
    fn name(&self) -> String {
        "crop".to_string()
    }

    fn init(
        &self,
        common_args: &FilterCommonArguments,
    ) -> Result<Box<dyn FilterTraits>, FilterError> {
        Ok(Box::new(Crop::new(
            self.offset.clone(),
            self.shape.clone(),
            *common_args.chunk_limit(),
        )))
    }
}

pub struct Crop {
    offset: Vec<u64>,
    shape: Vec<u64>,
    chunk_limit: Option<usize>,
}

impl Crop {
    pub fn new(offset: Vec<u64>, shape: Vec<u64>, chunk_limit: Option<usize>) -> Self {
        Self {
            offset,
            shape,
            chunk_limit,
        }
    }

    // Determine the input and output subset
    fn get_input_output_subset(
        &self,
        output: &Array<FilesystemStore>,
        chunk_indices: &[u64],
    ) -> (ArraySubset, ArraySubset) {
        let output_subset = output.chunk_subset_bounded(chunk_indices).unwrap();
        let input_subset = ArraySubset::new_with_start_shape(
            std::iter::zip(output_subset.start(), self.offset.clone())
                .map(|(s, o)| s + o)
                .collect::<Vec<_>>(),
            output_subset.shape().to_vec(),
        )
        .unwrap();
        (input_subset, output_subset)
    }

    /// Retrieve from input subset, convert via type T, and store to output subset.
    fn convert_via<T>(
        input: &Array<FilesystemStore>,
        output: &Array<FilesystemStore>,
        input_subset: &ArraySubset,
        output_subset: &ArraySubset,
        progress: &Progress,
    ) -> Result<(), FilterError>
    where
        T: Copy + Send + Sync + 'static,
        u8: AsPrimitive<T>,
        i8: AsPrimitive<T>,
        i16: AsPrimitive<T>,
        i32: AsPrimitive<T>,
        i64: AsPrimitive<T>,
        u16: AsPrimitive<T>,
        u32: AsPrimitive<T>,
        u64: AsPrimitive<T>,
        half::f16: AsPrimitive<T>,
        half::bf16: AsPrimitive<T>,
        f32: AsPrimitive<T>,
        f64: AsPrimitive<T>,
        T: AsPrimitive<u8>,
        T: AsPrimitive<i8>,
        T: AsPrimitive<i16>,
        T: AsPrimitive<i32>,
        T: AsPrimitive<i64>,
        T: AsPrimitive<u16>,
        T: AsPrimitive<u32>,
        T: AsPrimitive<u64>,
        T: AsPrimitive<half::f16>,
        T: AsPrimitive<half::bf16>,
        T: AsPrimitive<f32>,
        T: AsPrimitive<f64>,
    {
        let (elements, retrieve_timing) = retrieve_as::<T, _>(input, input_subset)?;
        progress.add_retrieve_timing(retrieve_timing);
        let store_timing = store_from(output, output_subset, elements)?;
        progress.add_store_timing(store_timing);
        Ok(())
    }

    pub fn apply_chunk(
        &self,
        input: &Array<FilesystemStore>,
        output: &Array<FilesystemStore>,
        chunk_indices: &[u64],
        progress: &Progress,
    ) -> Result<(), FilterError> {
        let (input_subset, output_subset) = self.get_input_output_subset(output, chunk_indices);

        if input.data_type().identifier() == output.data_type().identifier() {
            // No conversion needed, use raw bytes
            let output_bytes: ArrayBytes =
                progress.read(|| input.retrieve_array_subset(&input_subset))?;
            progress.write(|| output.store_array_subset(&output_subset, output_bytes))?;
        } else {
            // Convert via intermediate type
            // Note: We use separate subsets since crop has different input/output positions
            match IntermediateType::for_data_type(input.data_type()) {
                IntermediateType::F32 => Self::convert_via::<f32>(
                    input,
                    output,
                    &input_subset,
                    &output_subset,
                    progress,
                )?,
                IntermediateType::F64 => Self::convert_via::<f64>(
                    input,
                    output,
                    &input_subset,
                    &output_subset,
                    progress,
                )?,
                IntermediateType::I32 => Self::convert_via::<i32>(
                    input,
                    output,
                    &input_subset,
                    &output_subset,
                    progress,
                )?,
                IntermediateType::I64 => Self::convert_via::<i64>(
                    input,
                    output,
                    &input_subset,
                    &output_subset,
                    progress,
                )?,
                IntermediateType::U32 => Self::convert_via::<u32>(
                    input,
                    output,
                    &input_subset,
                    &output_subset,
                    progress,
                )?,
                IntermediateType::U64 => Self::convert_via::<u64>(
                    input,
                    output,
                    &input_subset,
                    &output_subset,
                    progress,
                )?,
            }
        }

        progress.next();
        Ok(())
    }
}

impl FilterTraits for Crop {
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

    fn memory_per_chunk(&self, _chunk_input: ChunkInfo, chunk_output: ChunkInfo) -> usize {
        chunk_output.1.fixed_size().unwrap()
    }

    fn output_shape(&self, _input: &Array<FilesystemStore>) -> Option<Vec<u64>> {
        Some(self.shape.clone())
    }

    fn apply(
        &self,
        input: &Array<FilesystemStore>,
        output: &mut Array<FilesystemStore>,
        progress_callback: &ProgressCallback,
    ) -> Result<(), FilterError> {
        assert_eq!(output.shape(), self.shape);

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
