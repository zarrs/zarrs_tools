use clap::Parser;
use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{
        data_type, Array, ArrayBytes, ArrayIndicesTinyVec, DataTypeExt, Element, ElementOwned,
    },
    array_subset::ArraySubset,
    filesystem::FilesystemStore,
    plugin::ExtensionIdentifier,
};

use crate::{
    filter::{
        calculate_chunk_limit,
        filter_error::FilterError,
        filter_traits::{ChunkInfo, FilterTraits},
        FilterArguments, FilterCommonArguments, UnsupportedDataTypeError,
    },
    progress::{Progress, ProgressCallback},
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
        let subset_bytes: ArrayBytes =
            progress.read(|| input.retrieve_array_subset(&input_output_subset))?;
        progress.write(|| output.store_array_subset(&input_output_subset, subset_bytes))?;
        progress.next();
        Ok(())
    }

    pub fn apply_chunk_convert<TIn, TOut>(
        &self,
        input: &Array<FilesystemStore>,
        output: &Array<FilesystemStore>,
        chunk_indices: &[u64],
        progress: &Progress,
    ) -> Result<(), FilterError>
    where
        TIn: ElementOwned + Send + Sync + AsPrimitive<TOut>,
        TOut: Element + Send + Sync + Copy + 'static,
    {
        let input_output_subset = output.chunk_subset_bounded(chunk_indices).unwrap();

        let input_elements: Vec<TIn> =
            progress.read(|| input.retrieve_array_subset(&input_output_subset))?;

        let output_elements = progress.process(|| {
            input_elements
                .par_iter()
                .map(|input| input.as_())
                .collect::<Vec<TOut>>()
        });
        drop(input_elements);

        progress.write(|| output.store_array_subset(&input_output_subset, output_elements))?;

        progress.next();
        Ok(())
    }
}

impl FilterTraits for Reencode {
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
                Err(FilterError::UnsupportedDataType(
                    UnsupportedDataTypeError::from(data_type.identifier().to_string()),
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
        if input.data_type().identifier() == output.data_type().identifier() {
            rayon_iter_concurrent_limit::iter_concurrent_limit!(
                chunk_limit,
                indices,
                try_for_each,
                |chunk_indices: ArrayIndicesTinyVec| {
                    self.apply_chunk(input, output, &chunk_indices, &progress)
                }
            )?;
        } else {
            rayon_iter_concurrent_limit::iter_concurrent_limit!(
                chunk_limit,
                indices,
                try_for_each,
                |chunk_indices: ArrayIndicesTinyVec| {
                    macro_rules! apply_output {
                        ( $type_in:ty, [$( ( $dt_out:ty, $type_out:ty ) ),* ]) => {
                            match output.data_type().identifier() {
                                $(<$dt_out>::IDENTIFIER => {
                                    self.apply_chunk_convert::<$type_in, $type_out>(input, output, &chunk_indices, &progress)
                                },)*
                                id => panic!("Unsupported output data type: {}", id)
                            }
                        };
                    }
                    macro_rules! apply_input {
                        ([$( ( $dt_in:ty, $type_in:ty ) ),* ]) => {
                            match input.data_type().identifier() {
                                $(<$dt_in>::IDENTIFIER => {
                                    apply_output!($type_in, [
                                        (data_type::BoolDataType, u8),
                                        (data_type::Int8DataType, i8),
                                        (data_type::Int16DataType, i16),
                                        (data_type::Int32DataType, i32),
                                        (data_type::Int64DataType, i64),
                                        (data_type::UInt8DataType, u8),
                                        (data_type::UInt16DataType, u16),
                                        (data_type::UInt32DataType, u32),
                                        (data_type::UInt64DataType, u64),
                                        (data_type::BFloat16DataType, half::bf16),
                                        (data_type::Float16DataType, half::f16),
                                        (data_type::Float32DataType, f32),
                                        (data_type::Float64DataType, f64)
                                    ])
                                },)*
                                id => panic!("Unsupported input data type: {}", id)
                            }
                        };
                    }
                    apply_input!([
                        (data_type::BoolDataType, u8),
                        (data_type::Int8DataType, i8),
                        (data_type::Int16DataType, i16),
                        (data_type::Int32DataType, i32),
                        (data_type::Int64DataType, i64),
                        (data_type::UInt8DataType, u8),
                        (data_type::UInt16DataType, u16),
                        (data_type::UInt32DataType, u32),
                        (data_type::UInt64DataType, u64),
                        (data_type::BFloat16DataType, half::bf16),
                        (data_type::Float16DataType, half::f16),
                        (data_type::Float32DataType, f32),
                        (data_type::Float64DataType, f64)
                    ])
                }
            )?;
        }

        Ok(())
    }
}
