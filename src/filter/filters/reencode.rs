use clap::Parser;
use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{Array, ArrayBytes, ArrayIndicesTinyVec, DataType, Element, ElementOwned},
    array_subset::ArraySubset,
    filesystem::FilesystemStore,
};

use crate::{
    filter::{
        calculate_chunk_limit, filter_error::FilterError, filter_traits::FilterTraits,
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
        chunk_input: &zarrs::array::ChunkRepresentation,
        chunk_output: &zarrs::array::ChunkRepresentation,
    ) -> Result<(), FilterError> {
        for data_type in [chunk_input.data_type(), chunk_output.data_type()] {
            match data_type {
                DataType::Bool
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::BFloat16 => {}
                _ => Err(FilterError::UnsupportedDataType(
                    UnsupportedDataTypeError::from(data_type.to_string()),
                ))?,
            };
        }
        Ok(())
    }

    fn memory_per_chunk(
        &self,
        _chunk_input: &zarrs::array::ChunkRepresentation,
        chunk_output: &zarrs::array::ChunkRepresentation,
    ) -> usize {
        chunk_output.fixed_element_size().unwrap()
    }

    fn apply(
        &self,
        input: &Array<FilesystemStore>,
        output: &mut Array<FilesystemStore>,
        progress_callback: &ProgressCallback,
    ) -> Result<(), FilterError> {
        assert_eq!(input.shape(), output.shape());

        let chunks = ArraySubset::new_with_shape(output.chunk_grid_shape().clone());
        let progress = Progress::new(chunks.num_elements_usize(), progress_callback);

        let chunk_limit = if let Some(chunk_limit) = self.chunk_limit {
            chunk_limit
        } else {
            calculate_chunk_limit(self.memory_per_chunk(
                &input.chunk_array_representation(&vec![0; input.dimensionality()])?,
                &output.chunk_array_representation(&vec![0; input.dimensionality()])?,
            ))?
        };

        let indices = chunks.indices();
        if output.data_type() == input.data_type() {
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
                        ( $type_in:ty, [$( ( $data_type_out:ident, $type_out:ty ) ),* ]) => {
                            match output.data_type() {
                                $(DataType::$data_type_out => { self.apply_chunk_convert::<$type_in, $type_out>(input, output, &chunk_indices, &progress) } ,)*
                                _ => panic!()
                            }
                        };
                    }
                    macro_rules! apply_input {
                    ([$( ( $data_type_in:ident, $type_in:ty ) ),* ]) => {
                            match input.data_type() {
                                $(
                                    DataType::$data_type_in => {
                                        apply_output!($type_in, [
                                            (Bool, u8),
                                            (Int8, i8),
                                            (Int16, i16),
                                            (Int32, i32),
                                            (Int64, i64),
                                            (UInt8, u8),
                                            (UInt16, u16),
                                            (UInt32, u32),
                                            (UInt64, u64),
                                            (BFloat16, half::bf16),
                                            (Float16, half::f16),
                                            (Float32, f32),
                                            (Float64, f64)
                                        ]
                                    )}
                                ,)*
                                _ => panic!()
                            }
                        };
                    }
                    apply_input!([
                        (Bool, u8),
                        (Int8, i8),
                        (Int16, i16),
                        (Int32, i32),
                        (Int64, i64),
                        (UInt8, u8),
                        (UInt16, u16),
                        (UInt32, u32),
                        (UInt64, u64),
                        (BFloat16, half::bf16),
                        (Float16, half::f16),
                        (Float32, f32),
                        (Float64, f64)
                    ])
                }
            )?;
        }

        Ok(())
    }
}
