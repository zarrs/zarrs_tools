use clap::Parser;
use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{Array, DataType, Element, ElementOwned},
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

    pub fn apply_chunk<TIn, TOut>(
        &self,
        input: &Array<FilesystemStore>,
        output: &Array<FilesystemStore>,
        chunk_indices: &[u64],
        progress: &Progress,
    ) -> Result<(), FilterError>
    where
        TIn: ElementOwned + Send + Sync + AsPrimitive<f64>,
        TOut: Element + Send + Sync + Copy + 'static,
        f64: AsPrimitive<TOut>,
    {
        // Determine the input and output subset
        let input_output_subset = output.chunk_subset_bounded(chunk_indices).unwrap();

        let elements_in =
            progress.read(|| input.retrieve_array_subset_elements::<TIn>(&input_output_subset))?;

        let elements_out = if self.add_first {
            progress.process(|| {
                elements_in
                    .iter()
                    .map(|value| {
                        let value_f64: f64 = value.as_();
                        ((value_f64 + self.add) * self.multiply).as_()
                    })
                    .collect::<Vec<TOut>>()
            })
        } else {
            progress.process(|| self.apply_elements(&elements_in))
        };
        drop(elements_in);

        progress.write(|| {
            output.store_array_subset_elements::<TOut>(&input_output_subset, &elements_out)
        })?;

        progress.next();
        Ok(())
    }

    pub fn apply_elements<TIn, TOut>(&self, elements_in: &[TIn]) -> Vec<TOut>
    where
        TIn: Send + Sync + AsPrimitive<f64>,
        TOut: Send + Sync + Copy + 'static,
        f64: AsPrimitive<TOut>,
    {
        elements_in
            .par_iter()
            .map(|value| {
                let value_f64: f64 = value.as_();
                value_f64.mul_add(self.multiply, self.add).as_()
            })
            .collect::<Vec<TOut>>()
    }
}

impl FilterTraits for Rescale {
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
                _ => Err(UnsupportedDataTypeError::from(data_type.to_string()))?,
            };
        }
        Ok(())
    }

    fn memory_per_chunk(
        &self,
        chunk_input: &zarrs::array::ChunkRepresentation,
        chunk_output: &zarrs::array::ChunkRepresentation,
    ) -> usize {
        chunk_input.fixed_element_size().unwrap() + chunk_output.fixed_element_size().unwrap()
    }

    fn apply(
        &self,
        input: &Array<FilesystemStore>,
        output: &mut Array<FilesystemStore>,
        progress_callback: &ProgressCallback,
    ) -> Result<(), FilterError> {
        assert_eq!(output.shape(), input.shape());

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
        rayon_iter_concurrent_limit::iter_concurrent_limit!(
            chunk_limit,
            indices,
            try_for_each,
            |chunk_indices: Vec<u64>| {
                macro_rules! apply_input {
                    ( $t_out:ty, [$( ( $data_type:ident, $t_in:ty ) ),* ]) => {
                        match input.data_type() {
                            $(DataType::$data_type => { self.apply_chunk::<$t_in, $t_out>(&input, &output, &chunk_indices, &progress) } ,)*
                            _ => panic!()
                        }
                    };
                }
                macro_rules! apply_output {
                    ([$( ( $data_type:ident, $type_out:ty ) ),* ]) => {
                            match output.data_type() {
                                $(
                                    DataType::$data_type => {
                                        apply_input!($type_out, [
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
                apply_output!([
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

        Ok(())
    }
}
