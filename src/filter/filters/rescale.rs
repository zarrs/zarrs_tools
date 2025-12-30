use clap::Parser;
use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{data_type, Array, ArrayIndicesTinyVec, DataTypeExt, Element, ElementOwned},
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

        let elements_in: Vec<TIn> =
            progress.read(|| input.retrieve_array_subset(&input_output_subset))?;

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

        progress.write(|| output.store_array_subset(&input_output_subset, elements_out))?;

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
                macro_rules! apply_input {
                    ( $t_out:ty, [$( ( $dt_in:ty, $t_in:ty ) ),* ]) => {
                        match input.data_type().identifier() {
                            $(<$dt_in>::IDENTIFIER => {
                                self.apply_chunk::<$t_in, $t_out>(&input, &output, &chunk_indices, &progress)
                            },)*
                            id => panic!("Unsupported input data type: {}", id)
                        }
                    };
                }
                macro_rules! apply_output {
                    ([$( ( $dt_out:ty, $type_out:ty ) ),* ]) => {
                        match output.data_type().identifier() {
                            $(<$dt_out>::IDENTIFIER => {
                                apply_input!($type_out, [
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
                            id => panic!("Unsupported output data type: {}", id)
                        }
                    };
                }
                apply_output!([
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

        Ok(())
    }
}
