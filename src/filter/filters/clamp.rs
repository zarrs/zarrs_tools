use clap::Parser;
use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{data_type, Array, ArrayIndicesTinyVec, DataTypeExt},
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

    pub fn apply_elements_inplace<T>(&self, elements: &mut [T]) -> Result<(), FilterError>
    where
        T: bytemuck::Pod + Copy + Send + Sync + PartialOrd,
        f64: AsPrimitive<T>,
    {
        let min: T = self.min.as_();
        let max: T = self.max.as_();
        elements
            .par_iter_mut()
            .for_each(|value| *value = num_traits::clamp(*value, min, max));
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
                macro_rules! apply_input {
                    ( $t_out:ty, [$( ( $dt_in:ty, $t_in:ty ) ),* ]) => {
                        match input.data_type().identifier() {
                            $(<$dt_in>::IDENTIFIER => {
                                let input_output_subset = output.chunk_subset_bounded(&chunk_indices).unwrap();
                                let mut elements_in: Vec<$t_in> =
                                    progress.read(|| input.retrieve_array_subset(&input_output_subset))?;
                                progress.process(|| self.apply_elements_inplace::<$t_in>(&mut elements_in))?;

                                let elements_out = elements_in.iter().map(|v| v.as_()).collect::<Vec<$t_out>>();
                                drop(elements_in);
                                progress.write(|| {
                                    output.store_array_subset(&input_output_subset, elements_out)
                                })?;
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
                ]);

                progress.next();
                Ok::<_, FilterError>(())
            }
        )?;

        Ok(())
    }
}
