use std::time::Instant;

use clap::Parser;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{
        data_type, Array, ArrayIndicesTinyVec, ArraySubset, DataType, DataTypeExt, ElementOwned,
        FillValue,
    },
    filesystem::FilesystemStore,
    metadata::FillValueMetadata,
    plugin::ZarrVersion,
};

use crate::{
    parse_fill_value,
    progress::{Progress, ProgressCallback},
    UnsupportedDataTypeError,
};

use crate::filter::{
    calculate_chunk_limit,
    filter_error::FilterError,
    filter_traits::{ChunkInfo, FilterTraits},
    FilterArguments, FilterCommonArguments,
};

#[derive(Debug, Clone, Parser, Serialize, Deserialize)]
pub struct EqualArguments {
    /// The value to compare against.
    ///
    /// The value must be compatible with the data type.
    ///
    /// Examples:
    ///   int/uint: 0
    ///   float: 0.0 "NaN" "Infinity" "-Infinity"
    ///   r*: "[0, 255]"
    #[arg(allow_hyphen_values(true), value_parser = parse_fill_value)]
    pub value: FillValueMetadata,
}

impl FilterArguments for EqualArguments {
    fn name(&self) -> String {
        "equal".to_string()
    }

    fn init(
        &self,
        common_args: &FilterCommonArguments,
    ) -> Result<Box<dyn FilterTraits>, FilterError> {
        Ok(Box::new(Equal::new(
            self.value.clone(),
            *common_args.chunk_limit(),
        )))
    }
}

pub struct Equal {
    value: FillValueMetadata,
    chunk_limit: Option<usize>,
}

impl Equal {
    pub fn new(value: FillValueMetadata, chunk_limit: Option<usize>) -> Self {
        Self { value, chunk_limit }
    }

    /// Compare input elements against a value, returning a Vec<bool>.
    /// This is generic only over the input type, avoiding combinatorial explosion.
    fn compare_elements<T>(input_elements: Vec<T>, equal: &T) -> Vec<bool>
    where
        T: ElementOwned + Copy + Send + Sync + PartialEq,
    {
        input_elements
            .into_par_iter()
            .map(|value| value == *equal)
            .collect()
    }
}

fn is_supported_input_type(dt: &DataType) -> bool {
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

fn is_supported_output_type(dt: &DataType) -> bool {
    dt.is::<data_type::BoolDataType>() || dt.is::<data_type::UInt8DataType>()
}

impl FilterTraits for Equal {
    fn is_compatible(
        &self,
        chunk_input: ChunkInfo,
        chunk_output: ChunkInfo,
    ) -> Result<(), FilterError> {
        if !is_supported_input_type(chunk_input.1) {
            Err(UnsupportedDataTypeError::from(format!(
                "{:?}",
                chunk_input.1
            )))?;
        }
        if !is_supported_output_type(chunk_output.1) {
            Err(UnsupportedDataTypeError::from(format!(
                "{:?}",
                chunk_output.1
            )))?;
        }
        Ok(())
    }

    fn memory_per_chunk(&self, chunk_input: ChunkInfo, chunk_output: ChunkInfo) -> usize {
        chunk_input.1.fixed_size().unwrap() + chunk_output.1.fixed_size().unwrap()
    }

    fn output_data_type(&self, _input: &Array<FilesystemStore>) -> Option<(DataType, FillValue)> {
        Some((data_type::bool(), FillValue::from(false)))
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

        let value = input
            .data_type()
            .fill_value(&self.value, ZarrVersion::V3)
            .unwrap();

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
                let input_output_subset = output.chunk_subset_bounded(&chunk_indices).unwrap();

                macro_rules! apply {
                    ([$( ( $dt_in:ty, $t_in:ty ) ),* ]) => {
                        {
                            let dt = input.data_type();
                            $(if dt.is::<$dt_in>() {
                                let start = Instant::now();
                                let input_elements: Vec<$t_in> =
                                    input.retrieve_array_subset(&input_output_subset)?;
                                progress.add_read_duration(start.elapsed());

                                let start = Instant::now();
                                let compare_value = <$t_in>::from_ne_bytes(
                                    value.as_ne_bytes().try_into().unwrap()
                                );
                                let bool_result = Self::compare_elements(input_elements, &compare_value);
                                progress.add_process_duration(start.elapsed());
                                Ok::<_, FilterError>(bool_result)
                            } else)*
                            { Err(UnsupportedDataTypeError::from(format!("{:?}", dt)).into()) }
                        }
                    };
                }

                let bool_result: Vec<bool> = apply!([
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
                ])?;

                // Store output based on output data type
                let start = Instant::now();
                let out_dt = output.data_type();
                if out_dt.is::<data_type::BoolDataType>() {
                    output.store_array_subset(&input_output_subset, bool_result)?;
                } else if out_dt.is::<data_type::UInt8DataType>() {
                    let u8_result: Vec<u8> = bytemuck::cast_vec(bool_result);
                    output.store_array_subset(&input_output_subset, u8_result)?;
                } else {
                    return Err(UnsupportedDataTypeError::from(format!("{:?}", out_dt)).into());
                }
                progress.add_write_duration(start.elapsed());

                progress.next();
                Ok(())
            }
        )
    }
}
