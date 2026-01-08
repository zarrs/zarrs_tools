use std::time::Instant;

use clap::Parser;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{
        data_type, Array, ArrayIndicesTinyVec, DataTypeExt, ElementOwned, FillValue,
        FillValueMetadataV3, NamedDataType,
    },
    array_subset::ArraySubset,
    filesystem::FilesystemStore,
    plugin::ExtensionIdentifier,
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
    pub value: FillValueMetadataV3,
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
    value: FillValueMetadataV3,
    chunk_limit: Option<usize>,
}

impl Equal {
    pub fn new(value: FillValueMetadataV3, chunk_limit: Option<usize>) -> Self {
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

impl FilterTraits for Equal {
    fn is_compatible(
        &self,
        chunk_input: ChunkInfo,
        chunk_output: ChunkInfo,
    ) -> Result<(), FilterError> {
        const SUPPORTED_INPUT_TYPES: &[&str] = &[
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
        const SUPPORTED_OUTPUT_TYPES: &[&str] = &[
            data_type::BoolDataType::IDENTIFIER,
            data_type::UInt8DataType::IDENTIFIER,
        ];
        if !SUPPORTED_INPUT_TYPES.contains(&chunk_input.1.identifier()) {
            Err(UnsupportedDataTypeError::from(
                chunk_input.1.identifier().to_string(),
            ))?;
        }
        if !SUPPORTED_OUTPUT_TYPES.contains(&chunk_output.1.identifier()) {
            Err(UnsupportedDataTypeError::from(
                chunk_output.1.identifier().to_string(),
            ))?;
        }
        Ok(())
    }

    fn memory_per_chunk(&self, chunk_input: ChunkInfo, chunk_output: ChunkInfo) -> usize {
        chunk_input.1.fixed_size().unwrap() + chunk_output.1.fixed_size().unwrap()
    }

    fn output_data_type(
        &self,
        _input: &Array<FilesystemStore>,
    ) -> Option<(NamedDataType, FillValue)> {
        Some((data_type::bool().to_named(), FillValue::from(false)))
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

        let value = input.data_type().fill_value(&self.value).unwrap();

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
                        match input.data_type().identifier() {
                            $(<$dt_in>::IDENTIFIER => {
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
                            },)*
                            id => Err(UnsupportedDataTypeError::from(id.to_string()).into())
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
                match output.data_type().identifier() {
                    data_type::BoolDataType::IDENTIFIER => {
                        output.store_array_subset(&input_output_subset, bool_result)?;
                    }
                    data_type::UInt8DataType::IDENTIFIER => {
                        let u8_result: Vec<u8> = bytemuck::cast_vec(bool_result);
                        output.store_array_subset(&input_output_subset, u8_result)?;
                    }
                    id => {
                        return Err(UnsupportedDataTypeError::from(id.to_string()).into());
                    }
                }
                progress.add_write_duration(start.elapsed());

                progress.next();
                Ok(())
            }
        )
    }
}
