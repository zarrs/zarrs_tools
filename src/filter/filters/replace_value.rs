use clap::Parser;
use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{data_type, Array, ArrayIndicesTinyVec, DataTypeExt, FillValueMetadataV3},
    array_subset::ArraySubset,
    filesystem::FilesystemStore,
    plugin::ExtensionIdentifier,
};

use crate::{
    filter::UnsupportedDataTypeError,
    parse_fill_value,
    progress::{Progress, ProgressCallback},
};

use crate::filter::{
    calculate_chunk_limit,
    filter_error::FilterError,
    filter_traits::{ChunkInfo, FilterTraits},
    FilterArguments, FilterCommonArguments,
};

#[derive(Debug, Clone, Parser, Serialize, Deserialize)]
pub struct ReplaceValueArguments {
    /// The value to change.
    ///
    /// The value must be compatible with the data type.
    ///
    /// Examples:
    ///   int/uint: 0
    ///   float: 0.0 "NaN" "Infinity" "-Infinity"
    ///   r*: "[0, 255]"
    #[arg(allow_hyphen_values(true), value_parser = parse_fill_value)]
    pub value: FillValueMetadataV3,
    /// The replacement value.
    ///
    /// The value must be compatible with the data type.
    ///
    /// Examples:
    ///   int/uint: 0
    ///   float: 0.0 "NaN" "Infinity" "-Infinity"
    ///   r*: "[0, 255]"
    #[arg(allow_hyphen_values(true), value_parser = parse_fill_value)]
    pub replace: FillValueMetadataV3,
}

impl FilterArguments for ReplaceValueArguments {
    fn name(&self) -> String {
        "replace_value".to_string()
    }

    fn init(
        &self,
        common_args: &FilterCommonArguments,
    ) -> Result<Box<dyn FilterTraits>, FilterError> {
        Ok(Box::new(ReplaceValue::new(
            self.value.clone(),
            self.replace.clone(),
            *common_args.chunk_limit(),
        )))
    }
}

pub struct ReplaceValue {
    value: FillValueMetadataV3,
    replace: FillValueMetadataV3,
    chunk_limit: Option<usize>,
}

impl ReplaceValue {
    pub fn new(
        value: FillValueMetadataV3,
        replace: FillValueMetadataV3,
        chunk_limit: Option<usize>,
    ) -> Self {
        Self {
            value,
            replace,
            chunk_limit,
        }
    }

    pub fn apply_elements<TIn, TOut>(
        &self,
        input_elements: &[TIn],
        value: TIn,
        replace: TOut,
    ) -> Result<Vec<TOut>, FilterError>
    where
        TIn: bytemuck::Pod + Copy + Send + Sync + PartialEq + AsPrimitive<TOut>,
        TOut: bytemuck::Pod + Send + Sync,
    {
        let output_elements = input_elements
            .into_par_iter()
            .map(|v_in| if v_in == &value { replace } else { v_in.as_() })
            .collect::<Vec<TOut>>();
        Ok(output_elements)
    }
}

impl FilterTraits for ReplaceValue {
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

        let value = input
            .named_data_type()
            .fill_value_from_metadata(&self.value)
            .expect("value not compatible with input image");
        let replace = output
            .named_data_type()
            .fill_value_from_metadata(&self.replace)
            .expect("replace not compatible with output image");

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
                macro_rules! apply_input {
                    ( $t_out:ty, [$( ( $dt_in:ty, $t_in:ty ) ),* ]) => {
                        match input.data_type().identifier() {
                            $(<$dt_in>::IDENTIFIER => {
                                let input_elements: Vec<$t_in> =
                                    progress.read(|| input.retrieve_array_subset(&input_output_subset))?;

                                let output_elements =
                                    progress.process(|| {
                                        let value = <$t_in>::from_ne_bytes(value.as_ne_bytes().try_into().unwrap());
                                        let replace = <$t_out>::from_ne_bytes(replace.as_ne_bytes().try_into().unwrap());
                                        self.apply_elements::<$t_in, $t_out>(&input_elements, value, replace)
                                    })?;
                                drop(input_elements);

                                progress.write(|| {
                                    output.store_array_subset(&input_output_subset, output_elements)
                                })?;

                                progress.next();
                                Ok(())
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
        )
    }
}
