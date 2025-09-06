use clap::Parser;
use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{Array, DataType, Element, ElementOwned, FillValue, FillValueMetadataV3},
    array_subset::ArraySubset,
    filesystem::FilesystemStore,
};

use crate::{
    filter::UnsupportedDataTypeError,
    parse_fill_value,
    progress::{Progress, ProgressCallback},
};

use crate::filter::{
    calculate_chunk_limit, filter_error::FilterError, filter_traits::FilterTraits, FilterArguments,
    FilterCommonArguments,
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

    pub fn apply_elements<TIn, TOut>(
        &self,
        input_elements: &[TIn],
        equal: &TIn,
    ) -> Result<Vec<TOut>, FilterError>
    where
        TIn: ElementOwned + Copy + Send + Sync + PartialEq,
        TOut: Element + Send + Sync + Copy + 'static,
        bool: AsPrimitive<TOut>,
    {
        let output_elements = input_elements
            .into_par_iter()
            .map(|value| (value == equal).as_())
            .collect::<Vec<TOut>>();
        Ok(output_elements)
    }
}

impl FilterTraits for Equal {
    fn is_compatible(
        &self,
        chunk_input: &zarrs::array::ChunkRepresentation,
        chunk_output: &zarrs::array::ChunkRepresentation,
    ) -> Result<(), FilterError> {
        match chunk_input.data_type() {
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
            _ => Err(UnsupportedDataTypeError::from(
                chunk_input.data_type().to_string(),
            ))?,
        };
        match chunk_output.data_type() {
            DataType::Bool | DataType::UInt8 => {}
            _ => Err(UnsupportedDataTypeError::from(
                chunk_output.data_type().to_string(),
            ))?,
        };
        Ok(())
    }

    fn memory_per_chunk(
        &self,
        chunk_input: &zarrs::array::ChunkRepresentation,
        chunk_output: &zarrs::array::ChunkRepresentation,
    ) -> usize {
        chunk_input.fixed_element_size().unwrap() + chunk_output.fixed_element_size().unwrap()
    }

    fn output_data_type(&self, _input: &Array<FilesystemStore>) -> Option<(DataType, FillValue)> {
        Some((DataType::Bool, FillValue::from(false)))
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

        let value = input
            .data_type()
            .fill_value_from_metadata(&self.value)
            .unwrap();

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
                let input_output_subset = output.chunk_subset_bounded(&chunk_indices).unwrap();
                macro_rules! apply_input {
                    ( $t_out:ty, [$( ( $data_type_in:ident, $t_in:ty ) ),* ]) => {
                        match input.data_type() {
                            $(DataType::$data_type_in => {
                                let input_elements =
                                    progress.read(|| input.retrieve_array_subset_elements::<$t_in>(&input_output_subset))?;

                                let output_elements =
                                    progress.process(|| {
                                        let value = <$t_in>::from_ne_bytes(value.as_ne_bytes().try_into().unwrap());
                                        self.apply_elements::<$t_in, $t_out>(&input_elements, &value)
                                    })?;
                                drop(input_elements);

                                progress.write(|| {
                                    output.store_array_subset_elements::<$t_out>(&input_output_subset, &output_elements)
                                })?;

                                progress.next();
                                Ok(())
                            } ,)*
                            _ => panic!()
                        }
                    };
                }
                macro_rules! apply_output {
                    ([$( ( $data_type_out:ident, $type_out:ty ) ),* ]) => {
                            match output.data_type() {
                                $(
                                    DataType::$data_type_out => {
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
                    (Bool, u8), // bool != bytemuck::Pod, but apply_chunk only stores 0 or 1, so can store as u8
                    (UInt8, u8)
                ])
            }
        )
    }
}
