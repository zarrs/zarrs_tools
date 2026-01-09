use clap::Parser;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{data_type, Array, ArrayIndicesTinyVec, ArraySubset, DataTypeExt, FillValueMetadataV3},
    filesystem::FilesystemStore,
    plugin::ExtensionIdentifier,
};

use crate::{
    parse_fill_value,
    progress::{Progress, ProgressCallback},
    type_dispatch::{retrieve_as, store_from, IntermediateType},
    UnsupportedDataTypeError,
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

    fn apply_elements<T>(elements: Vec<T>, value: T, replace: T) -> Vec<T>
    where
        T: PartialEq + Copy + Send + Sync,
    {
        elements
            .into_par_iter()
            .map(|v| if v == value { replace } else { v })
            .collect()
    }

    fn convert_to_intermediate_bytes(
        &self,
        input: &Array<FilesystemStore>,
        output: &Array<FilesystemStore>,
        intermediate_type: IntermediateType,
    ) -> Result<(Vec<u8>, Vec<u8>), FilterError> {
        use num_traits::AsPrimitive;

        // Get original fill values
        let value_fv = input
            .named_data_type()
            .fill_value_from_metadata(&self.value)
            .expect("value not compatible with input");
        let replace_fv = output
            .named_data_type()
            .fill_value_from_metadata(&self.replace)
            .expect("replace not compatible with output");

        macro_rules! type_list {
            ($macro:ident, $intermediate:ty, $fv:expr, $array:expr) => {
                $macro!(
                    $intermediate,
                    $fv,
                    $array,
                    [
                        (data_type::BoolDataType, u8),
                        (data_type::Int8DataType, i8),
                        (data_type::Int16DataType, i16),
                        (data_type::Int32DataType, i32),
                        (data_type::Int64DataType, i64),
                        (data_type::UInt8DataType, u8),
                        (data_type::UInt16DataType, u16),
                        (data_type::UInt32DataType, u32),
                        (data_type::UInt64DataType, u64),
                        (data_type::Float16DataType, half::f16),
                        (data_type::Float32DataType, f32),
                        (data_type::Float64DataType, f64),
                        (data_type::BFloat16DataType, half::bf16)
                    ]
                )
            };
        }

        // Convert fill value from array's data type to intermediate type
        macro_rules! convert_fv {
            ($intermediate:ty, $fv:expr, $array:expr, [$( ( $dt:ty, $t:ty ) ),* ]) => {
                match $array.data_type().identifier() {
                    $(<$dt>::IDENTIFIER => {
                        let v = <$t>::from_ne_bytes($fv.as_ne_bytes().try_into().unwrap());
                        let converted: $intermediate = v.as_();
                        converted.to_ne_bytes().to_vec()
                    },)*
                    id => return Err(UnsupportedDataTypeError::from(id.to_string()).into()),
                }
            };
        }

        macro_rules! convert_both {
            ($intermediate:ty) => {{
                let v = type_list!(convert_fv, $intermediate, value_fv, input);
                let r = type_list!(convert_fv, $intermediate, replace_fv, output);
                (v, r)
            }};
        }

        let (value_bytes, replace_bytes) = match intermediate_type {
            IntermediateType::F32 => convert_both!(f32),
            IntermediateType::F64 => convert_both!(f64),
            IntermediateType::I32 => convert_both!(i32),
            IntermediateType::I64 => convert_both!(i64),
            IntermediateType::U32 => convert_both!(u32),
            IntermediateType::U64 => convert_both!(u64),
        };

        Ok((value_bytes, replace_bytes))
    }

    pub fn apply_chunk(
        &self,
        input: &Array<FilesystemStore>,
        output: &Array<FilesystemStore>,
        subset: &ArraySubset,
        value_bytes: &[u8],
        replace_bytes: &[u8],
        progress: &Progress,
    ) -> Result<(), FilterError> {
        macro_rules! apply {
            ($t:ty) => {{
                let value = <$t>::from_ne_bytes(value_bytes.try_into().unwrap());
                let replace = <$t>::from_ne_bytes(replace_bytes.try_into().unwrap());
                let (elements, retrieve_timing) = retrieve_as::<$t, _>(input, subset)?;
                progress.add_retrieve_timing(retrieve_timing);
                let result = progress.process(|| Self::apply_elements(elements, value, replace));
                let store_timing = store_from(output, subset, result)?;
                progress.add_store_timing(store_timing);
            }};
        }

        match IntermediateType::for_data_type(input.data_type()) {
            IntermediateType::F32 => apply!(f32),
            IntermediateType::F64 => apply!(f64),
            IntermediateType::I32 => apply!(i32),
            IntermediateType::I64 => apply!(i64),
            IntermediateType::U32 => apply!(u32),
            IntermediateType::U64 => apply!(u64),
        }
        Ok(())
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

        // Convert value and replace to intermediate type bytes
        let intermediate_type = IntermediateType::for_data_type(input.data_type());
        let (value_bytes, replace_bytes) =
            self.convert_to_intermediate_bytes(input, output, intermediate_type)?;

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
                let subset = output.chunk_subset_bounded(&chunk_indices).unwrap();
                self.apply_chunk(
                    input,
                    output,
                    &subset,
                    &value_bytes,
                    &replace_bytes,
                    &progress,
                )?;
                progress.next();
                Ok::<_, FilterError>(())
            }
        )?;

        Ok(())
    }
}
