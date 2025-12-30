use std::collections::HashMap;

use clap::Parser;
use num_traits::{AsPrimitive, FromPrimitive};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
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
pub struct DownsampleArguments {
    /// Downsample stride, comma delimited.
    #[arg(required = true, value_delimiter = ',')]
    pub stride: Vec<u64>,
    #[serde(default)]
    /// Perform majority filtering (mode downsampling).
    #[arg(long, default_value_t = false)]
    pub discrete: bool,
}

impl FilterArguments for DownsampleArguments {
    fn name(&self) -> String {
        "downsample".to_string()
    }

    fn init(
        &self,
        common_args: &FilterCommonArguments,
    ) -> Result<Box<dyn FilterTraits>, FilterError> {
        Ok(Box::new(Downsample::new(
            self.stride.clone(),
            self.discrete,
            *common_args.chunk_limit(),
        )))
    }
}

pub struct Downsample {
    stride: Vec<u64>,
    discrete: bool,
    chunk_limit: Option<usize>,
}

impl Downsample {
    pub fn new(stride: Vec<u64>, discrete: bool, chunk_limit: Option<usize>) -> Self {
        Self {
            stride,
            discrete,
            chunk_limit,
        }
    }

    pub fn input_subset(&self, input_shape: &[u64], output_subset: &ArraySubset) -> ArraySubset {
        let input_start = std::iter::zip(output_subset.start(), &self.stride)
            .map(|(start, stride)| start * stride);
        let input_end = itertools::izip!(output_subset.end_exc(), &self.stride, input_shape)
            .map(|(end, stride, shape)| std::cmp::min(end * stride, *shape));
        ArraySubset::new_with_start_end_exc(input_start.collect(), input_end.collect()).unwrap()
    }

    pub fn apply_ndarray_continuous<TIn, TOut>(
        &self,
        input: ndarray::ArrayD<TIn>,
        progress: &Progress,
    ) -> ndarray::ArrayD<TOut>
    where
        TIn: Copy + Send + Sync + AsPrimitive<f64>,
        TOut: Copy + Send + Sync + std::iter::Sum + 'static,
        f64: AsPrimitive<TOut>,
    {
        progress.process(|| {
            let chunk_size: Vec<usize> = std::iter::zip(&self.stride, input.shape())
                .map(|(stride, shape)| std::cmp::min(usize::try_from(*stride).unwrap(), *shape))
                .collect();
            ndarray::Zip::from(input.exact_chunks(chunk_size)).par_map_collect(|chunk| {
                (chunk
                    .iter()
                    .map(|v| AsPrimitive::<f64>::as_(*v))
                    .sum::<f64>()
                    / f64::from_usize(chunk.len()).unwrap())
                .as_()
                // chunk.map(|v| AsPrimitive::<TOut>::as_(*v)).mean().unwrap()
                // chunk.mean().unwrap().as_()
            })
        })
    }

    pub fn apply_ndarray_discrete<TIn, TOut>(
        &self,
        input: ndarray::ArrayD<TIn>,
        progress: &Progress,
    ) -> ndarray::ArrayD<TOut>
    where
        TIn: Copy + Send + Sync + PartialEq + Eq + core::hash::Hash + AsPrimitive<TOut>,
        TOut: Copy + Send + Sync + 'static,
    {
        progress.process(|| {
            let chunk_size: Vec<usize> = std::iter::zip(&self.stride, input.shape())
                .map(|(stride, shape)| std::cmp::min(usize::try_from(*stride).unwrap(), *shape))
                .collect();
            ndarray::Zip::from(input.exact_chunks(chunk_size)).par_map_collect(|chunk| {
                let mut map = HashMap::<TIn, usize>::new();
                for element in &chunk {
                    *map.entry(*element).or_insert(0) += 1;
                }
                map.iter().max_by(|a, b| a.1.cmp(b.1)).unwrap().0.as_()
            })
        })
    }
}

impl FilterTraits for Downsample {
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

    fn memory_per_chunk(&self, _chunk_input: ChunkInfo, chunk_output: ChunkInfo) -> usize {
        debug_assert_eq!(_chunk_input.1.identifier(), chunk_output.1.identifier());
        let input = chunk_output.1.fixed_size().unwrap()
            * usize::try_from(self.stride.iter().product::<u64>()).unwrap();
        let output = chunk_output.1.fixed_size().unwrap();
        input + output
    }

    fn output_shape(&self, input: &Array<FilesystemStore>) -> Option<Vec<u64>> {
        Some(
            std::iter::zip(input.shape(), &self.stride)
                .map(|(shape, stride)| std::cmp::max(shape / stride, 1))
                .collect(),
        )
    }

    fn apply(
        &self,
        input: &Array<FilesystemStore>,
        output: &mut Array<FilesystemStore>,
        progress_callback: &ProgressCallback,
    ) -> Result<(), FilterError> {
        assert_eq!(output.shape(), self.output_shape(input).unwrap());

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
                // Determine the input and output subset
                let output_subset = output.chunk_subset_bounded(&chunk_indices).unwrap();
                let input_subset = self.input_subset(input.shape(), &output_subset);

                macro_rules! downsample {
                    ( $t_in:ty, $t_out:ty ) => {{
                        let input_array: ndarray::ArrayD<$t_in> =
                            progress.read(|| input.retrieve_array_subset(&input_subset))?;
                        let output_array: ndarray::ArrayD<$t_out> = if self.discrete {
                            self.apply_ndarray_discrete(input_array, &progress)
                        } else {
                            self.apply_ndarray_continuous(input_array, &progress)
                        };
                        progress
                            .write(|| output.store_array_subset(&output_subset, output_array))?;
                    }};
                }
                macro_rules! downsample_continuous_only {
                    ( $t_in:ty, $t_out:ty ) => {{
                        let input_array: ndarray::ArrayD<$t_in> =
                            progress.read(|| input.retrieve_array_subset(&input_subset))?;
                        let output_array: ndarray::ArrayD<$t_out> =
                            self.apply_ndarray_continuous(input_array, &progress);
                        progress
                            .write(|| output.store_array_subset(&output_subset, output_array))?;
                    }};
                }
                macro_rules! apply_input {
                    ( $type_out:ty, [$( ( $dt_in:ty, $type_in:ty, $func:ident ) ),* ]) => {
                        match input.data_type().identifier() {
                            $(<$dt_in>::IDENTIFIER => { $func!($type_in, $type_out) },)*
                            id => panic!("Unsupported input data type: {}", id)
                        }
                    };
                }
                macro_rules! apply_output {
                    ([$( ( $dt_out:ty, $type_out:ty ) ),* ]) => {
                        match output.data_type().identifier() {
                            $(<$dt_out>::IDENTIFIER => {
                                apply_input!($type_out, [
                                    (data_type::BoolDataType, u8, downsample),
                                    (data_type::Int8DataType, i8, downsample),
                                    (data_type::Int16DataType, i16, downsample),
                                    (data_type::Int32DataType, i32, downsample),
                                    (data_type::Int64DataType, i64, downsample),
                                    (data_type::UInt8DataType, u8, downsample),
                                    (data_type::UInt16DataType, u16, downsample),
                                    (data_type::UInt32DataType, u32, downsample),
                                    (data_type::UInt64DataType, u64, downsample),
                                    (data_type::BFloat16DataType, half::bf16, downsample_continuous_only),
                                    (data_type::Float16DataType, half::f16, downsample_continuous_only),
                                    (data_type::Float32DataType, f32, downsample_continuous_only),
                                    (data_type::Float64DataType, f64, downsample_continuous_only)
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
