use std::collections::HashMap;

use clap::Parser;
use num_traits::{AsPrimitive, FromPrimitive};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{Array, ArrayIndicesTinyVec, DataType},
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
        _chunk_input: &zarrs::array::ChunkRepresentation,
        chunk_output: &zarrs::array::ChunkRepresentation,
    ) -> usize {
        debug_assert_eq!(_chunk_input.data_type(), chunk_output.data_type());
        let input = chunk_output.fixed_element_size().unwrap()
            * usize::try_from(self.stride.iter().product::<u64>()).unwrap();
        let output = chunk_output.fixed_element_size().unwrap();
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
                    ( $type_out:ty, [$( ( $data_type_in:ident, $type_in:ty,  $func:ident ) ),* ]) => {
                        match input.data_type() {
                            $(DataType::$data_type_in => { $func!($type_in, $type_out) } ,)*
                            _ => panic!("unsupported data type")
                        }
                    };
                }
                macro_rules! apply_output {
                ([$( ( $data_type_out:ident, $type_out:ty ) ),* ]) => {
                        match output.data_type() {
                            $(
                                DataType::$data_type_out => {
                                    apply_input!($type_out, [
                                        (Bool, u8, downsample),
                                        (Int8, i8, downsample),
                                        (Int16, i16, downsample),
                                        (Int32, i32, downsample),
                                        (Int64, i64, downsample),
                                        (UInt8, u8, downsample),
                                        (UInt16, u16, downsample),
                                        (UInt32, u32, downsample),
                                        (UInt64, u64, downsample),
                                        (BFloat16, half::bf16, downsample_continuous_only),
                                        (Float16, half::f16, downsample_continuous_only),
                                        (Float32, f32, downsample_continuous_only),
                                        (Float64, f64, downsample_continuous_only)
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
                ]);

                progress.next();
                Ok::<_, FilterError>(())
            }
        )?;

        Ok(())
    }
}
