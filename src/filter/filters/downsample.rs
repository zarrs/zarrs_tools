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
        FilterArguments, FilterCommonArguments,
    },
    progress::{Progress, ProgressCallback},
    type_dispatch::{retrieve_ndarray_as, store_ndarray_from, IntermediateType},
    UnsupportedDataTypeError,
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

    /// Apply continuous downsampling using type T as intermediate.
    fn apply_continuous<T>(
        &self,
        input: &Array<FilesystemStore>,
        output: &Array<FilesystemStore>,
        input_subset: &ArraySubset,
        output_subset: &ArraySubset,
        progress: &Progress,
    ) -> Result<(), FilterError>
    where
        T: Copy + Send + Sync + AsPrimitive<f64> + std::iter::Sum + 'static,
        f64: AsPrimitive<T>,
        u8: AsPrimitive<T>,
        i8: AsPrimitive<T>,
        i16: AsPrimitive<T>,
        i32: AsPrimitive<T>,
        i64: AsPrimitive<T>,
        u16: AsPrimitive<T>,
        u32: AsPrimitive<T>,
        u64: AsPrimitive<T>,
        half::f16: AsPrimitive<T>,
        half::bf16: AsPrimitive<T>,
        f32: AsPrimitive<T>,
        T: AsPrimitive<u8>,
        T: AsPrimitive<i8>,
        T: AsPrimitive<i16>,
        T: AsPrimitive<i32>,
        T: AsPrimitive<i64>,
        T: AsPrimitive<u16>,
        T: AsPrimitive<u32>,
        T: AsPrimitive<u64>,
        T: AsPrimitive<half::f16>,
        T: AsPrimitive<half::bf16>,
        T: AsPrimitive<f32>,
        T: AsPrimitive<f64>,
    {
        let (input_array, retrieve_timing) = retrieve_ndarray_as::<T, _>(input, input_subset)?;
        progress.add_retrieve_timing(retrieve_timing);
        let output_array: ndarray::ArrayD<T> = self.apply_ndarray_continuous(input_array, progress);
        let store_timing = store_ndarray_from::<T, _>(output, output_subset, output_array)?;
        progress.add_store_timing(store_timing);
        Ok(())
    }

    /// Apply discrete downsampling using type T as intermediate.
    fn apply_discrete<T>(
        &self,
        input: &Array<FilesystemStore>,
        output: &Array<FilesystemStore>,
        input_subset: &ArraySubset,
        output_subset: &ArraySubset,
        progress: &Progress,
    ) -> Result<(), FilterError>
    where
        T: Copy + Send + Sync + PartialEq + Eq + core::hash::Hash + AsPrimitive<T> + 'static,
        u8: AsPrimitive<T>,
        i8: AsPrimitive<T>,
        i16: AsPrimitive<T>,
        i32: AsPrimitive<T>,
        i64: AsPrimitive<T>,
        u16: AsPrimitive<T>,
        u32: AsPrimitive<T>,
        u64: AsPrimitive<T>,
        half::f16: AsPrimitive<T>,
        half::bf16: AsPrimitive<T>,
        f32: AsPrimitive<T>,
        f64: AsPrimitive<T>,
        T: AsPrimitive<u8>,
        T: AsPrimitive<i8>,
        T: AsPrimitive<i16>,
        T: AsPrimitive<i32>,
        T: AsPrimitive<i64>,
        T: AsPrimitive<u16>,
        T: AsPrimitive<u32>,
        T: AsPrimitive<u64>,
        T: AsPrimitive<half::f16>,
        T: AsPrimitive<half::bf16>,
        T: AsPrimitive<f32>,
        T: AsPrimitive<f64>,
    {
        let (input_array, retrieve_timing) = retrieve_ndarray_as::<T, _>(input, input_subset)?;
        progress.add_retrieve_timing(retrieve_timing);
        let output_array: ndarray::ArrayD<T> = self.apply_ndarray_discrete(input_array, progress);
        let store_timing = store_ndarray_from::<T, _>(output, output_subset, output_array)?;
        progress.add_store_timing(store_timing);
        Ok(())
    }

    pub fn apply_chunk(
        &self,
        input: &Array<FilesystemStore>,
        output: &Array<FilesystemStore>,
        input_subset: &ArraySubset,
        output_subset: &ArraySubset,
        progress: &Progress,
    ) -> Result<(), FilterError> {
        if self.discrete {
            // For discrete mode, use integer intermediates based on input type
            match IntermediateType::for_data_type(input.data_type()) {
                IntermediateType::I32 => {
                    self.apply_discrete::<i32>(input, output, input_subset, output_subset, progress)
                }
                IntermediateType::I64 => {
                    self.apply_discrete::<i64>(input, output, input_subset, output_subset, progress)
                }
                IntermediateType::U32 => {
                    self.apply_discrete::<u32>(input, output, input_subset, output_subset, progress)
                }
                IntermediateType::U64 => {
                    self.apply_discrete::<u64>(input, output, input_subset, output_subset, progress)
                }
                // Float types: fall back to continuous (discrete doesn't make sense for floats)
                IntermediateType::F32 => self.apply_continuous::<f32>(
                    input,
                    output,
                    input_subset,
                    output_subset,
                    progress,
                ),
                IntermediateType::F64 => self.apply_continuous::<f64>(
                    input,
                    output,
                    input_subset,
                    output_subset,
                    progress,
                ),
            }
        } else {
            // For continuous mode, use float intermediate based on input type
            match IntermediateType::for_data_type(input.data_type()) {
                // Use f32 for small types where f32 precision is sufficient
                IntermediateType::F32 => self.apply_continuous::<f32>(
                    input,
                    output,
                    input_subset,
                    output_subset,
                    progress,
                ),
                // Use f64 for f64 and all other intermediate types
                IntermediateType::F64
                | IntermediateType::I32
                | IntermediateType::I64
                | IntermediateType::U32
                | IntermediateType::U64 => self.apply_continuous::<f64>(
                    input,
                    output,
                    input_subset,
                    output_subset,
                    progress,
                ),
            }
        }
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
                let output_subset = output.chunk_subset_bounded(&chunk_indices).unwrap();
                let input_subset = self.input_subset(input.shape(), &output_subset);
                self.apply_chunk(input, output, &input_subset, &output_subset, &progress)?;
                progress.next();
                Ok::<_, FilterError>(())
            }
        )?;

        Ok(())
    }
}
