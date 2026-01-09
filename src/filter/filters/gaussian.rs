use std::iter::Sum;

use clap::Parser;
use itertools::Itertools;
use ndarray::ArrayD;
use num_traits::Float;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{data_type, Array, ArrayIndicesTinyVec, ArraySubset, DataTypeExt},
    filesystem::FilesystemStore,
    plugin::ExtensionIdentifier,
};

use crate::{
    filter::{
        calculate_chunk_limit,
        filter_error::FilterError,
        filter_traits::{ChunkInfo, FilterTraits},
        kernel::apply_1d_kernel,
        ArraySubsetOverlap, FilterArguments, FilterCommonArguments,
    },
    progress::{Progress, ProgressCallback},
    type_dispatch::{retrieve_ndarray_as, store_ndarray_from, IntermediateType},
    UnsupportedDataTypeError,
};

#[derive(Debug, Clone, Parser, Serialize, Deserialize, Default)]
pub struct GaussianArguments {
    /// Gaussian kernel sigma per axis, comma delimited.
    #[arg(required = true, value_delimiter = ',')]
    sigma: Vec<f32>,
    /// Gaussian kernel half size per axis, comma delimited. Kernel is 2 x half size + 1.
    #[arg(required = true, value_delimiter = ',')]
    kernel_half_size: Vec<u64>,
}

impl FilterArguments for GaussianArguments {
    fn name(&self) -> String {
        "gaussian".to_string()
    }

    fn init(
        &self,
        common_args: &FilterCommonArguments,
    ) -> Result<Box<dyn FilterTraits>, FilterError> {
        Ok(Box::new(Gaussian::new(
            self.sigma.clone(),
            self.kernel_half_size.clone(),
            *common_args.chunk_limit(),
        )))
    }
}

pub struct Gaussian {
    kernel_f32: Vec<ndarray::Array1<f32>>,
    kernel_f64: Vec<ndarray::Array1<f64>>,
    kernel_half_size: Vec<u64>,
    chunk_limit: Option<usize>,
}

impl Gaussian {
    pub fn new(sigma: Vec<f32>, kernel_half_size: Vec<u64>, chunk_limit: Option<usize>) -> Self {
        let kernel_f32 = std::iter::zip(&sigma, &kernel_half_size)
            .map(|(sigma, kernel_half_size)| {
                create_sampled_gaussian_kernel(*sigma, *kernel_half_size)
            })
            .collect_vec();
        let kernel_f64 = std::iter::zip(&sigma, &kernel_half_size)
            .map(|(sigma, kernel_half_size)| {
                create_sampled_gaussian_kernel(*sigma as f64, *kernel_half_size)
            })
            .collect_vec();
        Self {
            kernel_f32,
            kernel_f64,
            kernel_half_size,
            chunk_limit,
        }
    }

    pub fn kernel_half_size(&self) -> &[u64] {
        &self.kernel_half_size
    }

    pub fn apply_chunk(
        &self,
        input: &Array<FilesystemStore>,
        output: &Array<FilesystemStore>,
        chunk_indices: &[u64],
        progress: &Progress,
    ) -> Result<(), FilterError> {
        let subset_output = output.chunk_subset_bounded(chunk_indices).unwrap();
        let subset_overlap =
            ArraySubsetOverlap::new(input.shape(), &subset_output, &self.kernel_half_size);

        // Choose intermediate type based on input data type
        let use_f64 = matches!(
            IntermediateType::for_data_type(input.data_type()),
            IntermediateType::F64 | IntermediateType::I64 | IntermediateType::U64
        );

        if use_f64 {
            let (input_array, retrieve_timing) =
                retrieve_ndarray_as::<f64, _>(input, subset_overlap.subset_input())?;
            progress.add_retrieve_timing(retrieve_timing);
            let output_array = progress.process(|| {
                let processed = Self::apply_ndarray_with_kernel(input_array, &self.kernel_f64);
                subset_overlap.extract_subset(&processed)
            });
            let store_timing = store_ndarray_from::<f64, _>(output, &subset_output, output_array)?;
            progress.add_store_timing(store_timing);
        } else {
            let (input_array, retrieve_timing) =
                retrieve_ndarray_as::<f32, _>(input, subset_overlap.subset_input())?;
            progress.add_retrieve_timing(retrieve_timing);
            let output_array = progress.process(|| {
                let processed = Self::apply_ndarray_with_kernel(input_array, &self.kernel_f32);
                subset_overlap.extract_subset(&processed)
            });
            let store_timing = store_ndarray_from::<f32, _>(output, &subset_output, output_array)?;
            progress.add_store_timing(store_timing);
        }

        progress.next();
        Ok(())
    }

    /// Apply gaussian filter to an f32 ndarray
    pub fn apply_ndarray(&self, input: ndarray::ArrayD<f32>) -> ndarray::ArrayD<f32> {
        Self::apply_ndarray_with_kernel(input, &self.kernel_f32)
    }

    fn apply_ndarray_with_kernel<T>(
        mut input: ndarray::ArrayD<T>,
        kernel: &[ndarray::Array1<T>],
    ) -> ndarray::ArrayD<T>
    where
        T: Float + Send + Sync + Sum + Copy,
    {
        let mut gaussian = ArrayD::<T>::zeros(input.shape());
        for (dim, kernel_item) in kernel.iter().enumerate().take(input.ndim()) {
            apply_1d_kernel(dim, kernel_item, &input, &mut gaussian);
            if dim + 1 != input.ndim() {
                std::mem::swap(&mut input, &mut gaussian);
            }
        }
        gaussian
    }
}

impl FilterTraits for Gaussian {
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
        let num_input_elements = usize::try_from(
            chunk_output
                .0
                .iter()
                .zip(&self.kernel_half_size)
                .map(|(s, kernel_half_size)| s.get() + kernel_half_size * 2)
                .product::<u64>(),
        )
        .unwrap();
        let num_output_elements =
            usize::try_from(chunk_output.0.iter().map(|s| s.get()).product::<u64>()).unwrap();
        num_input_elements * (chunk_input.1.fixed_size().unwrap() + core::mem::size_of::<f32>() * 2)
            + num_output_elements
                * (core::mem::size_of::<f32>() + chunk_output.1.fixed_size().unwrap())
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
                self.apply_chunk(input, output, &chunk_indices, &progress)
            }
        )?;

        Ok(())
    }
}

fn create_sampled_gaussian_kernel<T>(sigma: T, kernel_half_size: u64) -> ndarray::Array1<T>
where
    T: Float,
{
    if sigma == T::zero() {
        ndarray::Array1::<T>::from_vec(vec![T::one()])
    } else {
        let two = T::from(2.0).unwrap();
        let t = sigma * sigma;
        let scale = T::one() / (two * T::from(std::f64::consts::PI).unwrap() * t).sqrt();
        let kernel_half_elements = (0..=kernel_half_size).map(move |n| {
            let n_sq = T::from(n * n).unwrap();
            scale * (-(n_sq / (two * t))).exp()
        });
        let kernel_elements = kernel_half_elements
            .clone()
            .rev()
            .chain(kernel_half_elements.skip(1))
            .collect::<Vec<_>>();
        ndarray::Array1::<T>::from_vec(kernel_elements)
    }
}

#[cfg(test)]
mod tests {
    use crate::progress::ProgressStats;

    use super::*;
    use std::error::Error;
    use zarrs::array::{data_type, ArrayBuilder};

    #[test]
    fn gaussian() -> Result<(), Box<dyn Error>> {
        let path = tempfile::TempDir::new()?;
        let store = FilesystemStore::new(path.path())?;
        let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::float32(), 0.0f32)
            .build(store.into(), "/")?;
        let array_subset = array.subset_all();
        array.store_array_subset(
            &array_subset,
            &(0..array_subset.num_elements_usize())
                .map(|u| ((u / array.shape()[1] as usize) + u % array.shape()[1] as usize) as f32)
                .collect::<Vec<f32>>(),
        )?;

        let elements: ndarray::ArrayD<f32> = array.retrieve_array_subset(&array_subset)?;
        println!("{elements:?}");

        let path = tempfile::TempDir::new()?;
        let store: FilesystemStore = FilesystemStore::new(path.path())?;
        let mut array_output = array.builder().build(store.into(), "/")?;
        let progress_callback = |_stats: ProgressStats| {};
        Gaussian::new(vec![1.0; 2], vec![3; 2], None).apply(
            &array,
            &mut array_output,
            &ProgressCallback::new(&progress_callback),
        )?;
        let elements: ndarray::ArrayD<f32> = array_output.retrieve_array_subset(&array_subset)?;
        println!("{elements:?}");

        let elements_ref: ndarray::ArrayD<f32> = ndarray::array![
            [0.7262998, 1.4210157, 2.3036606, 2.9983768],
            [1.4210159, 2.1157317, 2.9983768, 3.6930926],
            [2.3036606, 2.9983766, 3.8810213, 4.575738],
            [2.9983768, 3.6930926, 4.5757375, 5.2704535]
        ]
        .into_dyn();
        approx::assert_abs_diff_eq!(elements, elements_ref);

        Ok(())
    }
}
