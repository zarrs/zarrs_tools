use clap::Parser;
use itertools::Itertools;
use ndarray::ArrayD;
use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{Array, ArrayIndicesTinyVec, DataType, Element, ElementOwned},
    array_subset::ArraySubset,
    filesystem::FilesystemStore,
};

use crate::{
    filter::{
        calculate_chunk_limit,
        filter_error::FilterError,
        filter_traits::{ChunkInfo, FilterTraits},
        kernel::apply_1d_kernel,
        ArraySubsetOverlap, FilterArguments, FilterCommonArguments, UnsupportedDataTypeError,
    },
    progress::{Progress, ProgressCallback},
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
    kernel: Vec<ndarray::Array1<f32>>,
    kernel_half_size: Vec<u64>,
    chunk_limit: Option<usize>,
}

impl Gaussian {
    pub fn new(sigma: Vec<f32>, kernel_half_size: Vec<u64>, chunk_limit: Option<usize>) -> Self {
        let kernel = std::iter::zip(&sigma, &kernel_half_size)
            .map(|(sigma, kernel_half_size)| {
                create_sampled_gaussian_kernel(*sigma, *kernel_half_size)
            })
            .collect_vec();
        Self {
            kernel,
            kernel_half_size,
            chunk_limit,
        }
    }

    pub fn kernel_half_size(&self) -> &[u64] {
        &self.kernel_half_size
    }

    pub fn apply_chunk<TIn, TOut>(
        &self,
        input: &Array<FilesystemStore>,
        output: &Array<FilesystemStore>,
        chunk_indices: &[u64],
        progress: &Progress,
    ) -> Result<(), FilterError>
    where
        TIn: ElementOwned + Send + Sync + AsPrimitive<f32>,
        TOut: Element + Send + Sync + Copy + 'static,
        f32: AsPrimitive<TOut>,
    {
        let subset_output = output.chunk_subset_bounded(chunk_indices).unwrap();
        let subset_overlap =
            ArraySubsetOverlap::new(input.shape(), &subset_output, &self.kernel_half_size);

        let input_array: ndarray::ArrayD<TIn> =
            progress.read(|| input.retrieve_array_subset(subset_overlap.subset_input()))?;

        let output_array = progress.process(|| {
            let input_array = input_array.mapv(|x| x.as_()); // par?
            let output_array = self.apply_ndarray(input_array);
            let output_array = subset_overlap.extract_subset(&output_array);
            Ok::<_, FilterError>(output_array.mapv(|x| x.as_())) // par?
        })?;
        drop(input_array);

        progress.write(|| {
            output
                .store_array_subset(&subset_output, output_array)
                .unwrap()
        });

        progress.next();
        Ok(())
    }

    pub fn apply_ndarray(&self, mut input: ndarray::ArrayD<f32>) -> ndarray::ArrayD<f32> {
        let mut gaussian = ArrayD::<f32>::zeros(input.shape());
        for dim in 0..input.ndim() {
            apply_1d_kernel(dim, &self.kernel[dim], &input, &mut gaussian);
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
        for data_type in [chunk_input.1, chunk_output.1] {
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
                macro_rules! apply_output {
                    ( $type_in:ty, [$( ( $data_type_out:ident, $type_out:ty ) ),* ]) => {
                        match output.data_type() {
                            $(DataType::$data_type_out => { self.apply_chunk::<$type_in, $type_out>(&input, &output, &chunk_indices, &progress) } ,)*
                            _ => panic!()
                        }
                    };
                }
                macro_rules! apply_input {
                ([$( ( $data_type_in:ident, $type_in:ty ) ),* ]) => {
                        match input.data_type() {
                            $(
                                DataType::$data_type_in => {
                                    apply_output!($type_in, [
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
                apply_input!([
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
                ])
            }
        )?;

        Ok(())
    }
}

fn create_sampled_gaussian_kernel(sigma: f32, kernel_half_size: u64) -> ndarray::Array1<f32> {
    if sigma == 0.0 {
        ndarray::Array1::<f32>::from_vec(vec![1.0])
    } else {
        let t = sigma * sigma;
        let scale = 1.0 / (2.0 * std::f32::consts::PI * t).sqrt();
        let kernel_half_elements =
            (0..=kernel_half_size).map(|n| scale * (-((n * n) as f32 / (2.0 * t))).exp());
        let kernel_elements = kernel_half_elements
            .clone()
            .rev()
            .chain(kernel_half_elements.skip(1))
            .collect::<Vec<_>>();
        ndarray::Array1::<f32>::from_vec(kernel_elements)
    }
}

#[cfg(test)]
mod tests {
    use crate::progress::ProgressStats;

    use super::*;
    use std::error::Error;
    use zarrs::array::ArrayBuilder;

    #[test]
    fn gaussian() -> Result<(), Box<dyn Error>> {
        let path = tempfile::TempDir::new()?;
        let store = FilesystemStore::new(path.path())?;
        let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], DataType::Float32, 0.0f32)
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
