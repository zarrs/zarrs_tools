use clap::{Parser, ValueEnum};
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
    filter::{calculate_chunk_limit, ArraySubsetOverlap, UnsupportedDataTypeError},
    progress::{Progress, ProgressCallback},
};

use crate::filter::{
    filter_error::FilterError,
    filter_traits::{ChunkInfo, FilterTraits},
    kernel::{apply_1d_difference_operator, apply_1d_triangle_filter},
    FilterArguments, FilterCommonArguments,
};

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize, Default)]
pub enum GradientMagnitudeOperator {
    #[default]
    Sobel,
    CentralDifference,
}

#[derive(Debug, Clone, Parser, Serialize, Deserialize, Default)]
pub struct GradientMagnitudeArguments {
    /// Gradient magnitude operator (kernel).
    #[arg(long)]
    #[clap(value_enum, default_value_t=GradientMagnitudeOperator::Sobel)]
    pub operator: GradientMagnitudeOperator,
}

impl FilterArguments for GradientMagnitudeArguments {
    fn name(&self) -> String {
        "gradient_magnitude".to_string()
    }

    fn init(
        &self,
        common_args: &FilterCommonArguments,
    ) -> Result<Box<dyn FilterTraits>, FilterError> {
        Ok(Box::new(GradientMagnitude::new(
            self,
            *common_args.chunk_limit(),
        )))
    }
}

pub struct GradientMagnitude {
    operator: GradientMagnitudeOperator,
    chunk_limit: Option<usize>,
}

impl GradientMagnitude {
    pub fn new(arguments: &GradientMagnitudeArguments, chunk_limit: Option<usize>) -> Self {
        Self {
            operator: arguments.operator,
            chunk_limit,
        }
    }

    pub fn apply_chunk<TIn, TOut>(
        &self,
        input: &Array<FilesystemStore>,
        output: &Array<FilesystemStore>,
        chunk_indices: &[u64],
        progress: &Progress,
    ) -> Result<(), FilterError>
    where
        TIn: ElementOwned + AsPrimitive<f32>,
        TOut: Element + Copy + 'static,
        f32: AsPrimitive<TOut>,
    {
        // Determine the input and output subset
        let subset_output = output.chunk_subset_bounded(chunk_indices).unwrap();
        let subset_overlap = ArraySubsetOverlap::new(
            input.shape(),
            &subset_output,
            &vec![1; input.dimensionality()],
        );

        let input_array: ndarray::ArrayD<TIn> =
            progress.read(|| input.retrieve_array_subset(subset_overlap.subset_input()))?;

        let gradient_magnitude = progress.process(|| {
            let input_array_f32 = input_array.map(|x| x.as_());
            let gradient_magnitude = self.apply_ndarray(&input_array_f32);
            let gradient_magnitude = subset_overlap.extract_subset(&gradient_magnitude);
            gradient_magnitude.map(|x| x.as_())
        });
        drop(input_array);

        progress.write(|| {
            output
                .store_array_subset(&subset_output, gradient_magnitude)
                .unwrap()
        });

        progress.next();
        Ok(())
    }

    pub fn apply_ndarray(&self, input: &ndarray::ArrayD<f32>) -> ndarray::ArrayD<f32> {
        let mut staging_out = ArrayD::<f32>::zeros(input.shape());
        let mut gradient_magnitude = ArrayD::<f32>::zeros(input.shape());

        match self.operator {
            GradientMagnitudeOperator::Sobel => {
                let mut staging_in = ArrayD::<f32>::zeros(input.shape());
                for axis in 0..input.ndim() {
                    staging_in.assign(input);
                    for i in 0..input.ndim() {
                        if i == axis {
                            apply_1d_difference_operator(i, &staging_in, &mut staging_out);
                        } else {
                            apply_1d_triangle_filter(i, &staging_in, &mut staging_out);
                        }
                        if i != input.ndim() - 1 {
                            std::mem::swap(&mut staging_in, &mut staging_out);
                        }
                    }

                    ndarray::Zip::from(&mut gradient_magnitude)
                        .and(&staging_out)
                        .par_for_each(|g, &s| *g += s * s);
                }
            }
            GradientMagnitudeOperator::CentralDifference => {
                for axis in 0..input.ndim() {
                    apply_1d_difference_operator(axis, input, &mut staging_out);

                    ndarray::Zip::from(&mut gradient_magnitude)
                        .and(&staging_out)
                        .par_for_each(|g, &s| *g += s * s);
                }
            }
        }
        gradient_magnitude.map_inplace(|x| *x = x.sqrt());

        gradient_magnitude
    }
}

impl FilterTraits for GradientMagnitude {
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
        let num_input_elements =
            usize::try_from(chunk_input.0.iter().map(|s| s.get() + 2).product::<u64>()).unwrap();
        let num_output_elements =
            usize::try_from(chunk_input.0.iter().map(|s| s.get()).product::<u64>()).unwrap();
        num_input_elements * (chunk_input.1.fixed_size().unwrap() + core::mem::size_of::<f32>() * 4)
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
                            $(DataType::$data_type_out => { self.apply_chunk::<$type_in, $type_out>(input, output, &chunk_indices, &progress) } ,)*
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
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::progress::ProgressStats;

    use super::*;
    use std::error::Error;
    use zarrs::array::ArrayBuilder;

    #[test]
    fn gradients() -> Result<(), Box<dyn Error>> {
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
        GradientMagnitude::new(&GradientMagnitudeArguments::default(), None).apply(
            &array,
            &mut array_output,
            &ProgressCallback::new(&progress_callback),
        )?;
        let elements: ndarray::ArrayD<f32> = array_output.retrieve_array_subset(&array_subset)?;
        println!("{elements:?}");

        let elements_ref: ndarray::ArrayD<f32> = ndarray::array![
            [0.70710677, 1.118034, 1.118034, 0.70710677],
            [1.118034, 1.4142135, 1.4142135, 1.118034],
            [1.118034, 1.4142135, 1.4142135, 1.118034],
            [0.70710677, 1.118034, 1.118034, 0.70710677]
        ]
        .into_dyn();

        approx::assert_abs_diff_eq!(elements, elements_ref);

        Ok(())
    }
}
