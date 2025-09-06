use clap::Parser;
use num_traits::AsPrimitive;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{Array, DataType, Element, ElementOwned},
    array_subset::ArraySubset,
    filesystem::FilesystemStore,
};

use crate::filter::{
    filters::summed_area_table::{summed_area_table, summed_area_table_mean},
    UnsupportedDataTypeError,
};
use crate::{
    filter::{
        calculate_chunk_limit, filter_error::FilterError, filter_traits::FilterTraits,
        ArraySubsetOverlap, FilterArguments, FilterCommonArguments,
    },
    progress::{Progress, ProgressCallback},
};

#[derive(Debug, Clone, Parser, Serialize, Deserialize)]
pub struct GuidedFilterArguments {
    /// Guided filter "epsilon".
    #[arg(required = true)]
    epsilon: f32,
    /// Guided filter "radius".
    #[arg(required = true)]
    radius: u8,
}

impl FilterArguments for GuidedFilterArguments {
    fn name(&self) -> String {
        "guided_filter".to_string()
    }

    fn init(
        &self,
        common_args: &FilterCommonArguments,
    ) -> Result<Box<dyn FilterTraits>, FilterError> {
        Ok(Box::new(GuidedFilter::new(
            self.epsilon,
            self.radius,
            *common_args.chunk_limit(),
        )))
    }
}

pub struct GuidedFilter {
    epsilon: f32,
    radius: u8,
    chunk_limit: Option<usize>,
}

impl GuidedFilter {
    pub fn new(epsilon: f32, radius: u8, chunk_limit: Option<usize>) -> Self {
        Self {
            epsilon,
            radius,
            chunk_limit,
        }
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    pub fn radius(&self) -> u8 {
        self.radius
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
        let subset_overlap = ArraySubsetOverlap::new(
            input.shape(),
            &subset_output,
            // double radius is needed for correct guided filter because kernel of radius is applied twice
            &vec![(self.radius * 2) as u64; input.dimensionality()],
        );

        let input_array = progress
            .read(|| input.retrieve_array_subset_ndarray::<TIn>(subset_overlap.subset_input()))?;

        let output_array = progress.process(|| {
            let input_array = input_array.mapv(|x| x.as_()); // par?
            let output_array = self.apply_ndarray(input_array);
            let output_array = subset_overlap.extract_subset(&output_array);
            Ok::<_, FilterError>(output_array.mapv(|x| x.as_())) // par?
        })?;
        drop(input_array);

        progress.write(|| {
            output
                .store_array_subset_ndarray::<TOut, _>(subset_output.start(), output_array)
                .unwrap()
        });

        progress.next();
        Ok(())
    }

    // FIXME: Generic
    pub fn apply_ndarray(&self, v_i: ndarray::ArrayD<f32>) -> ndarray::ArrayD<f32> {
        let subset = zarrs::array_subset::ArraySubset::new_with_shape(
            v_i.shape().iter().map(|i| *i as u64).collect(),
        );

        // Alloc: f64
        let mut sat = ndarray::ArrayD::<f64>::zeros(v_i.shape());

        // Alloc: f32
        summed_area_table(&v_i, &mut sat);
        let mut u_k = self.sat_to_mean(&sat);

        // Alloc: f32
        let mut vi_minus_uk_2 = ndarray::Zip::from(&v_i)
            .and(&u_k)
            .par_map_collect(|v, u| (v - u).powf(2.0));
        summed_area_table(&vi_minus_uk_2, &mut sat);

        ndarray::par_azip!((u in &mut u_k, sigma2 in &mut vi_minus_uk_2) {
            let a = *sigma2 / (*sigma2 + self.epsilon);
            let b = (1.0 - a) * *u;
            *u = a;
            *sigma2 = b;
        });
        let a_k = u_k;
        let b_k = vi_minus_uk_2;

        summed_area_table(&a_k, &mut sat);
        drop(a_k);
        #[allow(deprecated)]
        let mut v_i = v_i.into_raw_vec();
        v_i.par_iter_mut()
            .zip(&subset.indices())
            .for_each(|(v_i, indices)| {
                let (p0, p1) = self.get_block(&indices, sat.shape());
                *v_i *= summed_area_table_mean(&sat, &p0, &p1);
            });

        summed_area_table(&b_k, &mut sat);
        drop(b_k);
        v_i.par_iter_mut()
            .zip(&subset.indices())
            .for_each(|(v_i, indices)| {
                let (p0, p1) = self.get_block(&indices, sat.shape());
                *v_i += summed_area_table_mean(&sat, &p0, &p1);
            });
        ndarray::ArrayD::from_shape_vec(sat.shape(), v_i).unwrap()
    }

    fn get_block(&self, indices: &[u64], shape: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let p0: Vec<usize> = std::iter::zip(indices, shape)
            .map(|(indices, shape)| {
                std::cmp::min(
                    usize::try_from(indices.saturating_sub(self.radius as u64)).unwrap(),
                    shape - 1,
                )
            })
            .collect();
        let p1: Vec<usize> = std::iter::zip(indices, shape)
            .map(|(indices, shape)| {
                std::cmp::min(
                    usize::try_from(indices + self.radius as u64).unwrap(),
                    shape - 1,
                )
            })
            .collect();
        (p0, p1)
    }

    fn sat_to_mean(&self, sat: &ndarray::ArrayD<f64>) -> ndarray::ArrayD<f32> {
        let subset = zarrs::array_subset::ArraySubset::new_with_shape(
            sat.shape().iter().map(|i| *i as u64).collect(),
        );
        let mean: Vec<f32> = subset
            .indices()
            .into_par_iter()
            .map(|indices| {
                let (p0, p1) = self.get_block(&indices, sat.shape());
                summed_area_table_mean(sat, &p0, &p1)
            })
            .collect();
        ndarray::ArrayD::from_shape_vec(sat.shape(), mean).unwrap()
    }
}

impl FilterTraits for GuidedFilter {
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
        chunk_input: &zarrs::array::ChunkRepresentation,
        chunk_output: &zarrs::array::ChunkRepresentation,
    ) -> usize {
        chunk_input.fixed_element_size().unwrap()
            + chunk_output.fixed_element_size().unwrap()
            + chunk_output.num_elements_usize()
                * (core::mem::size_of::<f64>() + core::mem::size_of::<f32>() * 2)
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

#[cfg(test)]
mod tests {
    use crate::progress::ProgressStats;

    use super::*;
    use std::error::Error;
    use zarrs::array::ArrayBuilder;

    #[test]
    fn guided_filter() -> Result<(), Box<dyn Error>> {
        let path = tempfile::TempDir::new()?;
        let store = FilesystemStore::new(path.path())?;
        let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], DataType::Float32, 0.0f32)
            .build(store.into(), "/")?;
        let array_subset = array.subset_all();
        array.store_array_subset_elements(
            &array_subset,
            &(0..array_subset.num_elements_usize())
                .map(|u| ((u / array.shape()[1] as usize) + u % array.shape()[1] as usize) as f32)
                .collect::<Vec<f32>>(),
        )?;

        let elements = array.retrieve_array_subset_ndarray::<f32>(&array_subset)?;
        println!("{elements:?}");

        let path = tempfile::TempDir::new()?;
        let store: FilesystemStore = FilesystemStore::new(path.path())?;
        let mut array_output = array.builder().build(store.into(), "/")?;
        let progress_callback = |_stats: ProgressStats| {};
        GuidedFilter::new(1.0, 2, None).apply(
            &array,
            &mut array_output,
            &ProgressCallback::new(&progress_callback),
        )?;
        let elements = array_output.retrieve_array_subset_ndarray::<f32>(&array_subset)?;
        println!("{elements:?}");

        let elements_ref: ndarray::ArrayD<f32> = ndarray::array![
            [1.659829, 2.1910257, 2.5641026, 3.0],
            [2.1910257, 2.614423, 3.0, 3.4358974],
            [2.5641026, 3.0, 3.385577, 3.8089743],
            [3.0, 3.4358974, 3.8089743, 4.340171]
        ]
        .into_dyn();
        approx::assert_abs_diff_eq!(elements, elements_ref);

        Ok(())
    }
}
