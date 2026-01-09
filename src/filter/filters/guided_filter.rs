use clap::Parser;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
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
        filters::summed_area_table::{summed_area_table, summed_area_table_mean},
        ArraySubsetOverlap, FilterArguments, FilterCommonArguments,
    },
    progress::{Progress, ProgressCallback},
    type_dispatch::{retrieve_ndarray_as, store_ndarray_from},
    UnsupportedDataTypeError,
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

    pub fn apply_chunk(
        &self,
        input: &Array<FilesystemStore>,
        output: &Array<FilesystemStore>,
        chunk_indices: &[u64],
        progress: &Progress,
    ) -> Result<(), FilterError> {
        let subset_output = output.chunk_subset_bounded(chunk_indices).unwrap();
        let subset_overlap = ArraySubsetOverlap::new(
            input.shape(),
            &subset_output,
            // double radius is needed for correct guided filter because kernel of radius is applied twice
            &vec![(self.radius * 2) as u64; input.dimensionality()],
        );

        // Read input and convert to f32
        let (input_array, retrieve_timing) =
            retrieve_ndarray_as::<f32, _>(input, subset_overlap.subset_input())?;
        progress.add_retrieve_timing(retrieve_timing);

        // Process in f32
        let output_array = progress.process(|| {
            let processed = self.apply_ndarray(input_array);
            subset_overlap.extract_subset(&processed)
        });

        // Convert from f32 and store
        let store_timing = store_ndarray_from::<f32, _>(output, &subset_output, output_array)?;
        progress.add_store_timing(store_timing);

        progress.next();
        Ok(())
    }

    pub fn apply_ndarray(&self, v_i: ndarray::ArrayD<f32>) -> ndarray::ArrayD<f32> {
        let subset = ArraySubset::new_with_shape(v_i.shape().iter().map(|i| *i as u64).collect());

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
        let subset = ArraySubset::new_with_shape(sat.shape().iter().map(|i| *i as u64).collect());
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
        let num_output_elements =
            usize::try_from(chunk_output.0.iter().map(|s| s.get()).product::<u64>()).unwrap();
        chunk_input.1.fixed_size().unwrap()
            + chunk_output.1.fixed_size().unwrap()
            + num_output_elements * (core::mem::size_of::<f64>() + core::mem::size_of::<f32>() * 2)
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

#[cfg(test)]
mod tests {
    use crate::progress::ProgressStats;

    use super::*;
    use std::error::Error;
    use zarrs::array::{data_type, ArrayBuilder};

    #[test]
    fn guided_filter() -> Result<(), Box<dyn Error>> {
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
        GuidedFilter::new(1.0, 2, None).apply(
            &array,
            &mut array_output,
            &ProgressCallback::new(&progress_callback),
        )?;
        let elements: ndarray::ArrayD<f32> = array_output.retrieve_array_subset(&array_subset)?;
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
