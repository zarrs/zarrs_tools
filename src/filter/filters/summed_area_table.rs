use std::ops::AddAssign;

use clap::Parser;
use itertools::Itertools;
use num_traits::{AsPrimitive, Zero};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use zarrs::{
    array::{
        data_type, Array, ArrayIndicesTinyVec, ArraySubset, DataTypeExt, Element, ElementOwned,
    },
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
    type_dispatch::retrieve_ndarray_as,
    UnsupportedDataTypeError,
};

#[derive(Debug, Clone, Parser, Serialize, Deserialize)]
pub struct SummedAreaTableArguments {}

impl FilterArguments for SummedAreaTableArguments {
    fn name(&self) -> String {
        "summed area table".to_string()
    }

    fn init(
        &self,
        common_args: &FilterCommonArguments,
    ) -> Result<Box<dyn FilterTraits>, FilterError> {
        Ok(Box::new(SummedAreaTable::new(*common_args.chunk_limit())))
    }
}

pub struct SummedAreaTable {
    chunk_limit: Option<usize>,
}

impl SummedAreaTable {
    pub fn new(chunk_limit: Option<usize>) -> Self {
        Self { chunk_limit }
    }

    /// Apply summed area table computation for a single dimension.
    /// TOut must match the output array's data type exactly since intermediate
    /// results are stored and read back from the output array.
    pub fn apply_dim<TOut>(
        &self,
        input: &Array<FilesystemStore>,
        output: &Array<FilesystemStore>,
        chunk_start_dim: &[u64],
        chunk_grid_shape: &[u64],
        dim: usize,
        progress: &Progress,
    ) -> Result<(), FilterError>
    where
        TOut: Element + ElementOwned + Send + Sync + Zero + AddAssign + Copy + 'static,
        // Bounds needed for retrieve_ndarray_as to convert any input type to TOut
        u8: AsPrimitive<TOut>,
        i8: AsPrimitive<TOut>,
        i16: AsPrimitive<TOut>,
        i32: AsPrimitive<TOut>,
        i64: AsPrimitive<TOut>,
        u16: AsPrimitive<TOut>,
        u32: AsPrimitive<TOut>,
        u64: AsPrimitive<TOut>,
        half::f16: AsPrimitive<TOut>,
        half::bf16: AsPrimitive<TOut>,
        f32: AsPrimitive<TOut>,
        f64: AsPrimitive<TOut>,
    {
        let dimensionality = chunk_start_dim.len();
        let chunk_shape =
            zarrs::array::chunk_shape_to_array_shape(&output.chunk_shape(chunk_start_dim)?);
        let mut last_shape = chunk_shape
            .iter()
            .map(|i| usize::try_from(*i).unwrap())
            .collect::<Vec<_>>();
        last_shape[dim] = 1;
        let mut sum_last = ndarray::ArrayD::<TOut>::zeros(last_shape);
        for i in 0..chunk_grid_shape[dim] {
            let chunk_indices = chunk_start_dim
                .iter()
                .enumerate()
                .map(|(dim_i, indices)| if dim_i == dim { i } else { *indices })
                .collect::<Vec<_>>();
            let mut chunk: ndarray::ArrayD<TOut> = if dim == dimensionality - 1 {
                // First dimension: read from input and convert to TOut
                let chunk_subset = output.chunk_subset(&chunk_indices)?;
                let (array, timing) = retrieve_ndarray_as::<TOut, _>(input, &chunk_subset)?;
                progress.add_retrieve_timing(timing);
                array
            } else {
                // Subsequent dimensions: read from output (already TOut)
                progress.read(|| output.retrieve_chunk(&chunk_indices))?
            };

            progress.process(|| {
                itertools::izip!(
                    sum_last.lanes_mut(ndarray::Axis(dim)),
                    chunk.lanes_mut(ndarray::Axis(dim))
                )
                // .par_bridge() // Faster off?
                .for_each(|(mut sum_last, mut lane)| {
                    sum_last[0] = lane.iter_mut().fold(sum_last[0], |acc, element| {
                        *element += acc;
                        *element
                    });
                });
            });

            progress.write(|| output.store_chunk(&chunk_indices, chunk))?;
            progress.next();
        }

        Ok(())
    }
}

impl FilterTraits for SummedAreaTable {
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

        let progress = {
            Progress::new(
                usize::try_from(output.chunk_grid_shape().iter().product::<u64>()).unwrap()
                    * output.dimensionality(),
                progress_callback,
            )
        };

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

        let dimensionality = output.chunk_grid().dimensionality();
        let chunk_grid_shape = output.chunk_grid_shape();
        for dim in (0..dimensionality).rev() {
            let chunk_grid_shape_dim = chunk_grid_shape
                .iter()
                .enumerate()
                .map(|(i, dim_i)| if i == dim { 1 } else { *dim_i })
                .collect_vec();
            let chunks_dim = ArraySubset::new_with_shape(chunk_grid_shape_dim);
            let indices = chunks_dim.indices();
            rayon_iter_concurrent_limit::iter_concurrent_limit!(
                chunk_limit,
                indices,
                try_for_each,
                |chunk_start_dim: ArrayIndicesTinyVec| {
                    macro_rules! apply {
                        ([$( ( $dt_out:ty, $type_out:ty ) ),* ]) => {
                            match output.data_type().identifier() {
                                $(<$dt_out>::IDENTIFIER => {
                                    self.apply_dim::<$type_out>(
                                        input,
                                        output,
                                        &chunk_start_dim,
                                        &chunk_grid_shape,
                                        dim,
                                        &progress,
                                    )?;
                                },)*
                                id => panic!("Unsupported output data type: {}", id)
                            }
                        };
                    }
                    apply!([
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
                    Ok::<_, FilterError>(())
                }
            )?;
        }

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
    fn summed_area_table() -> Result<(), Box<dyn Error>> {
        let path = tempfile::TempDir::new()?;
        let store = FilesystemStore::new(path.path())?;
        let array = ArrayBuilder::new(vec![6, 6], vec![2, 2], data_type::uint8(), 0u8)
            .build(store.into(), "/")?;
        let array_subset = array.subset_all();
        let elements_in: ndarray::ArrayD<u8> = ndarray::array![
            [31, 2, 4, 33, 5, 36],
            [12, 26, 9, 10, 29, 25],
            [13, 17, 21, 22, 20, 18],
            [24, 23, 15, 16, 14, 19],
            [30, 8, 28, 27, 11, 7],
            [1, 35, 34, 3, 32, 6],
        ]
        .into_dyn();
        array.store_array_subset(&array_subset, elements_in)?;

        let elements: ndarray::ArrayD<u8> = array.retrieve_array_subset(&array_subset)?;
        println!("{elements:?}");

        let path = tempfile::TempDir::new()?;
        let store: FilesystemStore = FilesystemStore::new(path.path())?;
        let mut array_output = array
            .builder()
            .data_type(data_type::uint16())
            .fill_value(0u16)
            .build(store.into(), "/")?;
        let progress_callback = |_stats: ProgressStats| {};
        SummedAreaTable::new(None).apply(
            &array,
            &mut array_output,
            &ProgressCallback::new(&progress_callback),
        )?;
        let elements: ndarray::ArrayD<u16> = array_output.retrieve_array_subset(&array_subset)?;
        println!("{elements:?}");

        let elements_ref: ndarray::ArrayD<u16> = ndarray::array![
            [31, 33, 37, 70, 75, 111],
            [43, 71, 84, 127, 161, 222],
            [56, 101, 135, 200, 254, 333],
            [80, 148, 197, 278, 346, 444],
            [110, 186, 263, 371, 450, 555],
            [111, 222, 333, 444, 555, 666],
        ]
        .into_dyn();
        approx::assert_abs_diff_eq!(elements, elements_ref);

        Ok(())
    }
}

/// Computes the summed area table on a single ndarray. Not suitable for computing on an entire zarr array.
// FIXME: Generic
pub fn summed_area_table_inplace(mut array: ndarray::ArrayD<f64>) -> ndarray::ArrayD<f64> {
    for dim in (0..array.ndim()).rev() {
        ndarray::Zip::from(array.lanes_mut(ndarray::Axis(dim)))
            .into_par_iter()
            .for_each(|(mut lane,)| {
                lane.iter_mut().fold(0.0, |acc, element| {
                    *element += acc;
                    *element
                });
            });
    }
    array
}

/// Computes the summed area table on a single ndarray. Not suitable for computing on an entire zarr array.
// FIXME: Generic
pub fn summed_area_table(array: &ndarray::ArrayD<f32>, sat: &mut ndarray::ArrayD<f64>) {
    std::iter::zip(
        array.lanes(ndarray::Axis(array.ndim() - 1)),
        sat.lanes_mut(ndarray::Axis(array.ndim() - 1)),
    )
    .for_each(|(input, mut sat)| {
        std::iter::zip(input.iter(), sat.iter_mut()).fold(0.0, |acc, (input, sat)| {
            *sat = *input as f64 + acc;
            *sat
        });
    });
    for dim in (0..array.ndim() - 1).rev() {
        sat.lanes_mut(ndarray::Axis(dim))
            .into_iter()
            .for_each(|mut sat| {
                sat.iter_mut().fold(0.0, |acc, sat| {
                    *sat += acc;
                    *sat
                });
            });
    }
}

/// Compute the sum of the elements between p0 and p1 inclusive from a summed area table.
///
/// Panics if p0/p1 are out-of-bounds.
pub fn summed_area_table_sum(
    summed_area_table: &ndarray::ArrayD<f64>,
    p0: &[usize],
    p1: &[usize],
) -> f32 {
    let d = summed_area_table.ndim();
    assert_eq!(d, p0.len());
    assert_eq!(d, p1.len());

    let mut sum = 0.0;
    let mut x_p: Vec<usize> = Vec::with_capacity(d);
    'outer: for i in 0..2usize.pow(d.try_into().unwrap()) {
        x_p.clear();
        let mut p_sum: usize = 0;
        for (j, (p0, p1)) in std::iter::zip(p0, p1).enumerate() {
            assert!(p1 >= p0);
            let p = (i >> (d - 1 - j)) % 2;
            if p == 0 {
                if p0 == &0 {
                    continue 'outer;
                }
                x_p.push(*p0 - 1)
            } else {
                // x_p.push(std::cmp::min(*p1, shape - 1))
                x_p.push(*p1)
            };
            p_sum += p;
        }
        let sign: i8 = 1 - 2 * ((d - p_sum) % 2) as i8;
        let value = summed_area_table.get(x_p.as_slice()).unwrap();
        // println!("{i}\tp0={p0:?} p1={p1:?}\tx_p={x_p:?} I(x_p)={value} sign={sign}");
        sum += sign as f64 * value;
    }
    sum as f32
}

pub fn summed_area_table_mean(
    summed_area_table: &ndarray::ArrayD<f64>,
    p0: &[usize],
    p1: &[usize],
) -> f32 {
    let sum = summed_area_table_sum(summed_area_table, p0, p1);
    let n_elements = std::iter::zip(p0, p1)
        .map(|(p0, p1)| p1 - p0 + 1)
        .product::<usize>();
    sum / n_elements as f32
}
