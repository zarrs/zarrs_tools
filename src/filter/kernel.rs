use num_traits::Float;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::iter::Sum;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::{ravel_indices, unravel_index};

pub fn get_axis_start_index(axis: usize, index: usize, shape: &[usize]) -> Option<usize> {
    let shape1 = shape
        .iter()
        .enumerate()
        .map(|(i, &s)| if i == axis { 1 } else { s as u64 })
        .collect::<Vec<_>>();
    let shape3 = shape.iter().map(|&s| s as u64).collect::<Vec<_>>();
    let idx3 = unravel_index(index as u64, &shape1)?;
    Some(ravel_indices(&idx3, &shape3)? as usize)
}

/// Apply an even 1D kernel.
pub fn apply_1d_kernel<T>(
    dim: usize,
    kernel: &ndarray::Array1<T>,
    input: &ndarray::ArrayD<T>,
    output: &mut ndarray::ArrayD<T>,
) where
    T: Float + Send + Sync + Sum + Copy,
{
    assert!(kernel.len() % 2 == 1);
    let shape = input.shape();
    let input_slice = unsafe { std::slice::from_raw_parts(input.as_ptr(), input.len()) };
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr(), output.len()) };
    let output_slice = UnsafeCellSlice::new(output_slice);
    let stride = input.strides()[dim] as usize;
    let kernel_mid = kernel.len() / 2;
    let axis_len = shape[dim];
    let output_indices = 0..input.len() / axis_len;
    if stride == 1 {
        let axis_end_inc = axis_len - 1;
        output_indices.into_par_iter().for_each(|j| {
            let axis_start_index = get_axis_start_index(dim, j, shape).expect("inbounds");
            (0..axis_len).for_each(|k| {
                let sum = kernel
                    .iter()
                    .zip(k..k + kernel.len())
                    .map(|(kernel_i, i)| {
                        let element_i = axis_start_index
                            + std::cmp::min(i.saturating_sub(kernel_mid), axis_end_inc);
                        let value = unsafe { *input_slice.get_unchecked(element_i) } * *kernel_i;
                        value
                    })
                    .sum::<T>();
                let element: usize = axis_start_index + k;
                let output_element = unsafe { output_slice.index_mut(element) };
                *output_element = sum;
            })
        });
    } else {
        output_indices.into_par_iter().for_each(|j| {
            let axis_start_index = get_axis_start_index(dim, j, shape).expect("inbounds");
            let axis_end_inc = (axis_len - 1) * stride;
            (0..axis_len).for_each(|k| {
                let sum = kernel
                    .iter()
                    .zip(k..k + kernel.len())
                    .map(|(kernel_i, i)| {
                        let element_i = axis_start_index
                            + std::cmp::min(i.saturating_sub(kernel_mid) * stride, axis_end_inc);
                        let value = unsafe { *input_slice.get_unchecked(element_i) } * *kernel_i;
                        value
                    })
                    .sum::<T>();
                let element: usize = axis_start_index + k * stride;
                let output_element = unsafe { output_slice.index_mut(element) };
                *output_element = sum;
            })
        });
    }
}

// Apply triangle filter [1, 2, 1]
pub fn apply_1d_triangle_filter<T>(
    axis: usize,
    input: &ndarray::ArrayD<T>,
    output: &mut ndarray::ArrayD<T>,
) where
    T: Float + Send + Sync + Copy,
{
    let shape = input.shape();
    let input_slice = unsafe { std::slice::from_raw_parts(input.as_ptr(), input.len()) };
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr(), output.len()) };
    let output_slice = UnsafeCellSlice::new(output_slice);
    let stride = input.strides()[axis] as usize;
    let axis_len = shape[axis];
    let quarter: T = T::from(0.25).unwrap();
    let half: T = T::from(0.5).unwrap();
    (0..input.len() / axis_len).into_par_iter().for_each(|j| {
        let axis_start = get_axis_start_index(axis, j, shape).expect("inbounds");
        (0..axis_len).for_each(|k| {
            let prev = axis_start + k.saturating_sub(1) * stride;
            let element = axis_start + k * stride;
            let next = axis_start + std::cmp::min(k + 1, axis_len - 1) * stride;
            *unsafe { output_slice.index_mut(element) } =
                *unsafe { input_slice.get_unchecked(prev) } * quarter
                    + *unsafe { input_slice.get_unchecked(element) } * half
                    + *unsafe { input_slice.get_unchecked(next) } * quarter;
        })
    });
}

// Apply difference operator [-1, 0, 1]
pub fn apply_1d_difference_operator<T>(
    axis: usize,
    input: &ndarray::ArrayD<T>,
    output: &mut ndarray::ArrayD<T>,
) where
    T: Float + Send + Sync + Copy,
{
    let shape = input.shape();
    let input_slice = unsafe { std::slice::from_raw_parts(input.as_ptr(), input.len()) };
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr(), output.len()) };
    let output_slice = UnsafeCellSlice::new(output_slice);
    let stride = input.strides()[axis] as usize;
    let axis_len = shape[axis];
    let half: T = T::from(0.5).unwrap();
    (0..input.len() / axis_len).into_par_iter().for_each(|j| {
        let axis_start = get_axis_start_index(axis, j, shape).expect("inbounds");
        (0..axis_len).for_each(|k| {
            let prev = axis_start + k.saturating_sub(1) * stride;
            let element = axis_start + k * stride;
            let next = axis_start + std::cmp::min(k + 1, axis_len - 1) * stride;
            let difference = half
                * (*unsafe { input_slice.get_unchecked(next) }
                    - *unsafe { input_slice.get_unchecked(prev) });
            *unsafe { output_slice.index_mut(element) } = difference;
        })
    });
}
