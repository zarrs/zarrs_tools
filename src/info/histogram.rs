use half::{bf16, f16};
use num_traits::AsPrimitive;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use zarrs::{
    array::{Array, ArrayError, DataType, ElementOwned},
    array_subset::ArraySubset,
    storage::ReadableStorageTraits,
};

pub fn calculate_histogram<TStorage: ReadableStorageTraits + 'static>(
    array: &Array<TStorage>,
    n_bins: usize,
    min: f64,
    max: f64,
    chunk_limit: usize,
) -> Result<(Vec<f64>, Vec<u64>), ArrayError> {
    match array.data_type() {
        DataType::Int8 => calculate_histogram_t::<_, i8>(array, n_bins, min, max, chunk_limit),
        DataType::Int16 => calculate_histogram_t::<_, i16>(array, n_bins, min, max, chunk_limit),
        DataType::Int32 => calculate_histogram_t::<_, i32>(array, n_bins, min, max, chunk_limit),
        DataType::Int64 => calculate_histogram_t::<_, i64>(array, n_bins, min, max, chunk_limit),
        DataType::UInt8 => calculate_histogram_t::<_, u8>(array, n_bins, min, max, chunk_limit),
        DataType::UInt16 => calculate_histogram_t::<_, u16>(array, n_bins, min, max, chunk_limit),
        DataType::UInt32 => calculate_histogram_t::<_, u32>(array, n_bins, min, max, chunk_limit),
        DataType::UInt64 => calculate_histogram_t::<_, u64>(array, n_bins, min, max, chunk_limit),
        DataType::Float16 => calculate_histogram_t::<_, f16>(array, n_bins, min, max, chunk_limit),
        DataType::BFloat16 => {
            calculate_histogram_t::<_, bf16>(array, n_bins, min, max, chunk_limit)
        }
        DataType::Float32 => calculate_histogram_t::<_, f32>(array, n_bins, min, max, chunk_limit),
        DataType::Float64 => calculate_histogram_t::<_, f64>(array, n_bins, min, max, chunk_limit),
        DataType::Bool | DataType::Complex64 | DataType::Complex128 | DataType::RawBits(_) => {
            unimplemented!("Data type not supported")
        }
        _ => unimplemented!("Data type not supported"),
    }
}

pub fn calculate_histogram_t<
    TStorage: ReadableStorageTraits + 'static,
    T: ElementOwned + PartialOrd + Send + Sync + AsPrimitive<f64>,
>(
    array: &Array<TStorage>,
    n_bins: usize,
    min: f64,
    max: f64,
    chunk_limit: usize,
) -> Result<(Vec<f64>, Vec<u64>), ArrayError> {
    let chunks = ArraySubset::new_with_shape(array.chunk_grid_shape().clone());

    let chunk_incr_histogram = |histogram: Result<Vec<u64>, ArrayError>,
                                chunk_indices: Vec<u64>| {
        let mut histogram = histogram?;
        let elements = array.retrieve_chunk_elements::<T>(&chunk_indices)?;
        for element in elements {
            let norm: f64 = (element.as_() - min) / (max - min);
            let bin = ((norm * n_bins as f64).max(0.0).floor() as usize).min(n_bins - 1);
            histogram[bin] += 1;
        }
        Ok(histogram)
    };

    let bin_edges = (0..=n_bins)
        .map(|bin| {
            let binf = bin as f64 / n_bins as f64;
            binf * (max - min) + min
        })
        .collect();

    let indices = chunks.indices();
    let hist = indices
        .into_par_iter()
        .fold_chunks(
            indices.len().div_ceil(chunk_limit).max(1),
            || Ok(vec![0; n_bins]),
            chunk_incr_histogram,
        )
        .try_reduce_with(|histogram_a, histogram_b| {
            Ok(histogram_a
                .into_iter()
                .zip(histogram_b)
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>())
        })
        .expect("a value since the chunk is not empty")?;

    Ok((bin_edges, hist))
}
