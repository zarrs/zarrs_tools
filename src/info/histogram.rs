use half::{bf16, f16};
use num_traits::AsPrimitive;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use zarrs::{
    array::{
        data_type, Array, ArrayError, ArrayIndicesTinyVec, ArraySubset, DataTypeExt, ElementOwned,
    },
    storage::ReadableStorageTraits,
};

pub fn calculate_histogram<TStorage: ReadableStorageTraits + 'static>(
    array: &Array<TStorage>,
    n_bins: usize,
    min: f64,
    max: f64,
    chunk_limit: usize,
) -> Result<(Vec<f64>, Vec<u64>), ArrayError> {
    let dt = array.data_type();
    if dt.is::<data_type::Int8DataType>() {
        calculate_histogram_t::<_, i8>(array, n_bins, min, max, chunk_limit)
    } else if dt.is::<data_type::Int16DataType>() {
        calculate_histogram_t::<_, i16>(array, n_bins, min, max, chunk_limit)
    } else if dt.is::<data_type::Int32DataType>() {
        calculate_histogram_t::<_, i32>(array, n_bins, min, max, chunk_limit)
    } else if dt.is::<data_type::Int64DataType>() {
        calculate_histogram_t::<_, i64>(array, n_bins, min, max, chunk_limit)
    } else if dt.is::<data_type::UInt8DataType>() {
        calculate_histogram_t::<_, u8>(array, n_bins, min, max, chunk_limit)
    } else if dt.is::<data_type::UInt16DataType>() {
        calculate_histogram_t::<_, u16>(array, n_bins, min, max, chunk_limit)
    } else if dt.is::<data_type::UInt32DataType>() {
        calculate_histogram_t::<_, u32>(array, n_bins, min, max, chunk_limit)
    } else if dt.is::<data_type::UInt64DataType>() {
        calculate_histogram_t::<_, u64>(array, n_bins, min, max, chunk_limit)
    } else if dt.is::<data_type::Float16DataType>() {
        calculate_histogram_t::<_, f16>(array, n_bins, min, max, chunk_limit)
    } else if dt.is::<data_type::BFloat16DataType>() {
        calculate_histogram_t::<_, bf16>(array, n_bins, min, max, chunk_limit)
    } else if dt.is::<data_type::Float32DataType>() {
        calculate_histogram_t::<_, f32>(array, n_bins, min, max, chunk_limit)
    } else if dt.is::<data_type::Float64DataType>() {
        calculate_histogram_t::<_, f64>(array, n_bins, min, max, chunk_limit)
    } else {
        unimplemented!("Data type not supported: {:?}", dt)
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
    let chunks = ArraySubset::new_with_shape(array.chunk_grid_shape().to_vec());

    let chunk_incr_histogram = |histogram: Result<Vec<u64>, ArrayError>,
                                chunk_indices: ArrayIndicesTinyVec| {
        let mut histogram = histogram?;
        let elements: Vec<T> = array.retrieve_chunk(&chunk_indices)?;
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
    let n_indices = indices.len();
    let hist = indices
        .into_par_iter()
        .fold_chunks(
            n_indices.div_ceil(chunk_limit).max(1),
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
