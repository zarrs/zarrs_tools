use half::{bf16, f16};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use zarrs::{
    array::{Array, ArrayError, ArrayIndicesTinyVec, DataType, ElementOwned},
    array_subset::ArraySubset,
    storage::ReadableStorageTraits,
};

// TODO: Support Infinity, -Infinity, NaN, etc.
pub fn calculate_range<TStorage: ReadableStorageTraits + 'static>(
    array: &Array<TStorage>,
    chunk_limit: usize,
) -> Result<(serde_json::Number, serde_json::Number), ArrayError> {
    match array.data_type() {
        DataType::Int8 => {
            let (min, max) = calculate_range_t(array, i8::MIN, i8::MAX, chunk_limit)?;
            let min = serde_json::Number::from(min);
            let max = serde_json::Number::from(max);
            Ok((min, max))
        }
        DataType::Int16 => {
            let (min, max) = calculate_range_t(array, i16::MIN, i16::MAX, chunk_limit)?;
            let min = serde_json::Number::from(min);
            let max = serde_json::Number::from(max);
            Ok((min, max))
        }
        DataType::Int32 => {
            let (min, max) = calculate_range_t(array, i32::MIN, i32::MAX, chunk_limit)?;
            let min = serde_json::Number::from(min);
            let max = serde_json::Number::from(max);
            Ok((min, max))
        }
        DataType::Int64 => {
            let (min, max) = calculate_range_t(array, i64::MIN, i64::MAX, chunk_limit)?;
            let min = serde_json::Number::from(min);
            let max = serde_json::Number::from(max);
            Ok((min, max))
        }
        DataType::UInt8 => {
            let (min, max) = calculate_range_t(array, u8::MIN, u8::MAX, chunk_limit)?;
            let min = serde_json::Number::from(min);
            let max = serde_json::Number::from(max);
            Ok((min, max))
        }
        DataType::UInt16 => {
            let (min, max) = calculate_range_t(array, u16::MIN, u16::MAX, chunk_limit)?;
            let min = serde_json::Number::from(min);
            let max = serde_json::Number::from(max);
            Ok((min, max))
        }
        DataType::UInt32 => {
            let (min, max) = calculate_range_t(array, u32::MIN, u32::MAX, chunk_limit)?;
            let min = serde_json::Number::from(min);
            let max = serde_json::Number::from(max);
            Ok((min, max))
        }
        DataType::UInt64 => {
            let (min, max) = calculate_range_t(array, u64::MIN, u64::MAX, chunk_limit)?;
            let min = serde_json::Number::from(min);
            let max = serde_json::Number::from(max);
            Ok((min, max))
        }
        DataType::Float16 => {
            let (min, max) =
                calculate_range_t(array, f16::NEG_INFINITY, f16::INFINITY, chunk_limit)?;
            let min = serde_json::Number::from_f64(min.to_f64()).unwrap();
            let max = serde_json::Number::from_f64(max.to_f64()).unwrap();
            Ok((min, max))
        }
        DataType::BFloat16 => {
            let (min, max) =
                calculate_range_t(array, bf16::NEG_INFINITY, bf16::INFINITY, chunk_limit)?;
            let min = serde_json::Number::from_f64(min.to_f64()).unwrap();
            let max = serde_json::Number::from_f64(max.to_f64()).unwrap();
            Ok((min, max))
        }
        DataType::Float32 => {
            let (min, max) =
                calculate_range_t(array, f32::NEG_INFINITY, f32::INFINITY, chunk_limit)?;
            let min = serde_json::Number::from_f64(min as f64).unwrap();
            let max = serde_json::Number::from_f64(max as f64).unwrap();
            Ok((min, max))
        }
        DataType::Float64 => {
            let (min, max) =
                calculate_range_t(array, f64::NEG_INFINITY, f64::INFINITY, chunk_limit)?;
            let min = serde_json::Number::from_f64(min).unwrap();
            let max = serde_json::Number::from_f64(max).unwrap();
            Ok((min, max))
        }
        DataType::Bool | DataType::Complex64 | DataType::Complex128 | DataType::RawBits(_) => {
            unimplemented!("Data type not supported")
        }
        _ => unimplemented!("Data type not supported"),
    }
}

pub fn calculate_range_t<
    TStorage: ReadableStorageTraits + 'static,
    T: ElementOwned + PartialOrd + Send + Sync,
>(
    array: &Array<TStorage>,
    t_min: T,
    t_max: T,
    chunk_limit: usize,
) -> Result<(T, T), ArrayError> {
    let chunks = ArraySubset::new_with_shape(array.chunk_grid_shape().to_vec());

    let chunk_min_max = |chunk_indices: ArrayIndicesTinyVec| {
        // TODO: Codec concurrent limit
        let elements: Vec<T> = array.retrieve_chunk(&chunk_indices)?;
        let (mut min, mut max) = (t_min.clone(), t_max.clone());
        for element in &elements {
            min = if element < &min { element.clone() } else { min };
            max = if element > &max { element.clone() } else { max };
        }
        Ok::<_, ArrayError>((min, max))
    };

    let indices = chunks.indices();
    let iter_min_max = iter_concurrent_limit!(chunk_limit, indices, map, chunk_min_max);
    let (min, max) = iter_min_max
        .try_reduce_with(|(amin, amax), (bmin, bmax)| {
            Ok({
                let min = if amin < bmin { amin } else { bmin };
                let max = if amax < bmax { amax } else { bmax };
                (min, max)
            })
        })
        .expect("a value since the chunk is not empty")?;

    Ok((min, max))
}
