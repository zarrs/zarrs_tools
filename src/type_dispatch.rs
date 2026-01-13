//! Type dispatch utilities for reducing monomorphization in filter operations.
//!
//! This module provides centralized type conversion functions that convert between zarr data types and intermediate processing types (f32, f64, i32, i64, u32, u64).
//!
//! The intermediate type is chosen by promotion to a 32-bit type (E.g. f32, i32, u32) or 64-bit type (E.g. f64, i64, u64) as required.

use std::time::{Duration, Instant};

use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use zarrs::{
    array::{data_type, Array, ArrayError, ArraySubset, DataType},
    storage::{ReadableStorageTraits, ReadableWritableStorageTraits},
};

// ============================================================================
// Timing structs
// ============================================================================

/// Timing for retrieve operations (read + convert to intermediate).
#[derive(Debug, Clone, Copy, Default)]
pub struct RetrieveTiming {
    pub read: Duration,
    pub convert: Duration,
}

/// Timing for store operations (convert from intermediate + write).
#[derive(Debug, Clone, Copy, Default)]
pub struct StoreTiming {
    pub convert: Duration,
    pub write: Duration,
}

/// Combined timing for retrieve-and-store operations.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConversionTiming {
    pub read: Duration,
    pub process: Duration,
    pub write: Duration,
}

fn unsupported_data_type_error(data_type: &DataType) -> ArrayError {
    ArrayError::Other(format!("unsupported data type {:?}", data_type))
}

// ============================================================================
// Intermediate type selection
// ============================================================================

/// The intermediate type to use for type-erased operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntermediateType {
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
}

impl IntermediateType {
    /// Select the best intermediate type for the given data type.
    pub fn for_data_type(data_type: &DataType) -> Self {
        #[allow(clippy::if_same_then_else)]
        if data_type.is::<data_type::Float64DataType>() {
            IntermediateType::F64
        } else if data_type.is::<data_type::Float32DataType>()
            || data_type.is::<data_type::Float16DataType>()
            || data_type.is::<data_type::BFloat16DataType>()
        {
            IntermediateType::F32
        } else if data_type.is::<data_type::Int64DataType>() {
            IntermediateType::I64
        } else if data_type.is::<data_type::Int32DataType>()
            || data_type.is::<data_type::Int16DataType>()
            || data_type.is::<data_type::Int8DataType>()
        {
            IntermediateType::I32
        } else if data_type.is::<data_type::UInt64DataType>() {
            IntermediateType::U64
        } else if data_type.is::<data_type::UInt32DataType>()
            || data_type.is::<data_type::UInt16DataType>()
            || data_type.is::<data_type::UInt8DataType>()
            || data_type.is::<data_type::BoolDataType>()
        {
            IntermediateType::U32
        } else {
            IntermediateType::U32
        }
    }

    /// Select the best intermediate type for converting between two data types.
    /// Uses the "wider" type to avoid precision loss.
    pub fn for_conversion(input: &DataType, output: &DataType) -> Self {
        let in_type = Self::for_data_type(input);
        let out_type = Self::for_data_type(output);

        // If either is float, use float (prefer f64 if either needs it)
        if in_type == IntermediateType::F64 || out_type == IntermediateType::F64 {
            return IntermediateType::F64;
        }
        if in_type == IntermediateType::F32 || out_type == IntermediateType::F32 {
            return IntermediateType::F32;
        }

        // If either is signed, we need signed
        let in_signed = matches!(in_type, IntermediateType::I32 | IntermediateType::I64);
        let out_signed = matches!(out_type, IntermediateType::I32 | IntermediateType::I64);

        if in_signed || out_signed {
            // Use i64 if either needs 64-bit
            if in_type == IntermediateType::I64
                || out_type == IntermediateType::I64
                || in_type == IntermediateType::U64
                || out_type == IntermediateType::U64
            {
                IntermediateType::I64
            } else {
                IntermediateType::I32
            }
        } else {
            // Both unsigned
            if in_type == IntermediateType::U64 || out_type == IntermediateType::U64 {
                IntermediateType::U64
            } else {
                IntermediateType::U32
            }
        }
    }
}

// ============================================================================
// Type-erased value container
// ============================================================================

/// A container for type-erased array data.
pub enum TypeErasedVec {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

/// A container for type-erased ndarray data.
pub enum TypeErasedNdarray {
    F32(ndarray::ArrayD<f32>),
    F64(ndarray::ArrayD<f64>),
    I32(ndarray::ArrayD<i32>),
    I64(ndarray::ArrayD<i64>),
    U32(ndarray::ArrayD<u32>),
    U64(ndarray::ArrayD<u64>),
}

// ============================================================================
// Type conversion helpers
// ============================================================================

/// Convert `Vec<S>` to `Vec<T>`. No-op if types are identical.
#[inline]
fn convert_vec<S, T>(vec: Vec<S>) -> Vec<T>
where
    S: Copy + Send + Sync + 'static + AsPrimitive<T>,
    T: Copy + Send + Sync + 'static,
{
    if std::any::TypeId::of::<S>() == std::any::TypeId::of::<T>() {
        // Types are identical, cast without iteration
        let mut vec = std::mem::ManuallyDrop::new(vec);
        // SAFETY: S and T have the same TypeId, so they are the same type
        unsafe { Vec::from_raw_parts(vec.as_mut_ptr().cast::<T>(), vec.len(), vec.capacity()) }
    } else {
        vec.into_par_iter().map(|v| v.as_()).collect()
    }
}

/// Convert `ndarray<S>` to `ndarray<T>`. No-op if types are identical.
#[inline]
fn convert_ndarray<S, T>(arr: ndarray::ArrayD<S>) -> ndarray::ArrayD<T>
where
    S: Copy + 'static + AsPrimitive<T>,
    T: Copy + 'static,
{
    if std::any::TypeId::of::<S>() == std::any::TypeId::of::<T>() {
        // Types are identical, cast without iteration
        // SAFETY: S and T have the same TypeId, so they are the same type
        unsafe { std::mem::transmute::<ndarray::ArrayD<S>, ndarray::ArrayD<T>>(arr) }
    } else {
        arr.mapv(|v| v.as_())
    }
}

// ============================================================================
// Generic dispatch functions
// ============================================================================

/// Retrieve an array subset and convert all elements to type T.
/// Returns the converted elements and timing information.
pub fn retrieve_as<T, TStorage>(
    array: &Array<TStorage>,
    subset: &ArraySubset,
) -> Result<(Vec<T>, RetrieveTiming), ArrayError>
where
    T: Copy + Send + Sync + 'static,
    TStorage: ReadableStorageTraits + ?Sized + 'static,
    u8: AsPrimitive<T>,
    i8: AsPrimitive<T>,
    i16: AsPrimitive<T>,
    i32: AsPrimitive<T>,
    i64: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    half::f16: AsPrimitive<T>,
    half::bf16: AsPrimitive<T>,
    f32: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    macro_rules! retrieve_and_convert {
        ([$( ( $dt_type:ty, $rust_type:ty ) ),* ]) => {
            {
                let dt = array.data_type();
                $(if dt.is::<$dt_type>() {
                    let start = Instant::now();
                    let elements: Vec<$rust_type> = array.retrieve_array_subset(subset)?;
                    let read = start.elapsed();

                    let start = Instant::now();
                    let converted = convert_vec(elements);
                    let convert = start.elapsed();

                    return Ok((converted, RetrieveTiming { read, convert }));
                })*
                Err(unsupported_data_type_error(dt))
            }
        };
    }

    retrieve_and_convert!([
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
    ])
}

/// Convert type T elements and store to an array subset.
/// Returns timing information for convert and write phases.
pub fn store_from<T, TStorage>(
    array: &Array<TStorage>,
    subset: &ArraySubset,
    elements: Vec<T>,
) -> Result<StoreTiming, ArrayError>
where
    T: Copy + Send + Sync + 'static,
    TStorage: ReadableWritableStorageTraits + ?Sized + 'static,
    T: AsPrimitive<u8>,
    T: AsPrimitive<i8>,
    T: AsPrimitive<i16>,
    T: AsPrimitive<i32>,
    T: AsPrimitive<i64>,
    T: AsPrimitive<u16>,
    T: AsPrimitive<u32>,
    T: AsPrimitive<u64>,
    T: AsPrimitive<half::f16>,
    T: AsPrimitive<half::bf16>,
    T: AsPrimitive<f32>,
    T: AsPrimitive<f64>,
{
    macro_rules! convert_and_store {
        ([$( ( $dt_type:ty, $rust_type:ty ) ),* ]) => {
            {
                let dt = array.data_type();
                $(if dt.is::<$dt_type>() {
                    let start = Instant::now();
                    let converted: Vec<$rust_type> = convert_vec(elements);
                    let convert = start.elapsed();

                    let start = Instant::now();
                    array.store_array_subset(subset, converted)?;
                    let write = start.elapsed();

                    return Ok(StoreTiming { convert, write });
                })*
                Err(unsupported_data_type_error(dt))
            }
        };
    }

    convert_and_store!([
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
    ])
}

/// Retrieve an ndarray subset and convert all elements to type T.
/// Returns the converted ndarray and timing information.
pub fn retrieve_ndarray_as<T, TStorage>(
    array: &Array<TStorage>,
    subset: &ArraySubset,
) -> Result<(ndarray::ArrayD<T>, RetrieveTiming), ArrayError>
where
    T: Copy + Send + Sync + 'static,
    TStorage: ReadableStorageTraits + ?Sized + 'static,
    u8: AsPrimitive<T>,
    i8: AsPrimitive<T>,
    i16: AsPrimitive<T>,
    i32: AsPrimitive<T>,
    i64: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    half::f16: AsPrimitive<T>,
    half::bf16: AsPrimitive<T>,
    f32: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    macro_rules! retrieve_and_convert {
        ([$( ( $dt_type:ty, $rust_type:ty ) ),* ]) => {
            {
                let dt = array.data_type();
                $(if dt.is::<$dt_type>() {
                    let start = Instant::now();
                    let elements: ndarray::ArrayD<$rust_type> = array.retrieve_array_subset(subset)?;
                    let read = start.elapsed();

                    let start = Instant::now();
                    let converted = convert_ndarray(elements);
                    let convert = start.elapsed();

                    return Ok((converted, RetrieveTiming { read, convert }));
                })*
                Err(unsupported_data_type_error(dt))
            }
        };
    }

    retrieve_and_convert!([
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
    ])
}

/// Convert a type T ndarray and store to an array subset.
/// Returns timing information for convert and write phases.
pub fn store_ndarray_from<T, TStorage>(
    array: &Array<TStorage>,
    subset: &ArraySubset,
    elements: ndarray::ArrayD<T>,
) -> Result<StoreTiming, ArrayError>
where
    T: Copy + Send + Sync + 'static,
    TStorage: ReadableWritableStorageTraits + ?Sized + 'static,
    T: AsPrimitive<u8>,
    T: AsPrimitive<i8>,
    T: AsPrimitive<i16>,
    T: AsPrimitive<i32>,
    T: AsPrimitive<i64>,
    T: AsPrimitive<u16>,
    T: AsPrimitive<u32>,
    T: AsPrimitive<u64>,
    T: AsPrimitive<half::f16>,
    T: AsPrimitive<half::bf16>,
    T: AsPrimitive<f32>,
    T: AsPrimitive<f64>,
{
    macro_rules! convert_and_store {
        ([$( ( $dt_type:ty, $rust_type:ty ) ),* ]) => {
            {
                let dt = array.data_type();
                $(if dt.is::<$dt_type>() {
                    let start = Instant::now();
                    let converted: ndarray::ArrayD<$rust_type> = convert_ndarray(elements);
                    let convert = start.elapsed();

                    let start = Instant::now();
                    array.store_array_subset(subset, converted)?;
                    let write = start.elapsed();

                    return Ok(StoreTiming { convert, write });
                })*
                Err(unsupported_data_type_error(dt))
            }
        };
    }

    convert_and_store!([
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
    ])
}

// ============================================================================
// Smart dispatch functions
// ============================================================================

/// Retrieve an array subset using the best intermediate type for the data type.
/// Returns the type-erased data and timing information.
pub fn retrieve_type_erased<TStorage>(
    array: &Array<TStorage>,
    subset: &ArraySubset,
) -> Result<(TypeErasedVec, RetrieveTiming), ArrayError>
where
    TStorage: ReadableStorageTraits + ?Sized + 'static,
{
    match IntermediateType::for_data_type(array.data_type()) {
        IntermediateType::F32 => {
            retrieve_as::<f32, _>(array, subset).map(|(v, t)| (TypeErasedVec::F32(v), t))
        }
        IntermediateType::F64 => {
            retrieve_as::<f64, _>(array, subset).map(|(v, t)| (TypeErasedVec::F64(v), t))
        }
        IntermediateType::I32 => {
            retrieve_as::<i32, _>(array, subset).map(|(v, t)| (TypeErasedVec::I32(v), t))
        }
        IntermediateType::I64 => {
            retrieve_as::<i64, _>(array, subset).map(|(v, t)| (TypeErasedVec::I64(v), t))
        }
        IntermediateType::U32 => {
            retrieve_as::<u32, _>(array, subset).map(|(v, t)| (TypeErasedVec::U32(v), t))
        }
        IntermediateType::U64 => {
            retrieve_as::<u64, _>(array, subset).map(|(v, t)| (TypeErasedVec::U64(v), t))
        }
    }
}

/// Store a type-erased vec to an array subset.
/// Returns timing information for convert and write phases.
pub fn store_type_erased<TStorage>(
    array: &Array<TStorage>,
    subset: &ArraySubset,
    data: TypeErasedVec,
) -> Result<StoreTiming, ArrayError>
where
    TStorage: ReadableWritableStorageTraits + ?Sized + 'static,
{
    match data {
        TypeErasedVec::F32(v) => store_from(array, subset, v),
        TypeErasedVec::F64(v) => store_from(array, subset, v),
        TypeErasedVec::I32(v) => store_from(array, subset, v),
        TypeErasedVec::I64(v) => store_from(array, subset, v),
        TypeErasedVec::U32(v) => store_from(array, subset, v),
        TypeErasedVec::U64(v) => store_from(array, subset, v),
    }
}

/// Retrieve from input and store directly to output, converting types if needed.
/// Uses the appropriate intermediate type based on input/output data types.
/// Returns combined timing information (read, process, write).
pub fn retrieve_and_store_converting<TStorageIn, TStorageOut>(
    input: &Array<TStorageIn>,
    output: &Array<TStorageOut>,
    subset: &ArraySubset,
) -> Result<ConversionTiming, ArrayError>
where
    TStorageIn: ReadableStorageTraits + ?Sized + 'static,
    TStorageOut: ReadableWritableStorageTraits + ?Sized + 'static,
{
    if input.data_type() == output.data_type() {
        // No conversion needed, use raw bytes
        let start = Instant::now();
        let bytes = input.retrieve_array_subset::<zarrs::array::ArrayBytes>(subset)?;
        let read = start.elapsed();

        let start = Instant::now();
        output.store_array_subset(subset, bytes)?;
        let write = start.elapsed();

        Ok(ConversionTiming {
            read,
            process: Duration::ZERO,
            write,
        })
    } else {
        // Convert through an intermediate type
        match IntermediateType::for_conversion(input.data_type(), output.data_type()) {
            IntermediateType::F32 => retrieve_convert_store::<f32, _, _>(input, output, subset),
            IntermediateType::F64 => retrieve_convert_store::<f64, _, _>(input, output, subset),
            IntermediateType::I32 => retrieve_convert_store::<i32, _, _>(input, output, subset),
            IntermediateType::I64 => retrieve_convert_store::<i64, _, _>(input, output, subset),
            IntermediateType::U32 => retrieve_convert_store::<u32, _, _>(input, output, subset),
            IntermediateType::U64 => retrieve_convert_store::<u64, _, _>(input, output, subset),
        }
    }
}

/// Helper function that retrieves, converts, and stores with timing aggregation.
fn retrieve_convert_store<T, TStorageIn, TStorageOut>(
    input: &Array<TStorageIn>,
    output: &Array<TStorageOut>,
    subset: &ArraySubset,
) -> Result<ConversionTiming, ArrayError>
where
    T: Copy + Send + Sync + 'static,
    TStorageIn: ReadableStorageTraits + ?Sized + 'static,
    TStorageOut: ReadableWritableStorageTraits + ?Sized + 'static,
    u8: AsPrimitive<T>,
    i8: AsPrimitive<T>,
    i16: AsPrimitive<T>,
    i32: AsPrimitive<T>,
    i64: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    half::f16: AsPrimitive<T>,
    half::bf16: AsPrimitive<T>,
    f32: AsPrimitive<T>,
    f64: AsPrimitive<T>,
    T: AsPrimitive<u8>,
    T: AsPrimitive<i8>,
    T: AsPrimitive<i16>,
    T: AsPrimitive<i32>,
    T: AsPrimitive<i64>,
    T: AsPrimitive<u16>,
    T: AsPrimitive<u32>,
    T: AsPrimitive<u64>,
    T: AsPrimitive<half::f16>,
    T: AsPrimitive<half::bf16>,
    T: AsPrimitive<f32>,
    T: AsPrimitive<f64>,
{
    let (elements, retrieve_timing) = retrieve_as::<T, _>(input, subset)?;
    let store_timing = store_from(output, subset, elements)?;
    Ok(ConversionTiming {
        read: retrieve_timing.read,
        process: retrieve_timing.convert + store_timing.convert,
        write: store_timing.write,
    })
}
