#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
#![doc(hidden)]

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::{
    num::NonZeroU64,
    sync::{Arc, Mutex},
    time::SystemTime,
};

use clap::Parser;
use num_traits::AsPrimitive;
use progress::{Progress, ProgressCallback};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use serde::{Deserialize, Serialize};
use zarrs::plugin::ExtensionIdentifier;
use zarrs::{
    array::{
        chunk_grid::RegularChunkGrid,
        codec::{
            array_to_bytes::sharding, ArrayCodecTraits, ArrayToBytesCodecTraits, BytesCodec, Codec,
            CodecMetadataOptions, CodecOptions, Crc32cCodec, NamedArrayToArrayCodec,
            NamedArrayToBytesCodec, NamedBytesToBytesCodec, ShardingCodec,
        },
        concurrency::RecommendedConcurrency,
        data_type::{
            BFloat16DataType, BoolDataType, Float16DataType, Float32DataType, Float64DataType,
            Int16DataType, Int32DataType, Int64DataType, Int8DataType, UInt16DataType,
            UInt32DataType, UInt64DataType, UInt8DataType,
        },
        Array, ArrayBuilder, ArrayError, ArrayIndicesTinyVec, ArrayShardedExt, ChunkCache,
        ChunkCacheDecodedLruChunkLimit, ChunkCacheDecodedLruChunkLimitThreadLocal,
        ChunkCacheDecodedLruSizeLimit, ChunkCacheDecodedLruSizeLimitThreadLocal, ChunkShape,
        CodecChain, DataType, DimensionName, FillValue, FillValueMetadataV3, NamedDataType,
    },
    array_subset::{ArraySubset, IncompatibleDimensionalityError},
    config::global_config,
    metadata::v3::MetadataV3,
    storage::{ReadableStorageTraits, ReadableWritableListableStorageTraits},
};

pub mod filter;
pub mod info;
pub mod progress;

/// The `zarrs` tools version with the `zarrs` version.
///
/// Example:
/// `zarrs_tools 0.6.0-beta.1 (zarrs 0.18.0-beta.0)`
pub const ZARRS_TOOLS_VERSION_WITH_ZARRS: &str = const_format::formatcp!(
    "{} (zarrs {})",
    env!("CARGO_PKG_VERSION"),
    zarrs::version::version_str(),
);

#[derive(Parser)]
#[allow(rustdoc::bare_urls)]
pub struct ZarrEncodingArgs {
    /// Fill value. See https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#fill-value
    ///
    /// The fill value must be compatible with the data type.
    ///
    /// Examples:
    ///   int/uint: 0 100 -100
    ///   float: 0.0 "NaN" "Infinity" "-Infinity"
    ///   r*: "[0, 255]"
    #[arg(short, long, verbatim_doc_comment, allow_hyphen_values(true), value_parser = parse_fill_value)]
    pub fill_value: FillValueMetadataV3,

    /// The chunk key encoding separator. Either . or /.
    #[arg(long, default_value_t = '/')]
    pub separator: char,

    /// Chunk shape. A comma separated list of the chunk size along each array dimension.
    ///
    /// If any dimension has size zero, it will be set to match the array shape.
    #[arg(short, long, required = true, value_delimiter = ',')]
    pub chunk_shape: Vec<u64>,

    /// Shard shape (optional). A comma separated list of the shard size along each array dimension.
    ///
    /// If specified, the array is encoded using the sharding codec.
    /// If any dimension has size zero, it will be set to match the array shape.
    #[arg(short, long, verbatim_doc_comment, value_delimiter = ',')]
    pub shard_shape: Option<Vec<u64>>,

    /// Array to array codecs (optional).
    ///
    /// JSON holding an array of array to array codec metadata.
    ///
    /// Examples:
    ///   '[ { "name": "transpose", "configuration": { "order": [0, 2, 1] } } ]'
    ///   '[ { "name": "bitround", "configuration": { "keepbits": 9 } } ]'
    #[arg(long, verbatim_doc_comment)]
    pub array_to_array_codecs: Option<String>,

    /// Array to bytes codec (optional).
    ///
    /// JSON holding array to bytes codec metadata.
    /// If unspecified, this defaults to the `bytes` codec.
    ///
    /// The sharding codec can be used by setting `shard_shape`, but this can also be done explicitly here.
    ///
    /// Examples:
    ///   '{ "name": "bytes", "configuration": { "endian": "little" } }'
    ///   '{ "name": "pcodec", "configuration": { "level": 12 } }'
    ///   '{ "name": "zfp", "configuration": { "mode": "fixedprecision", "precision": 19 } }'
    #[arg(long, verbatim_doc_comment)]
    pub array_to_bytes_codec: Option<String>,

    /// Bytes to bytes codecs (optional).
    ///
    /// JSON holding an array of bytes to bytes codec configurations.
    ///
    /// Examples:
    ///   '[ { "name": "blosc", "configuration": { "cname": "blosclz", "clevel": 9, "shuffle": "bitshuffle", "typesize": 2, "blocksize": 0 } } ]'
    ///   '[ { "name": "bz2", "configuration": { "level": 9 } } ]'
    ///   '[ { "name": "crc32c" ]'
    ///   '[ { "name": "gzip", "configuration": { "level": 9 } } ]'
    ///   '[ { "name": "zstd", "configuration": { "level": 22, "checksum": false } } ]'
    #[arg(long, verbatim_doc_comment)]
    pub bytes_to_bytes_codecs: Option<String>,

    /// Attributes (optional).
    ///
    /// JSON holding array attributes.
    #[arg(long)]
    pub attributes: Option<String>,
}

fn parse_data_type(data_type: &str) -> std::io::Result<MetadataV3> {
    serde_json::from_value(serde_json::Value::String(data_type.to_string()))
        .map_err(|err| std::io::Error::other(err.to_string()))
}

fn parse_fill_value(fill_value: &str) -> std::io::Result<FillValueMetadataV3> {
    serde_json::from_str(fill_value).map_err(|err| std::io::Error::other(err.to_string()))
}

#[must_use]
pub fn get_array_builder(
    encoding_args: &ZarrEncodingArgs,
    array_shape: &[u64],
    data_type: NamedDataType,
    dimension_names: Option<Vec<DimensionName>>,
) -> zarrs::array::ArrayBuilder {
    // Set the chunk/shard shape to the array shape where it is 0, otherwise make it <= array shape
    let shard_shape = encoding_args.shard_shape.as_ref().map(|shard_shape| {
        std::iter::zip(shard_shape, array_shape)
            .map(|(&s, &a)| if s == 0 { a } else { std::cmp::min(s, a) })
            .collect::<Vec<_>>()
    });

    // Also ensure shard shape is a multiple of chunk shape
    let chunk_shape = std::iter::zip(&encoding_args.chunk_shape, array_shape)
        .map(|(&c, &a)| if c == 0 { a } else { c })
        .collect::<Vec<_>>();
    let shard_shape: Option<Vec<u64>> = shard_shape.map(|shard_shape| {
        std::iter::zip(&shard_shape, &chunk_shape)
            .map(|(s, c)| {
                // the shard shape must be a multiple of the chunk shape
                s.next_multiple_of(*c)
            })
            .collect()
    });

    // Get the "block shape", which is the shard shape if sharding, otherwise the chunk shape
    let block_shape = shard_shape
        .as_ref()
        .map_or(&chunk_shape, |shard_shape| shard_shape);

    // Get array to array codecs
    let array_to_array_codecs = encoding_args.array_to_array_codecs.as_ref().map_or_else(
        Vec::new,
        |array_to_array_codecs| {
            let metadatas: Vec<MetadataV3> =
                serde_json::from_str(array_to_array_codecs.as_str()).unwrap();
            let mut codecs = Vec::with_capacity(metadatas.len());
            for metadata in metadatas {
                codecs.push(match Codec::from_metadata(&metadata).unwrap() {
                    Codec::ArrayToArray(codec) => codec,
                    _ => panic!("Must be an array to array codec"),
                });
            }
            codecs
        },
    );

    // Get array to bytes codec
    let array_to_bytes_codec = encoding_args.array_to_bytes_codec.as_ref().map_or_else(
        || {
            let codec: Arc<dyn ArrayToBytesCodecTraits> = Arc::<BytesCodec>::default();
            codec
        },
        |array_codec| {
            let metadata = MetadataV3::try_from(array_codec.as_str()).unwrap();
            match Codec::from_metadata(&metadata).unwrap() {
                Codec::ArrayToBytes(codec) => codec,
                _ => panic!("Must be an array to bytes codec"),
            }
        },
    );

    // Get bytes to bytes codecs
    let bytes_to_bytes_codecs = encoding_args.bytes_to_bytes_codecs.as_ref().map_or_else(
        Vec::new,
        |bytes_to_bytes_codecs| {
            let metadatas: Vec<MetadataV3> =
                serde_json::from_str(bytes_to_bytes_codecs.as_str()).unwrap();
            let mut codecs = Vec::with_capacity(metadatas.len());
            for metadata in metadatas {
                codecs.push(match Codec::from_metadata(&metadata).unwrap() {
                    Codec::BytesToBytes(codec) => codec,
                    _ => panic!("Must be a bytes to bytes codec"),
                });
            }
            codecs
        },
    );

    // Get data type / fill value
    let fill_value = data_type
        .fill_value_from_metadata(&encoding_args.fill_value)
        .unwrap();

    // Create array
    let mut array_builder = ArrayBuilder::new(
        array_shape.to_vec(),
        block_shape.clone(),
        data_type,
        fill_value,
    );
    array_builder.dimension_names(dimension_names);
    if let Some(attributes) = &encoding_args.attributes {
        let attributes: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(attributes).expect("Attributes are invalid.");
        array_builder.attributes(attributes);
    }
    array_builder.chunk_key_encoding_default_separator(encoding_args.separator.try_into().unwrap());
    if shard_shape.is_some() {
        let index_codecs = Arc::new(CodecChain::new(
            vec![],
            Arc::<BytesCodec>::default(),
            vec![Arc::new(Crc32cCodec::new())],
        ));
        let inner_codecs = Arc::new(CodecChain::new(
            array_to_array_codecs,
            array_to_bytes_codec,
            bytes_to_bytes_codecs,
        ));
        let chunk_shape_nonzero: ChunkShape = chunk_shape
            .iter()
            .map(|&s| NonZeroU64::new(s).unwrap())
            .collect();
        array_builder.array_to_bytes_codec(Arc::new(ShardingCodec::new(
            chunk_shape_nonzero,
            inner_codecs,
            index_codecs,
            sharding::ShardingIndexLocation::End,
        )));
    } else {
        array_builder.array_to_array_codecs(array_to_array_codecs);
        array_builder.array_to_bytes_codec(array_to_bytes_codec);
        array_builder.bytes_to_bytes_codecs(bytes_to_bytes_codecs);
    }

    array_builder
}

fn is_false(value: &bool) -> bool {
    !value
}

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]

pub struct ZarrReencodingArgs {
    /// The data type as a string
    ///
    /// Changing between primitive data types is supported and uses standard rust numeric casting. See <https://doc.rust-lang.org/reference/expressions/operator-expr.html#r-expr.as.numeric>.
    /// - Casting from a larger integer to a smaller integer will truncate,
    /// - Casting from a float to an integer will round the float towards zero
    /// - Casting from an integer to float will produce the closest possible float
    ///
    /// Valid data types:
    ///   - bool
    ///   - int8, int16, int32, int64
    ///   - uint8, uint16, uint32, uint64
    ///   - float16, float32, float64, bfloat16
    ///   - complex64, complex 128
    ///   - r* (raw bits, where * is a multiple of 8)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(short, long, verbatim_doc_comment, value_parser = parse_data_type)]
    pub data_type: Option<MetadataV3>,

    /// Fill value. See <https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#fill-value>
    ///
    /// The fill value must be compatible with the data type.
    ///
    /// Examples:
    ///   int/uint: 0 100 -100
    ///   float: 0.0 "NaN" "Infinity" "-Infinity"
    ///   r*: "[0, 255]"
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(short, long, verbatim_doc_comment, allow_hyphen_values(true), value_parser = parse_fill_value)]
    pub fill_value: Option<FillValueMetadataV3>,

    /// The chunk key encoding separator. Either . or /.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub separator: Option<char>,

    /// Chunk shape. A comma separated list of the chunk size along each array dimension.
    ///
    /// If any dimension has size zero, it will be set to match the array shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(short, long, value_delimiter = ',')]
    pub chunk_shape: Option<Vec<u64>>,

    /// Shard shape. A comma separated list of the shard size along each array dimension.
    ///
    /// If specified, the array is encoded using the sharding codec.
    /// If any dimension has size zero, it will be set to match the array shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(short, long, verbatim_doc_comment, value_delimiter = ',')]
    pub shard_shape: Option<Vec<u64>>,

    /// If true, the sharding of the input will be ignored.
    #[serde(skip_serializing_if = "is_false")]
    #[arg(long, verbatim_doc_comment)]
    pub ignore_input_sharding: bool,

    /// Array to array codecs.
    ///
    /// JSON holding an array of array to array codec metadata.
    ///
    /// Examples:
    ///   '[ { "name": "transpose", "configuration": { "order": [0, 2, 1] } } ]'
    ///   '[ { "name": "bitround", "configuration": { "keepbits": 9 } } ]'
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long, verbatim_doc_comment)]
    pub array_to_array_codecs: Option<String>,

    /// Array to bytes codec.
    ///
    /// JSON holding array to bytes codec metadata.
    ///
    /// Examples:
    ///   '{ "name": "bytes", "configuration": { "endian": "little" } }'
    ///   '{ "name": "pcodec", "configuration": { "level": 12 } }'
    ///   '{ "name": "zfp", "configuration": { "mode": "fixedprecision", "precision": 19 } }'
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long, verbatim_doc_comment)]
    pub array_to_bytes_codec: Option<String>,

    /// Bytes to bytes codecs.
    ///
    /// JSON holding an array bytes to bytes codec configurations.
    ///
    /// Examples:
    ///   '[ { "name": "blosc", "configuration": { "cname": "blosclz", "clevel": 9, "shuffle": "bitshuffle", "typesize": 2, "blocksize": 0 } } ]'
    ///   '[ { "name": "bz2", "configuration": { "level": 9 } } ]'
    ///   '[ { "name": "crc32c" } ]'
    ///   '[ { "name": "gzip", "configuration": { "level": 9 } } ]'
    ///   '[ { "name": "zstd", "configuration": { "level": 22, "checksum": false } } ]'
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long, verbatim_doc_comment)]
    pub bytes_to_bytes_codecs: Option<String>,

    /// Dimension names (optional). Comma separated.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long, verbatim_doc_comment, value_delimiter = ',')]
    pub dimension_names: Option<Vec<String>>,

    /// Attributes (optional).
    ///
    /// JSON holding array attributes.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub attributes: Option<String>,

    /// Attributes to append (optional).
    ///
    /// JSON holding array attributes.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub attributes_append: Option<String>,
}

pub enum ZarrReEncodingChangeType {
    None,
    Metadata,
    MetadataAndChunks,
}

impl ZarrReencodingArgs {
    pub fn change_type(&self) -> ZarrReEncodingChangeType {
        if self.data_type.is_some()
            || self.fill_value.is_some()
            || self.separator.is_some()
            || self.chunk_shape.is_some()
            || self.shard_shape.is_some()
            || self.ignore_input_sharding
            || self.array_to_array_codecs.is_some()
            || self.array_to_bytes_codec.is_some()
            || self.bytes_to_bytes_codecs.is_some()
        {
            ZarrReEncodingChangeType::MetadataAndChunks
        } else if self.dimension_names.is_some()
            || self.attributes.is_some()
            || self.attributes_append.is_some()
        {
            ZarrReEncodingChangeType::Metadata
        } else {
            ZarrReEncodingChangeType::None
        }
    }
}

#[must_use]
pub fn get_array_builder_reencode<TStorage: ?Sized>(
    encoding_args: &ZarrReencodingArgs,
    array: &Array<TStorage>,
    array_shape: Option<Vec<u64>>,
) -> zarrs::array::ArrayBuilder {
    let codecs = array.codecs();
    let array_to_bytes_codec = codecs.array_to_bytes_codec();
    let array_to_bytes_identifier = array_to_bytes_codec.identifier();
    let (
        chunk_shape,
        shard_shape,
        array_to_array_codecs,
        array_array_to_bytes_codec,
        bytes_to_bytes_codecs,
    ) = if array_to_bytes_identifier == ShardingCodec::IDENTIFIER {
        let sharding_configuration = array_to_bytes_codec
            .configuration(&CodecMetadataOptions::default())
            .unwrap();
        // println!("{sharding_configuration:#?}");
        let chunk_shape: Vec<u64> =
            serde_json::from_value(sharding_configuration["chunk_shape"].clone()).unwrap();
        let shard_shape = array
            .chunk_shape(&vec![0; chunk_shape.len()])
            .unwrap()
            .iter()
            .map(|i| i.get())
            .collect::<Vec<_>>();
        let codecs: Vec<MetadataV3> =
            serde_json::from_value(sharding_configuration["codecs"].clone()).unwrap();
        let codec_chain = CodecChain::from_metadata(&codecs).unwrap();
        let array_to_array_codecs = codec_chain.array_to_array_codecs().to_vec();
        let array_to_bytes_codec = codec_chain.array_to_bytes_codec().clone();
        let bytes_to_bytes_codecs = codec_chain.bytes_to_bytes_codecs().to_vec();
        (
            chunk_shape,
            if encoding_args.ignore_input_sharding {
                None
            } else {
                Some(shard_shape)
            },
            array_to_array_codecs,
            array_to_bytes_codec,
            bytes_to_bytes_codecs,
        )
    } else {
        let chunk_shape = array.chunk_grid_shape().to_vec();
        let shard_shape = None;
        let array_to_array_codecs = array.codecs().array_to_array_codecs().to_vec();
        let array_to_bytes_codec = array.codecs().array_to_bytes_codec().clone();
        let bytes_to_bytes_codecs = array.codecs().bytes_to_bytes_codecs().to_vec();
        (
            chunk_shape,
            shard_shape,
            array_to_array_codecs,
            array_to_bytes_codec,
            bytes_to_bytes_codecs,
        )
    };

    // Chunk shape override
    let chunk_shape = encoding_args
        .chunk_shape
        .as_ref()
        .map(|chunk_shape| {
            std::iter::zip(chunk_shape.as_slice(), array.shape())
                .map(|(&c, &a)| if c == 0 { a } else { c })
                .collect::<Vec<_>>()
        })
        .unwrap_or(chunk_shape);

    // Shard shape override
    // Set the shard shape to the array shape where it is 0, otherwise make it <= array shape
    let shard_shape: Option<Vec<u64>> =
        encoding_args
            .shard_shape
            .as_ref()
            .map_or(shard_shape, |shard_shape| {
                let shard_shape = std::iter::zip(shard_shape, array.shape())
                    .map(|(&s, &a)| if s == 0 { a } else { std::cmp::min(s, a) })
                    .collect::<Vec<_>>();
                Some(shard_shape)
            });

    // Ensure shard shape is a multiple of the chunk shape
    let shard_shape: Option<Vec<u64>> = shard_shape.clone().map_or(shard_shape, |shard_shape| {
        let shard_shape = std::iter::zip(shard_shape.as_slice(), chunk_shape.as_slice())
            .map(|(s, c)| {
                // the shard shape must be a multiple of the chunk shape
                s.next_multiple_of(*c)
            })
            .collect::<Vec<_>>();
        Some(shard_shape)
    });

    // println!("{chunk_shape:?} {shard_shape:?}");

    // Get array to array codecs
    let array_to_array_codecs = encoding_args.array_to_array_codecs.clone().map_or(
        array_to_array_codecs,
        |array_to_array_codecs| {
            let metadatas: Vec<MetadataV3> =
                serde_json::from_str(array_to_array_codecs.as_str()).unwrap();
            let mut codecs = Vec::with_capacity(metadatas.len());
            for metadata in metadatas {
                let codec = match Codec::from_metadata(&metadata).unwrap() {
                    Codec::ArrayToArray(codec) => codec,
                    _ => panic!("Must be an array to array codec"),
                };
                codecs.push(NamedArrayToArrayCodec::new(
                    metadata.name().to_string(),
                    codec,
                ));
            }
            codecs
        },
    );

    // Get array to bytes codec
    let array_to_bytes_codec = encoding_args.array_to_bytes_codec.as_ref().map_or(
        array_array_to_bytes_codec,
        |array_codec| {
            let metadata = MetadataV3::try_from(array_codec.as_str()).unwrap();
            let codec = match Codec::from_metadata(&metadata).unwrap() {
                Codec::ArrayToBytes(codec) => codec,
                _ => panic!("Must be an array to bytes codec"),
            };
            NamedArrayToBytesCodec::new(metadata.name().to_string(), codec)
        },
    );

    // Get bytes to bytes codecs
    let bytes_to_bytes_codecs = encoding_args.bytes_to_bytes_codecs.as_ref().map_or(
        bytes_to_bytes_codecs,
        |bytes_to_bytes_codecs| {
            let metadatas: Vec<MetadataV3> =
                serde_json::from_str(bytes_to_bytes_codecs.as_str()).unwrap();
            let mut codecs = Vec::with_capacity(metadatas.len());
            for metadata in metadatas {
                let codec = match Codec::from_metadata(&metadata).unwrap() {
                    Codec::BytesToBytes(codec) => codec,
                    _ => panic!("Must be a bytes to bytes codec"),
                };
                codecs.push(NamedBytesToBytesCodec::new(
                    metadata.name().to_string(),
                    codec,
                ));
            }
            codecs
        },
    );

    // Create array
    let mut array_builder = array.builder();

    if let Some(attributes) = &encoding_args.attributes {
        let attributes: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(attributes).expect("Attributes are invalid.");
        array_builder.attributes(attributes);
    }

    if let Some(attributes_append) = &encoding_args.attributes_append {
        let mut attributes_append: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(attributes_append).expect("Attributes append are invalid.");
        array_builder
            .attributes_mut()
            .append(&mut attributes_append);
    }

    if let Some(separator) = encoding_args.separator {
        array_builder.chunk_key_encoding_default_separator(separator.try_into().unwrap());
    }

    if let Some(array_shape) = array_shape {
        array_builder.shape(array_shape);
    }

    let data_type = if let Some(data_type) = &encoding_args.data_type {
        let data_type = NamedDataType::try_from(data_type).unwrap();
        array_builder.data_type(data_type.clone());
        data_type
    } else {
        array.named_data_type().clone()
    };

    if let Some(dimension_names) = encoding_args.dimension_names.clone() {
        // TODO: Remove clone with zarrs 0.15.1+
        array_builder.dimension_names(dimension_names.into());
    }

    if let Some(fill_value) = &encoding_args.fill_value {
        // An explicit fill value was supplied
        let fill_value = data_type.fill_value_from_metadata(fill_value).unwrap();
        array_builder.fill_value(fill_value);
    } else if let Some(data_type) = &encoding_args.data_type {
        // The data type was changed, but no fill value supplied, so just cast it
        let data_type = NamedDataType::try_from(data_type).unwrap();
        let fill_value = convert_fill_value(array.data_type(), array.fill_value(), &data_type);
        array_builder.fill_value(fill_value);
    }

    if let Some(shard_shape) = shard_shape {
        array_builder.chunk_grid_metadata(shard_shape);
        let index_codecs = Arc::new(CodecChain::new(
            vec![],
            Arc::<BytesCodec>::default(),
            vec![Arc::new(Crc32cCodec::new())],
        ));
        let inner_codecs = Arc::new(CodecChain::new_named(
            array_to_array_codecs,
            array_to_bytes_codec,
            bytes_to_bytes_codecs,
        ));
        array_builder.array_to_array_codecs(vec![]);
        let chunk_shape_nonzero: ChunkShape = chunk_shape
            .iter()
            .map(|&s| NonZeroU64::new(s).unwrap())
            .collect();
        array_builder.array_to_bytes_codec(Arc::new(ShardingCodec::new(
            chunk_shape_nonzero,
            inner_codecs,
            index_codecs,
            sharding::ShardingIndexLocation::End,
        )));
        array_builder.bytes_to_bytes_codecs(vec![]);
    } else {
        array_builder.array_to_array_codecs_named(array_to_array_codecs);
        array_builder.array_to_bytes_codec_named(array_to_bytes_codec);
        array_builder.bytes_to_bytes_codecs_named(bytes_to_bytes_codecs);
    }

    array_builder
}

pub enum CacheSize {
    None,
    SizeTotal(u64),
    SizePerThread(u64),
    ChunksTotal(u64),
    ChunksPerThread(u64),
}

fn convert_and_store_subset(
    array_in: &Arc<Array<dyn ReadableStorageTraits>>,
    array_out: &Array<dyn ReadableWritableListableStorageTraits>,
    subset: &ArraySubset,
    progress: &Progress,
    bytes_decoded: &Mutex<usize>,
) -> anyhow::Result<()> {
    macro_rules! convert_elements {
        ( $t_in:ty, $t_out:ty ) => {{
            let elements_in: Vec<$t_in> =
                progress.read(|| array_in.retrieve_array_subset(subset))?;
            let bytes_size = elements_in.len() * std::mem::size_of::<$t_in>();
            let elements_out = progress.process(|| {
                elements_in
                    .into_par_iter()
                    .map(|value| value.as_())
                    .collect::<Vec<$t_out>>()
            });
            progress.write(|| array_out.store_array_subset(subset, elements_out))?;
            *bytes_decoded.lock().unwrap() += bytes_size;
        }};
    }

    macro_rules! convert_from_input {
        ( $t_out:ty, [$( ( $dt_type:ty, $t_in:ty ) ),* ]) => {
            {
                let dt_id = array_in.data_type().identifier();
                $(if dt_id == <$dt_type>::IDENTIFIER { convert_elements!($t_in, $t_out) } else)*
                { anyhow::bail!("Unsupported input data type: {}", array_in.data_type().identifier()) }
            }
        };
    }

    macro_rules! convert_to_output {
        ([$( ( $dt_type:ty, $type_out:ty ) ),* ]) => {
            {
                let dt_id = array_out.data_type().identifier();
                $(if dt_id == <$dt_type>::IDENTIFIER {
                    convert_from_input!($type_out, [
                        (BoolDataType, u8),
                        (Int8DataType, i8),
                        (Int16DataType, i16),
                        (Int32DataType, i32),
                        (Int64DataType, i64),
                        (UInt8DataType, u8),
                        (UInt16DataType, u16),
                        (UInt32DataType, u32),
                        (UInt64DataType, u64),
                        (BFloat16DataType, half::bf16),
                        (Float16DataType, half::f16),
                        (Float32DataType, f32),
                        (Float64DataType, f64)
                    ])
                } else)*
                { anyhow::bail!("Unsupported output data type: {}", array_out.data_type().identifier()) }
            }
        };
    }

    convert_to_output!([
        (BoolDataType, u8),
        (Int8DataType, i8),
        (Int16DataType, i16),
        (Int32DataType, i32),
        (Int64DataType, i64),
        (UInt8DataType, u8),
        (UInt16DataType, u16),
        (UInt32DataType, u32),
        (UInt64DataType, u64),
        (BFloat16DataType, half::bf16),
        (Float16DataType, half::f16),
        (Float32DataType, f32),
        (Float64DataType, f64)
    ]);

    Ok(())
}

pub fn do_reencode(
    array_in: Arc<Array<dyn ReadableStorageTraits>>,
    array_out: &Array<dyn ReadableWritableListableStorageTraits>,
    validate: bool,
    concurrent_chunks: Option<usize>,
    progress_callback: &ProgressCallback,
    cache_size: CacheSize,
    write_shape: Option<Vec<NonZeroU64>>,
) -> anyhow::Result<(f32, f32, f32, usize)> {
    if let Some(write_shape) = &write_shape {
        if write_shape.len() != array_out.chunk_grid().dimensionality() {
            anyhow::bail!("Write shape dimensionality does not match chunk grid dimensionality");
        }
    }

    let start = SystemTime::now();
    let bytes_decoded = Mutex::new(0);

    let cache: Option<Arc<dyn ChunkCache>> = match cache_size {
        CacheSize::None => None,
        CacheSize::SizeTotal(size) => Some(Arc::new(ChunkCacheDecodedLruSizeLimit::new(
            array_in.clone(),
            size,
        ))),
        CacheSize::SizePerThread(size) => Some(Arc::new(
            ChunkCacheDecodedLruSizeLimitThreadLocal::new(array_in.clone(), size),
        )),
        CacheSize::ChunksTotal(chunks) => Some(Arc::new(ChunkCacheDecodedLruChunkLimit::new(
            array_in.clone(),
            chunks,
        ))),
        CacheSize::ChunksPerThread(chunks) => Some(Arc::new(
            ChunkCacheDecodedLruChunkLimitThreadLocal::new(array_in.clone(), chunks),
        )),
    };

    let chunk_shape = array_out
        .chunk_shape(&vec![0; array_out.chunk_grid().dimensionality()])
        .unwrap();
    let chunks = ArraySubset::new_with_shape(array_out.chunk_grid_shape().to_vec());

    let concurrent_target = std::thread::available_parallelism().unwrap().get();
    let (chunks_concurrent_limit, codec_concurrent_target) = calculate_chunk_and_codec_concurrency(
        concurrent_target,
        concurrent_chunks,
        &array_out.codecs(),
        chunks.num_elements_usize(),
        &chunk_shape,
        array_out.data_type(),
    );

    let is_sharded = array_out.is_sharded();
    let write_shape = if is_sharded { write_shape } else { None };

    let codec_options = CodecOptions::default()
        .with_concurrent_target(codec_concurrent_target)
        .with_experimental_partial_encoding(write_shape.is_some());

    let num_iterations = if let Some(write_shape) = &write_shape {
        let indices = chunks.indices();
        indices
            .into_par_iter()
            .map(|chunk_indices| {
                let chunk_subset = array_out.chunk_subset(&chunk_indices).unwrap();
                chunk_subset
                    .shape()
                    .iter()
                    .zip(write_shape)
                    .map(|(s, w)| s.div_ceil(w.get()))
                    .sum::<u64>()
            })
            .sum::<u64>()
    } else {
        chunks.num_elements()
    };
    let num_iterations = usize::try_from(num_iterations).unwrap();

    let progress = Progress::new(num_iterations, progress_callback);

    let retrieve_array_subset = |subset: &ArraySubset| {
        if let Some(cache) = &cache {
            Ok(Arc::unwrap_or_clone(
                cache.retrieve_array_subset_bytes(subset, &codec_options)?,
            ))
        } else {
            array_in.retrieve_array_subset_opt(subset, &codec_options)
        }
    };

    let indices = chunks.indices();
    if array_in.data_type().identifier() == array_out.data_type().identifier() {
        iter_concurrent_limit!(
            chunks_concurrent_limit,
            indices,
            try_for_each,
            |chunk_indices: ArrayIndicesTinyVec| {
                let chunk_subset = array_out.chunk_subset(&chunk_indices).unwrap();
                if let Some(write_shape) = &write_shape {
                    use zarrs::array::chunk_grid::ChunkGridTraits;
                    let write_grid =
                        RegularChunkGrid::new(chunk_subset.shape().to_vec(), write_shape.clone())
                            .map_err(|_| {
                            IncompatibleDimensionalityError::new(
                                write_shape.len(),
                                chunk_subset.dimensionality(),
                            )
                        })?;
                    // FIXME: Add ChunkGrid iteration in `zarrs`
                    for chunk_write in
                        ArraySubset::new_with_shape(write_grid.grid_shape().to_vec()).indices()
                    {
                        let chunk_subset_write = write_grid
                            .subset(&chunk_write)
                            .expect("matching dimensionality")
                            .expect("determinate for regular chunk grid");
                        let chunk_subset_write = chunk_subset_write.overlap(&chunk_subset)?;
                        let bytes = progress.read(|| retrieve_array_subset(&chunk_subset_write))?;
                        *bytes_decoded.lock().unwrap() += bytes.size();
                        progress.write(|| {
                            array_out.store_array_subset_opt(
                                &chunk_subset_write,
                                bytes,
                                &codec_options,
                            )
                        })?;
                        progress.next();
                    }
                } else {
                    let bytes = progress.read(|| retrieve_array_subset(&chunk_subset))?;
                    *bytes_decoded.lock().unwrap() += bytes.size();

                    if validate {
                        progress.write(|| {
                            array_out.store_chunk_opt(&chunk_indices, bytes.clone(), &codec_options)
                        })?;
                        let bytes_out = array_out
                            .retrieve_chunk_opt(&chunk_indices, &codec_options)
                            .unwrap();
                        assert!(bytes == bytes_out);
                        // let bytes_in = array_in
                        //     .retrieve_array_subset_opt(&chunk_subset, &codec_options)
                        //     .unwrap();
                        // assert!(bytes_in == bytes_out);
                    } else {
                        progress.write(|| {
                            array_out.store_chunk_opt(&chunk_indices, bytes, &codec_options)
                        })?;
                    }
                    progress.next();
                }
                Ok::<_, ArrayError>(())
            }
        )?;
    } else {
        // Data type conversion required
        let convert_data = |chunk_indices: ArrayIndicesTinyVec| {
            let chunk_subset = array_out.chunk_subset(&chunk_indices).unwrap();
            if let Some(write_shape) = &write_shape {
                use zarrs::array::chunk_grid::ChunkGridTraits;
                let write_grid =
                    RegularChunkGrid::new(chunk_subset.shape().to_vec(), write_shape.clone())
                        .map_err(|_| {
                            IncompatibleDimensionalityError::new(
                                write_shape.len(),
                                chunk_subset.dimensionality(),
                            )
                        })?;
                // FIXME: Add ChunkGrid iteration in `zarrs`
                for chunk_write in
                    ArraySubset::new_with_shape(write_grid.grid_shape().to_vec()).indices()
                {
                    let chunk_subset_write = write_grid
                        .subset(&chunk_write)
                        .expect("matching dimensionality")
                        .expect("determinate for regular chunk grid");
                    let chunk_subset_write = chunk_subset_write.overlap(&chunk_subset)?;
                    convert_and_store_subset(
                        &array_in,
                        array_out,
                        &chunk_subset_write,
                        &progress,
                        &bytes_decoded,
                    )?;
                    progress.next();
                }
            } else {
                convert_and_store_subset(
                    &array_in,
                    array_out,
                    &chunk_subset,
                    &progress,
                    &bytes_decoded,
                )?;
                progress.next();
            }
            Ok::<_, anyhow::Error>(())
        };

        iter_concurrent_limit!(chunks_concurrent_limit, indices, try_for_each, convert_data)?;
    }

    let duration = start.elapsed().unwrap().as_secs_f32();
    let stats = progress.stats();
    let duration_read = stats.read.as_secs_f32();
    let duration_write = stats.write.as_secs_f32();
    let duration_read_write = duration_read + duration_write;
    let duration_read = duration_read * duration / duration_read_write;
    let duration_write = duration_write * duration / duration_read_write;

    Ok((
        duration,
        duration_read,
        duration_write,
        bytes_decoded.into_inner().unwrap(),
    ))
}

/// Convert an arrays fill value to a new data type
fn convert_fill_value(
    data_type_in: &DataType,
    fill_value_in: &FillValue,
    data_type_out: &DataType,
) -> FillValue {
    macro_rules! convert {
        ( $t_in:ty, $t_out:ty) => {{
            let input_fill_value =
                <$t_in>::from_ne_bytes(fill_value_in.as_ne_bytes().try_into().unwrap());
            use num_traits::AsPrimitive;
            let output_fill_value: $t_out = input_fill_value.as_();
            FillValue::from(output_fill_value)
        }};
    }
    macro_rules! apply_inner {
        ( $type_in:ty, [$( ( $dt_type_out:ty, $type_out:ty ) ),* ]) => {
            {
                let dt_id = data_type_out.identifier();
                $(if dt_id == <$dt_type_out>::IDENTIFIER { convert!($type_in, $type_out) } else)*
                { panic!("Unsupported output data type: {}", data_type_out.identifier()) }
            }
        };
    }
    macro_rules! apply_outer {
        ([$( ( $dt_type_in:ty, $type_in:ty ) ),* ]) => {
            {
                let dt_id = data_type_in.identifier();
                $(if dt_id == <$dt_type_in>::IDENTIFIER {
                    apply_inner!($type_in, [
                        (BoolDataType, u8),
                        (Int8DataType, i8),
                        (Int16DataType, i16),
                        (Int32DataType, i32),
                        (Int64DataType, i64),
                        (UInt8DataType, u8),
                        (UInt16DataType, u16),
                        (UInt32DataType, u32),
                        (UInt64DataType, u64),
                        (BFloat16DataType, half::bf16),
                        (Float16DataType, half::f16),
                        (Float32DataType, f32),
                        (Float64DataType, f64)
                    ])
                } else)*
                { panic!("Unsupported input data type: {}", data_type_in.identifier()) }
            }
        };
    }
    apply_outer!([
        (BoolDataType, u8),
        (Int8DataType, i8),
        (Int16DataType, i16),
        (Int32DataType, i32),
        (Int64DataType, i64),
        (UInt8DataType, u8),
        (UInt16DataType, u16),
        (UInt32DataType, u32),
        (UInt64DataType, u64),
        (BFloat16DataType, half::bf16),
        (Float16DataType, half::f16),
        (Float32DataType, f32),
        (Float64DataType, f64)
    ])
}

pub fn calculate_chunk_and_codec_concurrency(
    concurrent_target: usize,
    concurrent_chunks: Option<usize>,
    codecs: &CodecChain,
    num_chunks: usize,
    chunk_shape: &ChunkShape,
    data_type: &DataType,
) -> (usize, usize) {
    zarrs::array::concurrency::calc_concurrency_outer_inner(
        concurrent_target,
        &if let Some(concurrent_chunks) = concurrent_chunks {
            let concurrent_chunks = std::cmp::min(num_chunks, concurrent_chunks);
            RecommendedConcurrency::new(concurrent_chunks..concurrent_chunks)
        } else {
            let concurrent_chunks =
                std::cmp::min(num_chunks, global_config().chunk_concurrent_minimum());
            RecommendedConcurrency::new_minimum(concurrent_chunks)
        },
        &codecs
            .recommended_concurrency(chunk_shape, data_type)
            .unwrap(),
    )
}
