# zarrs_reencode

Reencode/rechunk a Zarr V2/V3 to a Zarr v3 array.

## Installation
`zarrs_reencode` packaged by default with `zarrs_tools` and requires no extra features.

### Prebuilt Binaries
```bash
# Requires cargo-binstall https://github.com/cargo-bins/cargo-binstall
cargo binstall zarrs_tools
```

### From Source
```bash
cargo install zarrs_tools
```

## Usage
<details>
<summary>zarrs_reencode --help</summary>

```text
Reencode a Zarr array

Usage: zarrs_reencode [OPTIONS] <PATH_IN> <PATH_OUT>

Arguments:
  <PATH_IN>
          The zarr array input path or URL

  <PATH_OUT>
          The zarr array output directory

Options:
  -d, --data-type <DATA_TYPE>
          The data type as a string
          
          Changing a data type will use standard rust numeric casting. See <https://doc.rust-lang.org/reference/expressions/operator-expr.html#r-expr.as.numeric>.
          - Casting from a larger integer to a smaller integer will truncate,
          - Casting from a float to an integer will round the float towards zero
          - Casting from an integer to float will produce the closest possible float
          
          Valid data types:
            - bool
            - int8, int16, int32, int64
            - uint8, uint16, uint32, uint64
            - float16, float32, float64, bfloat16
            - complex64, complex 128
            - r* (raw bits, where * is a multiple of 8)

  -f, --fill-value <FILL_VALUE>
          Fill value. See <https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#fill-value>
          
          The fill value must be compatible with the data type.
          
          Examples:
            int/uint: 0 100 -100
            float: 0.0 "NaN" "Infinity" "-Infinity"
            r*: "[0, 255]"

      --separator <SEPARATOR>
          The chunk key encoding separator. Either . or /

  -c, --chunk-shape <CHUNK_SHAPE>
          Chunk shape. A comma separated list of the chunk size along each array dimension.
          
          If any dimension has size zero, it will be set to match the array shape.

  -s, --shard-shape <SHARD_SHAPE>
          Shard shape. A comma separated list of the shard size along each array dimension.
          
          If specified, the array is encoded using the sharding codec.
          If any dimension has size zero, it will be set to match the array shape.

      --ignore-input-sharding
          If true, the sharding of the input will be ignored.

      --array-to-array-codecs <ARRAY_TO_ARRAY_CODECS>
          Array to array codecs.
          
          JSON holding an array of array to array codec metadata.
          
          Examples:
            '[ { "name": "transpose", "configuration": { "order": [0, 2, 1] } } ]'
            '[ { "name": "bitround", "configuration": { "keepbits": 9 } } ]'

      --array-to-bytes-codec <ARRAY_TO_BYTES_CODEC>
          Array to bytes codec.
          
          JSON holding array to bytes codec metadata.
          
          Examples:
            '{ "name": "bytes", "configuration": { "endian": "little" } }'
            '{ "name": "pcodec", "configuration": { "level": 12 } }'
            '{ "name": "zfp", "configuration": { "mode": "fixedprecision", "precision": 19 } }'

      --bytes-to-bytes-codecs <BYTES_TO_BYTES_CODECS>
          Bytes to bytes codecs.
          
          JSON holding an array bytes to bytes codec configurations.
          
          Examples:
            '[ { "name": "blosc", "configuration": { "cname": "blosclz", "clevel": 9, "shuffle": "bitshuffle", "typesize": 2, "blocksize": 0 } } ]'
            '[ { "name": "bz2", "configuration": { "level": 9 } } ]'
            '[ { "name": "crc32c" } ]'
            '[ { "name": "gzip", "configuration": { "level": 9 } } ]'
            '[ { "name": "zstd", "configuration": { "level": 22, "checksum": false } } ]'

      --dimension-names <DIMENSION_NAMES>
          Dimension names (optional). Comma separated.

      --attributes <ATTRIBUTES>
          Attributes (optional).
          
          JSON holding array attributes.

      --attributes-append <ATTRIBUTES_APPEND>
          Attributes to append (optional).
          
          JSON holding array attributes.

      --concurrent-chunks <CONCURRENT_CHUNKS>
          Number of concurrent chunks

      --ignore-checksums
          Ignore checksums.
          
          If set, checksum validation in codecs (e.g. crc32c) is skipped.

      --validate
          Validate written data

  -v, --verbose
          Print verbose information, such as the array header

      --cache-size <CACHE_SIZE>
          An optional chunk cache size (in bytes)

      --cache-chunks <CACHE_CHUNKS>
          An optional chunk cache size (in chunks)

      --cache-size-thread <CACHE_SIZE_THREAD>
          An optional per-thread chunk cache size (in bytes)

      --cache-chunks-thread <CACHE_CHUNKS_THREAD>
          An optional per-thread chunk cache size (in chunks)

      --write-shape <WRITE_SHAPE>
          Write shape (optional). A comma separated list of the write size along each array dimension.
          
          Use this parameter to incrementally write shards in batches of chunks of the specified write shape.
          The write shape defaults to the shard shape for sharded arrays.
          This parameter is ignored for unsharded arrays (the write shape is the chunk shape).
          
          Prefer to set the write shape to an integer multiple of the chunk shape to avoid unnecessary reads.

      --direct-io
          Enable direct I/O for filesystem operations.
          
          If set, filesystem operations will use direct I/O bypassing the page cache.

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

</details>

## Example
Reencode `array.zarr` (`uint16`) with:
 - a chunk shape of [32, 32, 32],
 - a shard shape of [128, 128, 0]
   - the last dimension of the shard shape will match the array shape to the nearest multiple of the chunk shape
 - level 9 blosclz compression with bitshuffling
 - an input chunk cache with a size of 1GB

```bash
zarrs_reencode \
--cache-size 1000000000 \
--chunk-shape 32,32,32 \
--shard-shape 128,128,0 \
--bytes-to-bytes-codecs '[ { "name": "blosc", "configuration": { "cname": "blosclz", "clevel": 9, "shuffle": "bitshuffle", "typesize": 2, "blocksize": 0 } } ]' \
array.zarr array_reencode.zarr
```
