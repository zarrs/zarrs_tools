use clap::Parser;
use indicatif::{DecimalBytes, ProgressBar, ProgressStyle};
use rayon::{iter::ParallelIterator, prelude::IntoParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use std::{io::Read, path::PathBuf, sync::atomic::AtomicUsize};
use zarrs_tools::{get_array_builder, ZarrEncodingArgs};

use zarrs::{
    array::{
        codec::{ArrayCodecTraits, CodecOptions},
        concurrency::RecommendedConcurrency,
        data_type, Array, ArraySubset, DataTypeExt, Endianness, NamedDataType,
    },
    config::global_config,
    filesystem::{FilesystemStore, FilesystemStoreOptions},
    metadata::v3::MetadataV3,
    plugin::ExtensionIdentifier,
    storage::ListableStorageTraits,
};

use zarrs::array::DataType;

fn reverse_endianness(v: &mut [u8], data_type: &DataType) -> anyhow::Result<()> {
    match data_type.identifier() {
        // 1-byte types: no swap needed
        data_type::BoolDataType::IDENTIFIER
        | data_type::Int8DataType::IDENTIFIER
        | data_type::UInt8DataType::IDENTIFIER
        | data_type::RawBitsDataType::IDENTIFIER => {}

        // 2-byte types
        data_type::Int16DataType::IDENTIFIER
        | data_type::UInt16DataType::IDENTIFIER
        | data_type::Float16DataType::IDENTIFIER
        | data_type::BFloat16DataType::IDENTIFIER => {
            let (chunks, _remainder) = v.as_chunks_mut::<2>();
            for chunk in chunks {
                let bytes = u16::from_ne_bytes(*chunk);
                *chunk = bytes.swap_bytes().to_ne_bytes();
            }
        }

        // 4-byte types
        data_type::Int32DataType::IDENTIFIER
        | data_type::UInt32DataType::IDENTIFIER
        | data_type::Float32DataType::IDENTIFIER
        | data_type::Complex64DataType::IDENTIFIER => {
            let (chunks, _remainder) = v.as_chunks_mut::<4>();
            for chunk in chunks {
                let bytes = u32::from_ne_bytes(*chunk);
                *chunk = bytes.swap_bytes().to_ne_bytes();
            }
        }

        // 8-byte types
        data_type::Int64DataType::IDENTIFIER
        | data_type::UInt64DataType::IDENTIFIER
        | data_type::Float64DataType::IDENTIFIER
        | data_type::Complex128DataType::IDENTIFIER => {
            let (chunks, _remainder) = v.as_chunks_mut::<8>();
            for chunk in chunks {
                let bytes = u64::from_ne_bytes(*chunk);
                *chunk = bytes.swap_bytes().to_ne_bytes();
            }
        }
        id => anyhow::bail!("unsupported data type {id} for reverse_endianness"),
    };
    Ok(())
}

/// Convert an N-dimensional binary array from standard input to a Zarr V3 array.
#[derive(Parser)]
#[command(author, version=zarrs_tools::ZARRS_TOOLS_VERSION_WITH_ZARRS)]
#[allow(rustdoc::bare_urls)]
struct Cli {
    /// The endianness of the binary data. If unspecified, it is assumed to match the host endianness.
    #[arg(long, value_parser = parse_endianness)]
    endianness: Option<Endianness>,

    #[command(flatten)]
    encoding: ZarrEncodingArgs,

    /// Number of concurrent chunk writers.
    #[arg(long)]
    concurrent_chunks: Option<usize>,

    /// Zarr data type. See https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#id11
    ///
    /// Examples:
    ///   bool
    ///   int8 int16 int32 int64
    ///   uint8 uint16 uint32 uint64
    ///   float32 float64 float16 bfloat16
    ///   complex64 complex128
    ///   r8 r16 r24 r32 r64 (r* where * is a multiple of 8)
    #[arg(short, long, verbatim_doc_comment, value_parser = parse_data_type)]
    data_type: MetadataV3,

    /// Array shape. A comma separated list of the sizes of each array dimension.
    #[arg(short, long, required = true, value_delimiter = ',')]
    array_shape: Vec<u64>,

    /// Dimension names. A comma separated list of the names of each array dimension.
    #[arg(long, value_delimiter = ',')]
    dimension_names: Option<Vec<String>>,

    /// The output directory for the zarr array.
    out: PathBuf,

    /// Enable direct I/O for filesystem operations.
    ///
    /// If set, filesystem operations will use direct I/O bypassing the page cache.
    #[arg(long, default_value_t = false)]
    direct_io: bool,
    // /// The path to a binary file or a directory of binary files.
    // #[arg(short, long, num_args = 1..)]
    // file: Vec<PathBuf>,
}

fn parse_data_type(data_type: &str) -> std::io::Result<MetadataV3> {
    serde_json::from_value(serde_json::Value::String(data_type.to_string()))
        .map_err(|err| std::io::Error::other(err.to_string()))
}

fn parse_endianness(endianness: &str) -> std::io::Result<Endianness> {
    if endianness == "little" {
        Ok(Endianness::Little)
    } else if endianness == "big" {
        Ok(Endianness::Big)
    } else {
        Err(std::io::Error::other("Endianness must be little or big"))
    }
}

fn stdin_to_array(
    array: &Array<FilesystemStore>,
    endianness: Option<Endianness>,
    concurrent_chunks: Option<usize>,
) -> anyhow::Result<usize> {
    let data_type_size = array
        .data_type()
        .fixed_size()
        .expect("data type should be fixed size");
    let dimensionality = array.chunk_grid().dimensionality();
    let array_shape = array.shape();
    let array_shape_n = *array_shape.first().unwrap();
    let chunk_shape = array
        .chunk_grid()
        .chunk_shape(&vec![0; array.chunk_grid().dimensionality()])
        .unwrap()
        .expect("lowest indices should have a chunk shape");
    let block_shape_n = *chunk_shape.first().unwrap();
    let n_blocks = array_shape_n.div_ceil(block_shape_n.get());

    let bar = ProgressBar::new(n_blocks);
    let style =
        ProgressStyle::with_template("[{elapsed_precise}] [{bar}] ({pos}/{len}, ETA {eta})")
            .unwrap();
    bar.set_style(style);

    let n_blocks = usize::try_from(n_blocks).unwrap();

    let chunk_shape = array
        .chunk_shape(&vec![0; array.chunk_grid().dimensionality()])
        .unwrap();
    let concurrent_target = std::thread::available_parallelism().unwrap().get();
    let (concurrent_chunks, codec_concurrent_target) =
        zarrs::array::concurrency::calc_concurrency_outer_inner(
            concurrent_target,
            &if let Some(concurrent_chunks) = concurrent_chunks {
                let concurrent_chunks = std::cmp::min(n_blocks, concurrent_chunks);
                RecommendedConcurrency::new(concurrent_chunks..concurrent_chunks)
            } else {
                let concurrent_chunks =
                    std::cmp::min(n_blocks, global_config().chunk_concurrent_minimum());
                RecommendedConcurrency::new_minimum(concurrent_chunks)
            },
            &array
                .codecs()
                .recommended_concurrency(&chunk_shape, array.data_type())
                .unwrap(),
        );
    let codec_options = CodecOptions::default().with_concurrent_target(codec_concurrent_target);

    #[allow(clippy::mutex_integer)]
    let idxm = std::sync::Mutex::new(0u64);
    let bytes_read: AtomicUsize = 0.into();
    let op = |_| {
        #[allow(clippy::mutex_integer)]
        let mut idxm = idxm.lock().unwrap();
        let idx = *idxm;
        bar.set_position(idx);
        *idxm += 1;

        let start = idx * block_shape_n.get();
        let end = std::cmp::min((idx + 1) * block_shape_n.get(), array_shape_n);

        let mut startn: Vec<u64> = vec![start];
        startn.resize(dimensionality, 0);
        let mut endn = vec![end];
        endn.extend(array_shape.iter().skip(1));
        let array_subset = ArraySubset::new_with_start_end_exc(startn, endn).unwrap();

        let bytes_len =
            usize::try_from(array_subset.num_elements() * data_type_size as u64).unwrap();
        let mut subset_bytes = vec![0; bytes_len];
        std::io::stdin().read_exact(&mut subset_bytes).unwrap();
        bytes_read.fetch_add(bytes_len, std::sync::atomic::Ordering::Relaxed);

        drop(idxm);

        if let Some(endianness) = endianness {
            if !endianness.is_native() {
                reverse_endianness(&mut subset_bytes, array.data_type())?;
            }
        }

        array
            .store_array_subset_opt(&array_subset, subset_bytes, &codec_options)
            .unwrap();
        Ok::<_, anyhow::Error>(())
    };
    iter_concurrent_limit!(concurrent_chunks, 0..n_blocks, try_for_each, op)?;
    Ok(bytes_read.load(std::sync::atomic::Ordering::Relaxed))
}

fn main() -> anyhow::Result<()> {
    // Parse and validate arguments
    let cli = Cli::parse();

    // Get data type
    let data_type = NamedDataType::try_from(&cli.data_type).unwrap();

    // Create storage
    let path_out = cli.out.as_path();
    let mut options = FilesystemStoreOptions::default();
    options.direct_io(cli.direct_io);
    let store = std::sync::Arc::new(FilesystemStore::new_with_options(path_out, options).unwrap());

    // Create array
    let dimension_names = cli
        .dimension_names
        .map(|dimension_names| dimension_names.into_iter().map(Some).collect());
    let array_builder =
        get_array_builder(&cli.encoding, &cli.array_shape, data_type, dimension_names);
    let array = array_builder.build(store.clone(), "/").unwrap();

    // Store array metadata
    array.store_metadata().unwrap();

    // Read stdin to the array and write chunks/shards
    let start = std::time::Instant::now();
    let bytes_read: usize = stdin_to_array(&array, cli.endianness, cli.concurrent_chunks)?;
    let duration_s = start.elapsed().as_secs_f32();

    // Output stats
    let duration_ms = duration_s * 1.0e3;
    let size_out = store.size().unwrap();
    // let space_saving = 100.0 * (1.0 - (size_out as f32 / bytes_read as f32));
    let relative_size = 100.0 * (size_out as f32 / bytes_read as f32);
    println!("Output {path_out:?} in {duration_ms:.2}ms ({gbs:.2} GB/s) [{bytes_read} -> {size_out} ({relative_size:.2}%)]",
    gbs = (bytes_read as f32 * 1e-9) / duration_s,
        bytes_read = DecimalBytes(bytes_read as u64),
        size_out = DecimalBytes(size_out),
    );
    Ok(())
}
