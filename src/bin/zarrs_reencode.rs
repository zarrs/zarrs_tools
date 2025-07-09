use core::f32;
use std::num::NonZeroU64;
use std::sync::Arc;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use zarrs::array::Array;
use zarrs::filesystem::{FilesystemStore, FilesystemStoreOptions};
use zarrs::storage::{
    ListableStorageTraits, ReadableListableStorage, ReadableWritableListableStorageTraits,
    ReadableWritableStorageTraits, StorePrefix, WritableStorageTraits,
};
use zarrs_tools::{
    do_reencode, get_array_builder_reencode,
    progress::{ProgressCallback, ProgressStats},
    CacheSize, ZarrReencodingArgs,
};

#[cfg(feature = "async")]
use zarrs::storage::{
    storage_adapter::async_to_sync::{AsyncToSyncBlockOn, AsyncToSyncStorageAdapter},
    AsyncReadableListableStorage,
};

#[cfg(feature = "async")]
use zarrs_opendal::AsyncOpendalStore;

/// Reencode a Zarr array.
#[derive(Parser, Debug)]
#[command(author, version=zarrs_tools::ZARRS_TOOLS_VERSION_WITH_ZARRS)]
struct Args {
    #[command(flatten)]
    encoding: ZarrReencodingArgs,

    /// The zarr array input path or URL.
    path_in: String,

    /// The zarr array output directory.
    path_out: String,

    /// Number of concurrent chunks.
    #[arg(long)]
    concurrent_chunks: Option<usize>,

    /// Ignore checksums.
    ///
    /// If set, checksum validation in codecs (e.g. crc32c) is skipped.
    #[arg(long, default_value_t = false)]
    ignore_checksums: bool,

    /// Validate written data.
    #[arg(long, default_value_t = false)]
    validate: bool,

    /// Print verbose information, such as the array header.
    #[arg(long, short, default_value_t = false)]
    verbose: bool,

    /// An optional chunk cache size (in bytes).
    #[arg(long)]
    cache_size: Option<u64>,

    /// An optional chunk cache size (in chunks).
    #[arg(long)]
    cache_chunks: Option<u64>,

    /// An optional per-thread chunk cache size (in bytes).
    #[arg(long)]
    cache_size_thread: Option<u64>,

    /// An optional per-thread chunk cache size (in chunks).
    #[arg(long)]
    cache_chunks_thread: Option<u64>,

    /// Write shape (optional). A comma separated list of the write size along each array dimension.
    ///
    /// Use this parameter to incrementally write shards in batches of chunks of the specified write shape.
    /// The write shape defaults to the shard shape for sharded arrays.
    /// This parameter is ignored for unsharded arrays (the write shape is the chunk shape).
    ///
    /// Prefer to set the write shape to an integer multiple of the chunk shape to avoid unnecessary reads.
    ///
    #[arg(long, verbatim_doc_comment, value_delimiter = ',')]
    write_shape: Option<Vec<NonZeroU64>>,
}

fn bar_style_run() -> ProgressStyle {
    ProgressStyle::with_template(
        "[{elapsed_precise}/{duration_precise}] {bar:40.black/bold} {pos}/{len} ({percent}%) {prefix} {msg}",
    )
    .unwrap_or(ProgressStyle::default_bar())
}

fn bar_style_finish() -> ProgressStyle {
    ProgressStyle::with_template("[{elapsed_precise}/{elapsed_precise}] {prefix} {msg}")
        .unwrap_or(ProgressStyle::default_bar())
}

fn progress_callback(stats: ProgressStats, bar: &ProgressBar) {
    bar.set_length(stats.num_steps as u64);
    bar.set_position(stats.step as u64);
    if stats.process_steps.is_empty() {
        bar.set_message(format!(
            "rw:{:.2}/{:.2} p:{:.2}",
            stats.read.as_secs_f32(),
            stats.write.as_secs_f32(),
            stats.process.as_secs_f32(),
        ));
    } else {
        bar.set_message(format!(
            "rw:{:.2}/{:.2} p:{:.2} {:.2?}",
            stats.read.as_secs_f32(),
            stats.write.as_secs_f32(),
            stats.process.as_secs_f32(),
            stats
                .process_steps
                .iter()
                .map(|t| t.as_secs_f32())
                .collect::<Vec<_>>(),
        ));
    }
}

#[cfg(feature = "async")]
struct TokioBlockOn(tokio::runtime::Runtime);

#[cfg(feature = "async")]
impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        self.0.block_on(future)
    }
}

fn get_storage(path: &str) -> anyhow::Result<ReadableListableStorage> {
    if path.starts_with("http://") || path.starts_with("https://") {
        #[cfg(feature = "async")]
        {
            let builder = opendal::services::Http::default().endpoint(path);
            let operator = opendal::Operator::new(builder)?.finish();
            let storage: AsyncReadableListableStorage = Arc::new(AsyncOpendalStore::new(operator));
            let block_on = TokioBlockOn(tokio::runtime::Runtime::new()?);
            Ok(Arc::new(AsyncToSyncStorageAdapter::new(storage, block_on)))
        }
        #[cfg(not(feature = "async"))]
        anyhow::bail!("zarrs_tools has not been compiled with the async feature for HTTP stores")
    // } else if path.starts_with("s3://") {
    //     let endpoint = "";
    //     let bucket = "";
    //     let root = "";
    //     let region = "auto";
    //     let builder = opendal::services::S3::default()
    //         .endpoint(&endpoint)
    //         .region(&region)
    //         .root(path)
    //         .allow_anonymous()
    //         .bucket(&bucket);
    //     let operator = opendal::Operator::new(builder)?.finish();
    //     Arc::new(AsyncOpendalStore::new(operator))
    } else {
        Ok(Arc::new(FilesystemStore::new_with_options(
            path,
            FilesystemStoreOptions::default().direct_io(true).clone(),
        )?))
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    zarrs::config::global_config_mut().set_validate_checksums(!args.ignore_checksums);

    let storage_in = get_storage(&args.path_in)?;
    let array_in = zarrs::array::Array::open(storage_in.clone().readable(), "/").unwrap();
    if args.verbose {
        println!(
            "{}",
            serde_json::to_string_pretty(&array_in.metadata()).unwrap()
        );
    }

    let bar = ProgressBar::new(0);
    bar.set_style(bar_style_run());
    let progress_callback = |stats: ProgressStats| progress_callback(stats, &bar);
    let progress_callback = ProgressCallback::new(&progress_callback);

    let storage_out: Arc<dyn ReadableWritableListableStorageTraits> =
        Arc::new(FilesystemStore::new(args.path_out.clone()).unwrap());
    storage_out.erase_prefix(&StorePrefix::root()).unwrap();
    let builder = get_array_builder_reencode(&args.encoding, &array_in, None);
    let array_out = builder.build(storage_out.clone(), "/").unwrap();
    array_out.store_metadata().unwrap();

    let cache_size = if let Some(cache_size_thread) = args.cache_size_thread {
        CacheSize::SizePerThread(cache_size_thread)
    } else if let Some(cache_size) = args.cache_size {
        CacheSize::SizeTotal(cache_size)
    } else if let Some(cache_chunks_thread) = args.cache_chunks_thread {
        CacheSize::ChunksPerThread(cache_chunks_thread)
    } else if let Some(cache_chunks) = args.cache_chunks {
        CacheSize::ChunksTotal(cache_chunks)
    } else {
        CacheSize::None
    };

    let (duration, duration_read, duration_write, bytes_decoded) = do_reencode(
        &array_in,
        &array_out,
        args.validate,
        args.concurrent_chunks,
        &progress_callback,
        cache_size,
        args.write_shape,
    )?;
    bar.set_style(bar_style_finish());
    bar.finish_and_clear();
    let size_in = storage_in
        .size()
        .map(|size| size as f32)
        .unwrap_or(f32::NAN);
    let size_out = storage_out.size().unwrap_or_default() as f32;
    let bytes_decoded = bytes_decoded as f32;
    println!(
        "Reencode {} to {}\n\tread:  ~{:.2}ms @ {:.2}GB/s\n\twrite: ~{:.2}ms @ {:.2}GB/s\n\ttotal: {:.2}ms\n\tsize:  {:.2}MB to {:.2}MB ({:.2}MB uncompressed)",
        args.path_in,
        args.path_out,
        duration_read * 1e3, // ms
        size_in / 1e9 / duration_read, // GB/s
        duration_write * 1e3, // ms
        size_out / 1e9 / duration_write, // GB/s
        duration * 1e3, // ms
        size_in / 1e6, // MB
        size_out / 1e6, // MB
        bytes_decoded / 1e6, // MB
    );
    Ok(())
}
