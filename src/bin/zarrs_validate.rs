use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use zarrs::array::codec::CodecOptionsBuilder;
use zarrs::array_subset::ArraySubset;
use zarrs::filesystem::{FilesystemStore, FilesystemStoreOptions};
use zarrs::storage::{
    storage_adapter::async_to_sync::{AsyncToSyncBlockOn, AsyncToSyncStorageAdapter},
    AsyncReadableListableStorage, ReadableListableStorage,
};
use zarrs_opendal::AsyncOpendalStore;
use zarrs_tools::calculate_chunk_and_codec_concurrency;

/// Compare the data in two Zarr arrays.
///
/// Equality of the arrays is determined by comparing the shape, data type, and data.
///
/// Differences in encoding (e.g codecs, chunk key encoding) and attributes are ignored.
#[derive(Parser, Debug)]
#[command(author, version=zarrs_tools::ZARRS_TOOLS_VERSION_WITH_ZARRS)]
struct Args {
    /// The path to the first zarr array.
    first: String,

    /// The path to the second zarr array.
    second: String,

    /// Number of concurrent chunks to compare.
    #[arg(long)]
    concurrent_chunks: Option<usize>,
}

fn bar_style_run() -> ProgressStyle {
    ProgressStyle::with_template(
        "[{elapsed_precise}/{duration_precise}] {bar:40.black/bold} {pos}/{len} ({percent}%) {prefix} {msg}",
    )
    .unwrap_or(ProgressStyle::default_bar())
}

struct TokioBlockOn(tokio::runtime::Runtime);

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        self.0.block_on(future)
    }
}

fn get_storage(path: &str) -> anyhow::Result<ReadableListableStorage> {
    if path.starts_with("http://") || path.starts_with("https://") {
        let builder = opendal::services::Http::default().endpoint(path);
        let operator = opendal::Operator::new(builder)?.finish();
        let storage: AsyncReadableListableStorage = Arc::new(AsyncOpendalStore::new(operator));
        let block_on = TokioBlockOn(tokio::runtime::Runtime::new()?);
        Ok(Arc::new(AsyncToSyncStorageAdapter::new(storage, block_on)))
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

fn main() {
    match try_main() {
        Ok(success) => println!("{success}"),
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}

fn try_main() -> anyhow::Result<String> {
    let args = Args::parse();

    let storage1 = get_storage(&args.first)?;
    let storage2 = get_storage(&args.second)?;
    let array1 = zarrs::array::Array::open(storage1.clone(), "/").unwrap();
    let array2 = zarrs::array::Array::open(storage2.clone(), "/").unwrap();

    let bar = ProgressBar::new(0);
    bar.set_style(bar_style_run());

    if array1.shape() != array2.shape() {
        anyhow::bail!(
            "Array shapes do not match: {:?} vs {:?}",
            array1.shape(),
            array2.shape()
        );
    } else if array1.data_type() != array2.data_type() {
        anyhow::bail!(
            "Array data types do not match: {} vs {}",
            array1.data_type(),
            array2.data_type()
        );
    }

    let chunks = ArraySubset::new_with_shape(array1.chunk_grid_shape().clone());

    let chunk_representation = array1
        .chunk_array_representation(&vec![0; array1.chunk_grid().dimensionality()])
        .unwrap();

    let concurrent_target = std::thread::available_parallelism().unwrap().get();
    let (chunks_concurrent_limit, codec_concurrent_target) = calculate_chunk_and_codec_concurrency(
        concurrent_target,
        args.concurrent_chunks,
        &array1.codecs(),
        chunks.num_elements_usize(),
        &chunk_representation,
    );
    let codec_options = CodecOptionsBuilder::new()
        .concurrent_target(codec_concurrent_target)
        .build();

    let num_iterations = chunks.num_elements_usize();
    bar.set_length(num_iterations as u64);
    let indices = chunks.indices();
    let step = AtomicU64::new(0);
    iter_concurrent_limit!(
        chunks_concurrent_limit,
        indices,
        try_for_each,
        |chunk_indices: Vec<u64>| {
            let step = step.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            bar.set_position(step);
            let chunk_subset = array1.chunk_subset(&chunk_indices).unwrap();
            let bytes_first = array1.retrieve_chunk_opt(&chunk_indices, &codec_options)?;
            let bytes_second = array2.retrieve_array_subset_opt(&chunk_subset, &codec_options)?;
            if bytes_first == bytes_second {
                Ok(())
            } else {
                anyhow::bail!("Data differs in region: {chunk_subset}");
            }
        }
    )?;
    bar.finish_and_clear();

    Ok(format!("Success: {} and {} match", args.first, args.second))
}
