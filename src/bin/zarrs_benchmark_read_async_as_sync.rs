use std::{
    sync::{Arc, Mutex},
    time::SystemTime,
};

use clap::Parser;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use zarrs::{
    array::{
        codec::CodecOptionsBuilder, ArrayBytes, ArrayIndicesTinyVec, ArrayShardedExt,
        ArrayShardedReadableExt, ArrayShardedReadableExtCache, ChunkRepresentation,
    },
    array_subset::ArraySubset,
    storage::{
        storage_adapter::async_to_sync::{AsyncToSyncBlockOn, AsyncToSyncStorageAdapter},
        AsyncReadableStorage, ReadableStorage,
    },
};
use zarrs_tools::calculate_chunk_and_codec_concurrency;

/// Benchmark zarrs read throughput with the sync API.
#[derive(Parser, Debug)]
#[command(author, version=zarrs_tools::ZARRS_TOOLS_VERSION_WITH_ZARRS)]
struct Args {
    /// The path or URL of a zarr array.
    path: String,

    /// Number of concurrent chunks.
    #[arg(long)]
    concurrent_chunks: Option<usize>,

    /// Read the entire array in one operation.
    ///
    /// If set, `concurrent_chunks` is ignored.
    #[arg(long, default_value_t = false)]
    read_all: bool,

    /// Read inner-chunk-by-inner-chunk for sharded arrays.
    ///
    /// Ignored for unsharded arrays.
    #[arg(long, default_value_t = false)]
    inner_chunks: bool,

    /// Ignore checksums.
    ///
    /// If set, checksum validation in codecs (e.g. crc32c) is skipped.
    #[arg(long, default_value_t = false)]
    ignore_checksums: bool,
}

struct TokioBlockOn(tokio::runtime::Runtime);

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        self.0.block_on(future)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let storage: AsyncReadableStorage = if args.path.starts_with("http") {
        // opendal
        let builder = opendal::services::Http::default().endpoint(&args.path);
        let operator = opendal::Operator::new(builder)?.finish();
        Arc::new(zarrs_opendal::AsyncOpendalStore::new(operator))

        // object_store
        // let options = object_store::ClientOptions::new().with_allow_http(true);
        // let store = object_store::http::HttpBuilder::new()
        //     .with_url(&args.path)
        //     .with_client_options(options)
        //     .build()?;
        // Arc::new(store::AsyncObjectStore::new(store))
    } else {
        // opendal
        let builder = opendal::services::Fs::default().root(&args.path);
        let operator = opendal::Operator::new(builder)?.finish();
        Arc::new(zarrs_opendal::AsyncOpendalStore::new(operator))

        // object_store
        // let store = object_store::local::LocalFileSystem::new_with_prefix(&args.path)?;
        // Arc::new(store::AsyncObjectStore::new(store))
    };

    let block_on = TokioBlockOn(tokio::runtime::Runtime::new()?);
    let storage: ReadableStorage = Arc::new(AsyncToSyncStorageAdapter::new(storage, block_on));

    let array = zarrs::array::Array::open(storage.clone(), "/")?;
    // println!("{:#?}", array.metadata());

    zarrs::config::global_config_mut().set_validate_checksums(!args.ignore_checksums);

    let chunks = ArraySubset::new_with_shape(array.chunk_grid_shape().clone());
    let chunk_indices = chunks.indices();

    let concurrent_target = std::thread::available_parallelism().unwrap().get();
    let start = SystemTime::now();
    let bytes_decoded = Mutex::new(0);
    if args.read_all {
        *bytes_decoded.lock().unwrap() += array
            .retrieve_array_subset::<ArrayBytes>(&array.subset_all())?
            .size();
    } else if let (Some(inner_chunk_shape), true) =
        (array.effective_inner_chunk_shape(), args.inner_chunks)
    {
        let inner_chunks = ArraySubset::new_with_shape(array.inner_chunk_grid_shape().clone());
        let inner_chunk_indices = inner_chunks.indices();
        let inner_chunk_representation = ChunkRepresentation::new(
            inner_chunk_shape.to_vec(),
            array.data_type().clone(),
            array.fill_value().clone(),
        )?;
        let (chunks_concurrent_limit, codec_concurrent_target) =
            calculate_chunk_and_codec_concurrency(
                concurrent_target,
                args.concurrent_chunks,
                &array.codecs(),
                inner_chunks.num_elements_usize(),
                &inner_chunk_representation,
            );
        let codec_options = CodecOptionsBuilder::new()
            .concurrent_target(codec_concurrent_target)
            .build();
        let shard_index_cache = ArrayShardedReadableExtCache::new(&array);

        rayon_iter_concurrent_limit::iter_concurrent_limit!(
            chunks_concurrent_limit,
            inner_chunk_indices,
            for_each,
            |inner_chunk_indices: ArrayIndicesTinyVec| {
                // println!("Chunk/shard: {:?}", chunk_indices);
                let bytes: ArrayBytes = array
                    .retrieve_inner_chunk_opt(
                        &shard_index_cache,
                        &inner_chunk_indices,
                        &codec_options,
                    )
                    .unwrap();
                *bytes_decoded.lock().unwrap() += bytes.size();
            }
        );
    } else {
        let chunk_representation =
            array.chunk_array_representation(&vec![0; array.chunk_grid().dimensionality()])?;
        let (chunks_concurrent_limit, codec_concurrent_target) =
            calculate_chunk_and_codec_concurrency(
                concurrent_target,
                args.concurrent_chunks,
                &array.codecs(),
                chunks.num_elements_usize(),
                &chunk_representation,
            );
        let codec_options = CodecOptionsBuilder::new()
            .concurrent_target(codec_concurrent_target)
            .build();

        // println!("chunks_concurrent_limit {chunks_concurrent_limit:?} codec_concurrent_target {codec_concurrent_target:?}");
        // NOTE: Could init memory per split with for_each_init and then reuse it with retrieve_chunk_into_array_view_opt.
        //       But that might be cheating against tensorstore.
        rayon_iter_concurrent_limit::iter_concurrent_limit!(
            chunks_concurrent_limit,
            chunk_indices,
            for_each,
            |chunk_indices: ArrayIndicesTinyVec| {
                // println!("Chunk/shard: {:?}", chunk_indices);
                let bytes: ArrayBytes = array
                    .retrieve_chunk_opt(&chunk_indices, &codec_options)
                    .unwrap();
                *bytes_decoded.lock().unwrap() += bytes.size();
            }
        );
    }
    let bytes_decoded = bytes_decoded.into_inner()?;
    let duration = SystemTime::now().duration_since(start)?.as_secs_f32();
    println!(
        "Decoded {} in {:.2}ms ({:.2}MB decoded @ {:.2}GB/s)",
        args.path,
        duration * 1e3,
        bytes_decoded as f32 / 1e6,
        (/* GB */bytes_decoded as f32 * 1e-9) / duration,
    );
    Ok(())
}
