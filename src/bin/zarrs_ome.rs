use std::{
    hash::Hash,
    num::NonZeroU64,
    path::{Path, PathBuf},
};

use anyhow::anyhow;
use clap::Parser;
use half::{bf16, f16};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use itertools::Itertools;
use num_traits::AsPrimitive;
use ome_zarr_metadata::v0_5::{
    Axis, AxisType, AxisUnit, CoordinateTransform, CoordinateTransformScale,
    CoordinateTransformTranslation, MultiscaleImageDataset, MultiscaleImageMetadata,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use zarrs::{
    array::{
        codec::array_to_bytes::sharding::ShardingCodec, codec::ArrayCodecTraits, data_type, Array,
        ArrayIndicesTinyVec, ArrayMetadata, Element, ElementOwned,
    },
    array_subset::ArraySubset,
    filesystem::{FilesystemStore, FilesystemStoreOptions},
    group::{Group, GroupMetadata, GroupMetadataV3},
    plugin::ExtensionIdentifier,
    storage::{StorePrefix, WritableStorageTraits},
};
use zarrs_tools::{
    filter::{
        filters::{downsample::Downsample, gaussian::Gaussian},
        ArraySubsetOverlap, FilterError, FilterTraits,
    },
    progress::{Progress, ProgressCallback, ProgressStats},
    ZarrReEncodingChangeType, ZarrReencodingArgs,
};

#[derive(clap::ValueEnum, Debug, Clone)]
enum OutputExists {
    /// Erase the output
    Erase,
    /// Overwrite existing files.
    /// Useful if the output includes additional non-zarr files to be preserved.
    /// May fail if changing the encoding.
    Overwrite,
    /// Exit if the output already exists
    Exit,
}

#[allow(non_camel_case_types)]
#[derive(clap::ValueEnum, Debug, Clone)]
enum OMEZarrVersion {
    /// https://ngff.openmicroscopy.org/0.5/
    #[value(name = "0.5")]
    V0_5,
}

impl std::fmt::Display for OMEZarrVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OMEZarrVersion::V0_5 => write!(f, "0.5"),
        }
    }
}

/// Convert a Zarr array to an OME-Zarr multiscales hierarchy.
#[derive(Parser, Debug)]
#[command(author, version=zarrs_tools::ZARRS_TOOLS_VERSION_WITH_ZARRS)]
struct Cli {
    /// The input array path.
    input: PathBuf,
    /// The output group path.
    output: PathBuf,

    // The OME-Zarr version.
    #[arg(long, default_value_t = OMEZarrVersion::V0_5)]
    ome_zarr_version: OMEZarrVersion,

    /// The downsample factor per axis, comma separated.
    ///
    /// Defaults to 2 on each axis.
    #[arg(value_delimiter = ',')]
    downsample_factor: Option<Vec<u64>>,

    /// Maximum number of downsample levels.
    #[arg(long, default_value_t = 10)]
    max_levels: usize,

    /// Physical size per axis, comma separated.
    #[arg(long, value_delimiter = ',')]
    physical_size: Option<Vec<f32>>,

    /// Physical units per axis, comma separated.
    ///
    /// Set to "channel" for a channel axis.
    #[arg(long, value_delimiter = ',')]
    physical_units: Option<Vec<String>>,

    /// OME Zarr dataset name.
    #[arg(long)]
    name: Option<String>,

    /// Set to true for discrete data.
    ///
    /// Performs majority downsampling instead of creating a Gaussian image pyramid or mean downsampling.
    #[arg(long)]
    discrete: bool,

    /// The Gaussian "sigma" to apply when creating a Gaussian image pyramid per axis, comma separated.
    ///
    /// This is typically set to 0.5 times the downsample factor for each axis.
    /// If omitted, then mean downsampling is applied.
    ///
    /// Ignored for discrete data.
    #[arg(long, value_delimiter = ',')]
    gaussian_sigma: Option<Vec<f32>>,

    /// The Gaussian kernel half size per axis, comma separated.
    ///
    /// If omitted, defaults to ceil(3 * sigma).
    ///
    /// Ignored for discrete data or if --gaussian-sigma is not set.
    #[arg(long, value_delimiter = ',')]
    gaussian_kernel_half_size: Option<Vec<u64>>,

    /// Behaviour if the output exists.
    #[arg(long)]
    #[clap(value_enum, default_value_t=OutputExists::Erase)]
    exists: OutputExists,

    /// Attributes (optional).
    ///
    /// JSON holding group attributes.
    #[arg(long)]
    group_attributes: Option<String>,

    #[command(flatten)]
    reencoding: ZarrReencodingArgs,

    /// The maximum number of chunks concurrently processed.
    ///
    /// By default, this is set to the number of CPUs.
    /// Consider reducing this for images with large chunk sizes or on systems with low memory availability.
    #[arg(long)]
    chunk_limit: Option<usize>,

    /// Enable direct I/O for filesystem operations.
    ///
    /// If set, filesystem operations will use direct I/O bypassing the page cache.
    #[arg(long, default_value_t = false)]
    direct_io: bool,
}

fn create_filesystem_store(path: &Path, direct_io: bool) -> anyhow::Result<FilesystemStore> {
    let mut options = FilesystemStoreOptions::default();
    options.direct_io(direct_io);
    FilesystemStore::new_with_options(path, options).map_err(Into::into)
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

fn count_dir(src: impl AsRef<Path>) -> std::io::Result<usize> {
    let mut count = 0;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            count += 1;
        } else {
            count += count_dir(entry.path())?;
        }
    }
    Ok(count)
}

fn copy_dir(
    src: impl AsRef<Path>,
    dst: impl AsRef<Path>,
    progress: &Progress,
) -> std::io::Result<()> {
    std::fs::create_dir_all(&dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            progress.write(|| std::fs::copy(entry.path(), dst.as_ref().join(entry.file_name())))?;
            progress.next();
        } else {
            copy_dir(entry.path(), dst.as_ref().join(entry.file_name()), progress)?;
        }
    }
    Ok(())
}

fn apply_chunk_discrete<T>(
    array_input: &Array<FilesystemStore>,
    array_output: &Array<FilesystemStore>,
    chunk_indices: &[u64],
    downsample_filter: &Downsample,
    progress: &Progress,
) -> Result<(), FilterError>
where
    T: Element + ElementOwned + Copy + Send + Sync + Eq + PartialEq + Hash + AsPrimitive<T>,
{
    let output_subset = array_output.chunk_subset_bounded(chunk_indices).unwrap();
    let downsample_input_subset =
        downsample_filter.input_subset(array_input.shape(), &output_subset);
    let output_chunk = {
        let input_chunk: ndarray::ArrayD<T> =
            progress.read(|| array_input.retrieve_array_subset(&downsample_input_subset))?;
        downsample_filter.apply_ndarray_discrete(input_chunk, progress)
    };
    progress.write(|| array_output.store_array_subset(&output_subset, output_chunk))?;
    Ok(())
}

fn apply_chunk_continuous<T>(
    array_input: &Array<FilesystemStore>,
    array_output: &Array<FilesystemStore>,
    chunk_indices: &[u64],
    downsample_filter: &Downsample,
    progress: &Progress,
) -> Result<(), FilterError>
where
    T: Element + ElementOwned + Copy + Send + Sync + AsPrimitive<f64> + std::iter::Sum,
    f64: AsPrimitive<T>,
{
    let output_subset = array_output.chunk_subset_bounded(chunk_indices).unwrap();
    let downsample_input_subset =
        downsample_filter.input_subset(array_input.shape(), &output_subset);
    let output_chunk = {
        let input_chunk: ndarray::ArrayD<T> =
            progress.read(|| array_input.retrieve_array_subset(&downsample_input_subset))?;
        downsample_filter.apply_ndarray_continuous(input_chunk, progress)
    };
    progress.write(|| array_output.store_array_subset(&output_subset, output_chunk))?;
    Ok(())
}

fn apply_chunk_continuous_gaussian<T>(
    array_input: &Array<FilesystemStore>,
    array_output: &Array<FilesystemStore>,
    chunk_indices: &[u64],
    downsample_filter: &Downsample,
    gaussian_filter: &Gaussian,
    progress: &Progress,
) -> Result<(), FilterError>
where
    T: Element + ElementOwned + Copy + Send + Sync + AsPrimitive<f32> + std::iter::Sum,
    f64: AsPrimitive<T>,
{
    let output_subset = array_output.chunk_subset_bounded(chunk_indices).unwrap();
    let downsample_input_subset =
        downsample_filter.input_subset(array_input.shape(), &output_subset);
    let gaussian_subset_overlap = ArraySubsetOverlap::new(
        array_input.shape(),
        &downsample_input_subset,
        gaussian_filter.kernel_half_size(),
    );
    let gaussian_chunk = {
        let input_chunk: ndarray::ArrayD<T> = progress
            .read(|| array_input.retrieve_array_subset(gaussian_subset_overlap.subset_input()))?;
        progress.process(|| {
            let input_chunk: ndarray::ArrayD<f32> = input_chunk.map(|x| x.as_()); // par?
            let output_chunk = gaussian_filter.apply_ndarray(input_chunk);
            gaussian_subset_overlap.extract_subset(&output_chunk)
        })
    };
    let output_chunk = downsample_filter.apply_ndarray_continuous(gaussian_chunk, progress);
    progress.write(|| array_output.store_array_subset(&output_subset, output_chunk))?;
    Ok(())
}

fn progress_callback(stats: ProgressStats, bar: &ProgressBar) {
    bar.set_length(stats.num_steps as u64);
    bar.set_position(stats.step as u64);
    bar.set_message(format!(
        "rw:{:.2}/{:.2} p:{:.2}",
        stats.read.as_secs_f32(),
        stats.write.as_secs_f32(),
        stats.process.as_secs_f32(),
    ));
}

fn run() -> Result<(), anyhow::Error> {
    // Parse command line arguments
    let cli = Cli::parse();

    println!("Input {:?}", cli.input);

    let start = std::time::Instant::now();

    let store_in = create_filesystem_store(&cli.input, cli.direct_io)?;
    let array_in = Array::open(store_in.into(), "/")?;

    let multi_progress = MultiProgress::new();
    let bars = (0..=cli.max_levels)
        .map(|level| {
            let bar = multi_progress.add(ProgressBar::new(1));
            bar.set_style(bar_style_run());
            if level == 0 {
                bar.set_prefix(format!("0 {:?}", array_in.shape()));
            } else {
                bar.set_prefix(format!("{level}"));
            }
            bar
        })
        .collect_vec();

    let finish_step = |bar: &ProgressBar, path: &Path| {
        bar.set_style(bar_style_finish());
        bar.set_prefix(format!("{} {}", bar.prefix(), path.to_string_lossy()));
        bar.abandon();
    };

    // Create group
    let store = std::sync::Arc::new(create_filesystem_store(&cli.output, cli.direct_io)?);
    let mut group = Group::new_with_metadata(
        store.clone(),
        "/",
        GroupMetadata::V3(GroupMetadataV3::default()),
    )?;
    if let Some(attributes) = &cli.group_attributes {
        let mut group_attributes: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(attributes).expect("Group attributes are invalid.");
        group.attributes_mut().append(&mut group_attributes);
    }

    // Handle an existing output
    match cli.exists {
        OutputExists::Exit => {
            if cli.output.exists() {
                Err(FilterError::Other("Output exists, exiting".to_string()))?;
            }
        }
        OutputExists::Erase => {
            store.erase_prefix(&StorePrefix::root()).unwrap();
        }
        OutputExists::Overwrite => {}
    }

    {
        let bar = bars.first().unwrap();
        bar.reset();

        let output_0_path = cli.output.join("0");
        let progress_callback = |stats: ProgressStats| progress_callback(stats, bar);
        let progress_callback = ProgressCallback::new(&progress_callback);
        if let (ZarrReEncodingChangeType::None, ArrayMetadata::V3(_)) =
            (cli.reencoding.change_type(), array_in.metadata())
        {
            // Copy full res input to output if it is Zarr V3 and does not need any changes
            let dir_count = count_dir(&cli.input)?;
            let progress = Progress::new(dir_count, &progress_callback);
            copy_dir(&cli.input, &output_0_path, &progress)?;
        } else {
            // Reencode the input
            let reencode = zarrs_tools::filter::filters::reencode::Reencode::new(cli.chunk_limit);
            let store_out = create_filesystem_store(&cli.output, cli.direct_io)?;
            let mut array_out = reencode
                .output_array_builder(&array_in, &cli.reencoding)?
                .build(store_out.into(), "/0")?;
            reencode.apply(&array_in, &mut array_out, &progress_callback)?;
            array_out.store_metadata()?;
        }
        finish_step(bar, &output_0_path);
    }

    // Setup attributes
    let store = std::sync::Arc::new(create_filesystem_store(&cli.output, cli.direct_io)?);
    // store.erase_prefix(&StorePrefix::root()).unwrap();
    let mut array0 = Array::open(store.clone(), "/0")?;
    {
        // Move array0 attributes to group
        group.attributes_mut().append(array0.attributes_mut()); // this clears array0 attributes
        group.attributes_mut().remove_entry("_zarrs");
        array0.store_metadata()?;
    }

    // Check inputs
    if let Some(downsample_factor) = &cli.downsample_factor {
        if downsample_factor.len() != array0.dimensionality() {
            return Err(anyhow!(
                "downsample factor {downsample_factor:?} length does not match the array dimensionality {}",
                array0.dimensionality()
            ));
        }
    }
    if let Some(physical_size) = &cli.physical_size {
        if physical_size.len() != array0.dimensionality() {
            return Err(anyhow!(
                "physical size {physical_size:?} length does not match the array dimensionality {}",
                array0.dimensionality()
            ));
        }
    }
    if let Some(physical_units) = &cli.physical_units {
        if physical_units.len() != array0.dimensionality() {
            return Err(anyhow!(
                "physical units {physical_units:?} length does not match the array dimensionality {}",
                array0.dimensionality()
            ));
        }
    }
    if let Some(gaussian_sigma) = &cli.gaussian_sigma {
        if gaussian_sigma.len() != array0.dimensionality() {
            return Err(anyhow!(
                "gaussian sigma {gaussian_sigma:?} length does not match the array dimensionality {}",
                array0.dimensionality()
            ));
        }
    }
    if let Some(gaussian_kernel_half_size) = &cli.gaussian_kernel_half_size {
        if gaussian_kernel_half_size.len() != array0.dimensionality() {
            return Err(anyhow!(
                "gaussian kernel half size {gaussian_kernel_half_size:?} length does not match the array dimensionality {}",
                array0.dimensionality()
            ));
        }
    }

    // Initialise multiscales metadata
    let mut axes: Vec<Axis> = Vec::with_capacity(array0.dimensionality());
    let to_unit = |physical_unit: String| {
        Some(
            serde_json::from_value::<AxisUnit>(serde_json::json!(physical_unit))
                .expect("Not a recognised physical unit"),
        )
    };
    let physical_units = cli
        .physical_units
        .map(|physical_units| physical_units.into_iter().map(to_unit).collect_vec())
        .unwrap_or_else(|| vec![None; array0.dimensionality()]);

    let units_to_axis = |name: String, unit: Option<AxisUnit>| {
        if let Some(unit) = unit {
            match unit {
                AxisUnit::Space(unit) => Axis {
                    name,
                    r#type: Some(AxisType::Space),
                    unit: Some(AxisUnit::Space(unit)),
                },
                AxisUnit::Time(unit) => Axis {
                    name,
                    r#type: Some(AxisType::Time),
                    unit: Some(AxisUnit::Time(unit)),
                },
                AxisUnit::Custom(unit) => {
                    if unit == "channel" {
                        Axis {
                            name,
                            r#type: Some(AxisType::Channel),
                            unit: None,
                        }
                    } else {
                        Axis {
                            name,
                            r#type: None,
                            unit: Some(AxisUnit::Custom(unit)),
                        }
                    }
                }
                _ => unimplemented!("Unsupported axis unit"),
            }
        } else {
            Axis {
                name,
                r#type: None,
                unit: None,
            }
        }
    };

    if let Some(dimension_names) = array0.dimension_names() {
        for (i, (dimension_name, unit)) in
            std::iter::zip(dimension_names.iter(), physical_units).enumerate()
        {
            let axis = units_to_axis(
                dimension_name
                    .as_ref()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| i.to_string()),
                unit,
            );
            axes.push(axis)
        }
    } else {
        for (i, unit) in physical_units.into_iter().enumerate() {
            let axis = units_to_axis(i.to_string(), unit);
            axes.push(axis)
        }
    }

    let base_transform = cli.physical_size.map(|physical_size| {
        vec![CoordinateTransform::Scale(CoordinateTransformScale::from(
            physical_size,
        ))]
    });

    // let mut multiscales_metadata = serde_json::Map::with_capacity(3);
    let serde_json::Value::Object(multiscales_metadata) = serde_json::json!({
        "description": "Created with zarrs_ome",
        "repository": env!("CARGO_PKG_REPOSITORY"),
        "version": zarrs_tools::ZARRS_TOOLS_VERSION_WITH_ZARRS,
    }) else {
        unreachable!()
    };
    let multiscales_metadata: MultiscaleImageMetadata =
        MultiscaleImageMetadata(multiscales_metadata);

    let downsample_type = if cli.discrete {
        "mode"
    } else if cli.gaussian_sigma.is_none() {
        "average"
    } else {
        "gaussian"
    }
    .to_string();

    let mut datasets = Vec::with_capacity(cli.max_levels);

    let mut relative_scale = vec![1.0; array0.dimensionality()];
    {
        let dataset = MultiscaleImageDataset {
            path: "0".to_string(),
            coordinate_transformations: vec![CoordinateTransform::Scale(
                CoordinateTransformScale::from(relative_scale.clone()),
            )],
        };
        datasets.push(dataset);
    }

    // Calculate gaussian sigma/kernel size for each axis
    let downsample_factor: Vec<u64> = cli
        .downsample_factor
        .unwrap_or_else(|| vec![2; array0.dimensionality()]);
    let gaussian_filter = if let Some(gaussian_sigma) = cli.gaussian_sigma {
        let kernel_half_size = if let Some(kernel_half_size) = cli.gaussian_kernel_half_size {
            kernel_half_size
        } else {
            gaussian_sigma
                .iter()
                .map(|sigma| (sigma * 3.0).ceil() as u64)
                .collect_vec()
        };
        Some(Gaussian::new(
            gaussian_sigma.clone(),
            kernel_half_size.clone(),
            None,
        ))
    } else {
        None
    };

    let downsample_filter = Downsample::new(downsample_factor.clone(), cli.discrete, None);
    // println!("sigma:{sigma} kernel_half_size:{kernel_half_size}");

    for i in 1..=cli.max_levels {
        let bar = bars.get(i).unwrap();
        bar.reset();

        let progress_callback = |stats: ProgressStats| progress_callback(stats, bar);
        let progress_callback = ProgressCallback::new(&progress_callback);

        // Input
        let store = create_filesystem_store(&cli.output, cli.direct_io)?;
        let array_input = Array::open(store.into(), &format!("/{}", i - 1))?;

        // Filters

        // Setup reencoding (this is a bit hacky)
        let chunk_shape = array_input.chunk_shape(&vec![0; array_input.dimensionality()])?;
        let output_shape = downsample_filter.output_shape(&array_input).unwrap();
        let mut reencoding = ZarrReencodingArgs::default();
        if array_input.codecs().array_to_bytes_codec().identifier() == ShardingCodec::IDENTIFIER {
            reencoding.shard_shape = Some(
                std::iter::zip(&chunk_shape, &output_shape)
                    .map(|(c, s)| std::cmp::min(c.get(), *s))
                    .collect_vec(),
            );
            let decode_granularity = array_input
                .codecs()
                .partial_decode_granularity(&chunk_shape);
            reencoding.chunk_shape = Some(
                std::iter::zip(decode_granularity.as_slice(), &output_shape)
                    .map(|(g, s)| std::cmp::min(g.get(), *s))
                    .collect_vec(),
            );
        } else {
            reencoding.chunk_shape = Some(
                std::iter::zip(
                    array_input
                        .chunk_shape(&vec![0; array_input.dimensionality()])?
                        .as_slice(),
                    &output_shape,
                )
                .map(|(g, s)| std::cmp::min(g.get(), *s))
                .collect_vec(),
            );
        }
        // println!("{:?} {:?}", reencoding.chunk_shape, reencoding.shard_shape);
        let output_builder = downsample_filter.output_array_builder(&array_input, &reencoding)?;

        // Output
        let output_path = cli.output.join(i.to_string());
        let output_store = create_filesystem_store(&cli.output, cli.direct_io)?;
        let array_output = output_builder.build(output_store.into(), &format!("/{i}"))?;
        bar.set_prefix(format!("{i} {:?}", array_output.shape()));

        // Scale factor (inverse of downsample factor)
        std::iter::zip(&mut relative_scale, &downsample_factor).for_each(
            |(scale, downsample_factor)| {
                *scale *= *downsample_factor as f32;
            },
        );
        // println!("{downsample_factor:?} -> {scale:?}");

        // Chunks
        let chunks = ArraySubset::new_with_shape(array_output.chunk_grid_shape().to_vec());
        let progress = Progress::new(chunks.num_elements_usize(), &progress_callback);

        let chunk_limit = if let Some(chunk_limit) = cli.chunk_limit {
            chunk_limit
        } else {
            // Get memory usage
            let input_chunk_shape =
                array_input.chunk_shape(&vec![0; array_input.dimensionality()])?;
            let output_chunk_shape =
                array_output.chunk_shape(&vec![0; array_output.dimensionality()])?;
            let downsample_memory = downsample_filter.memory_per_chunk(
                (
                    &input_chunk_shape,
                    array_input.data_type(),
                    array_input.fill_value(),
                ),
                (
                    &output_chunk_shape,
                    array_output.data_type(),
                    array_output.fill_value(),
                ),
            );
            let memory_per_chunk = downsample_memory
                + if let Some(gaussian_filter) = &gaussian_filter {
                    let downsample_input_subset = downsample_filter.input_subset(
                        array_input.shape(),
                        &ArraySubset::new_with_shape(
                            bytemuck::must_cast_slice(&output_chunk_shape).to_vec(),
                        ),
                    );
                    let downsample_input_shape = downsample_input_subset
                        .shape()
                        .iter()
                        .map(|u| NonZeroU64::new(*u).unwrap())
                        .collect_vec();
                    gaussian_filter.memory_per_chunk(
                        (
                            &downsample_input_shape,
                            array_input.data_type(),
                            array_input.fill_value(),
                        ),
                        (
                            &output_chunk_shape,
                            array_input.data_type(),
                            array_input.fill_value(),
                        ),
                    )
                } else {
                    0
                };
            // let system = sysinfo::System::new_with_specifics(
            //     sysinfo::RefreshKind::new()
            //         .with_memory(sysinfo::MemoryRefreshKind::new().with_ram()),
            // );
            // let available_memory = usize::try_from(system.available_memory()).unwrap();
            // let free_memory = usize::try_from(system.free_memory()).unwrap();
            // let chunk_limit = zarrs_tools::filter::calculate_chunk_limit(memory_per_chunk)?;
            // println!(
            //     "{memory_per_chunk} -> {chunk_limit}/{} = {:.2}GB of {:.2}GB/{:.2}GB",
            //     chunks.num_elements(),
            //     (memory_per_chunk * chunk_limit) as f32 / 1e9,
            //     available_memory as f32 / 1e9,
            //     free_memory as f32 / 1e9,
            // );
            zarrs_tools::filter::calculate_chunk_limit(memory_per_chunk)?
        };

        // Apply
        let indices = chunks.indices();
        rayon_iter_concurrent_limit::iter_concurrent_limit!(
            chunk_limit,
            indices,
            try_for_each,
            |chunk_indices: ArrayIndicesTinyVec| {
                macro_rules! discrete_or_continuous {
                    ( $t:ty ) => {{
                        if cli.discrete {
                            apply_chunk_discrete::<$t>(
                                &array_input,
                                &array_output,
                                &chunk_indices,
                                &downsample_filter,
                                &progress,
                            )?
                        } else if let Some(gaussian_filter) = &gaussian_filter {
                            apply_chunk_continuous_gaussian::<$t>(
                                &array_input,
                                &array_output,
                                &chunk_indices,
                                &downsample_filter,
                                &gaussian_filter,
                                &progress,
                            )?
                        } else {
                            apply_chunk_continuous::<$t>(
                                &array_input,
                                &array_output,
                                &chunk_indices,
                                &downsample_filter,
                                &progress,
                            )?
                        }
                    }};
                }
                macro_rules! continuous {
                    ( $t:ty ) => {{
                        if let Some(gaussian_filter) = &gaussian_filter {
                            apply_chunk_continuous_gaussian::<$t>(
                                &array_input,
                                &array_output,
                                &chunk_indices,
                                &downsample_filter,
                                &gaussian_filter,
                                &progress,
                            )?
                        } else {
                            apply_chunk_continuous::<$t>(
                                &array_input,
                                &array_output,
                                &chunk_indices,
                                &downsample_filter,
                                &progress,
                            )?
                        }
                    }};
                }
                macro_rules! apply {
                    ( [$( ( $dt:ty, $t:ty, $inner:ident ) ),* ]) => {
                        match array_input.data_type().identifier() {
                            $(<$dt>::IDENTIFIER => { $inner!($t) } ,)*
                            id => panic!("unsupported data type: {}", id)
                        }
                    };
                }
                apply!([
                    (data_type::Int8DataType, i8, discrete_or_continuous),
                    (data_type::Int16DataType, i16, discrete_or_continuous),
                    (data_type::Int32DataType, i32, discrete_or_continuous),
                    (data_type::Int64DataType, i64, discrete_or_continuous),
                    (data_type::UInt8DataType, u8, discrete_or_continuous),
                    (data_type::UInt16DataType, u16, discrete_or_continuous),
                    (data_type::UInt32DataType, u32, discrete_or_continuous),
                    (data_type::UInt64DataType, u64, discrete_or_continuous),
                    (data_type::BFloat16DataType, bf16, continuous),
                    (data_type::Float16DataType, f16, continuous),
                    (data_type::Float32DataType, f32, continuous),
                    (data_type::Float64DataType, f64, continuous)
                ]);

                progress.next();
                Ok::<_, FilterError>(())
            }
        )?;

        // Append multiscales dataset metadata
        let dataset = MultiscaleImageDataset {
            path: format!("{i}"),
            coordinate_transformations: vec![
                CoordinateTransform::Scale(CoordinateTransformScale::from(relative_scale.clone())),
                CoordinateTransform::Translation(CoordinateTransformTranslation::from(
                    relative_scale.iter().map(|s| (s - 1.0) * 0.5).collect_vec(),
                )),
            ],
        };
        datasets.push(dataset);

        array_output.store_metadata()?;
        finish_step(bar, &output_path);

        // Stop when for all axis the output shape is 1 or stride is 1
        if std::iter::zip(&downsample_factor, &output_shape).all(|(df, s)| *df == 1 || *s == 1) {
            bars[i + 1..=cli.max_levels]
                .iter()
                .for_each(|bar| bar.finish_and_clear());
            break;
        }
    }

    match cli.ome_zarr_version {
        OMEZarrVersion::V0_5 => {
            let multiscales = vec![ome_zarr_metadata::v0_5::MultiscaleImage {
                name: cli.name,
                axes,
                datasets,
                coordinate_transformations: base_transform,
                r#type: Some(downsample_type),
                metadata: Some(multiscales_metadata),
            }];
            let ome = ome_zarr_metadata::v0_5::OmeFields {
                multiscales: Some(multiscales),
                ..Default::default()
            };
            group
                .attributes_mut()
                .insert("ome".to_string(), serde_json::to_value(ome).unwrap());
        }
    }

    // Store metadata
    group.store_metadata()?;

    let duration_s = start.elapsed().as_secs_f32();
    println!("Output {:?} in {duration_s:.2}s", cli.output);

    Ok(())
}

fn main() -> std::process::ExitCode {
    if let Err(err) = run() {
        println!("{err}");
        std::process::ExitCode::FAILURE
    } else {
        std::process::ExitCode::SUCCESS
    }
}
