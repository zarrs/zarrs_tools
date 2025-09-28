use std::{
    collections::HashMap,
    error::Error,
    path::{Path, PathBuf},
    sync::Arc,
};

use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use itertools::Itertools;
use tempfile::TempDir;
use zarrs::{
    array::{Array, ArrayBuilder, ArrayCreateError},
    filesystem::{FilesystemStore, FilesystemStoreOptions},
    storage::{StorageError, StorePrefix, WritableStorageTraits},
};
use zarrs_tools::{
    filter::{
        FilterCommand, FilterCommandTraits, FilterError, FilterTraits, PathOrIdentifier,
        PathOrTempPath,
    },
    progress::{ProgressCallback, ProgressStats},
    ZarrReencodingArgs,
};

#[derive(clap::ValueEnum, Debug, Clone)]
enum OutputExists {
    /// Erase the output
    Erase,
    /// Exit if the output already exists
    Exit,
}

/// Apply simple image filters (transformations) to a Zarr array.
#[derive(Parser, Debug)]
#[command(author, version=zarrs_tools::ZARRS_TOOLS_VERSION_WITH_ZARRS)]
struct Cli {
    /// Behaviour if the output exists.
    #[arg(long)]
    #[clap(value_enum, default_value_t=OutputExists::Erase)]
    exists: OutputExists,

    /// Directory for temporary arrays.
    ///
    /// If omitted, defaults to the platform-specific temporary directory (e.g. ${TMPDIR}, /tmp, etc.)
    #[arg(long)]
    pub tmp: Option<PathBuf>,

    /// The maximum number of chunks concurrently processed.
    ///
    /// By default, this is set to the number of CPUs.
    /// Consider reducing this for images with large chunk sizes or on systems with low memory availability.
    #[arg(long)]
    pub chunk_limit: Option<usize>,

    /// Path to a JSON run configuration.
    pub run_config: Option<PathBuf>,

    /// Enable direct I/O for filesystem operations.
    ///
    /// If set, filesystem operations will use direct I/O bypassing the page cache.
    #[arg(long, default_value_t = false)]
    direct_io: bool,

    #[command(subcommand)]
    filter: Option<FilterCommand>,
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

fn load_array<P: Into<PathBuf>>(
    path: P,
    direct_io: bool,
) -> Result<Array<FilesystemStore>, ArrayCreateError> {
    let mut options = FilesystemStoreOptions::default();
    options.direct_io(direct_io);
    let store = FilesystemStore::new_with_options(path.into(), options)
        .map_err(|err| ArrayCreateError::StorageError(StorageError::Other(err.to_string())))?;
    Array::open(store.into(), "/")
}

/// Removes array if it exists
fn create_array<P: Into<PathBuf>>(
    path: P,
    builder: &ArrayBuilder,
    direct_io: bool,
) -> Result<Array<FilesystemStore>, ArrayCreateError> {
    let mut options = FilesystemStoreOptions::default();
    options.direct_io(direct_io);
    let store = FilesystemStore::new_with_options(path.into(), options)
        .map_err(|err| ArrayCreateError::StorageError(StorageError::Other(err.to_string())))?;
    store.erase_prefix(&StorePrefix::root()).unwrap();
    builder.build(store.into(), "/")
}

fn get_array_input_output(
    filter: &dyn FilterTraits,
    input: &std::path::Path,
    output: &std::path::Path,
    reencode: &ZarrReencodingArgs,
    direct_io: bool,
) -> Result<(Array<FilesystemStore>, Array<FilesystemStore>), ArrayCreateError> {
    let array_input = load_array(input, direct_io)?;
    let array_output = create_array(
        output,
        &filter.output_array_builder(&array_input, reencode)?,
        direct_io,
    )?;
    Ok((array_input, array_output))
}

fn get_path(
    path_or_id: &Option<PathOrIdentifier>,
    tmp_dir: &std::path::Path,
    id_to_path: &mut HashMap<String, Arc<TempDir>>,
    last_output: &Option<std::path::PathBuf>,
) -> std::io::Result<PathOrTempPath> {
    if let Some(path_or_id) = path_or_id {
        match path_or_id {
            PathOrIdentifier::Identifier(id) => {
                // Named temporary output
                let entry = id_to_path.entry(id.clone()).or_insert_with(|| {
                    tempfile::TempDir::with_prefix_in(id, tmp_dir)
                        .unwrap()
                        .into()
                });
                Ok(PathOrTempPath::TempPath(entry.clone()))
            }
            PathOrIdentifier::Path(path) => {
                // Long lived output
                Ok(PathOrTempPath::Path(path.clone()))
            }
        }
    } else {
        // Unnamed temporary
        if let Some(last_output) = last_output {
            Ok(PathOrTempPath::Path(last_output.clone()))
        } else {
            Ok(PathOrTempPath::TempPath(
                tempfile::TempDir::new_in(tmp_dir)?.into(),
            ))
        }
    }
}

fn main() -> std::process::ExitCode {
    if let Err(err) = run() {
        println!("{err}");
        std::process::ExitCode::FAILURE
    } else {
        std::process::ExitCode::SUCCESS
    }
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

fn run() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let cli = Cli::parse();

    let start = std::time::Instant::now();

    let multi_progress = MultiProgress::new();

    // Create temporary directory
    let tmp_dir = if let Some(tmp) = cli.tmp {
        PathOrTempPath::Path(tmp.clone())
    } else {
        PathOrTempPath::TempPath(tempfile::tempdir()?.into())
    };

    // Get the filters
    let mut filter_commands: Vec<FilterCommand> = if let Some(config) = cli.run_config {
        let config = std::fs::read_to_string(config)?;
        serde_json::from_str(&config)?
    } else if let Some(filter) = cli.filter {
        vec![filter]
    } else {
        Err(std::io::Error::other("no filters supplied"))?
    };

    // Setup progress bars
    let bars = filter_commands
        .iter()
        .map(|filter| {
            let bar = multi_progress.add(ProgressBar::new(1));
            bar.set_style(bar_style_run());
            bar.set_prefix(filter.name());
            bar
        })
        .collect_vec();

    // Propagate global settings to filters
    for filter in &mut filter_commands {
        if let Some(global_chunk_limit) = cli.chunk_limit {
            let filter_chunk_limit = filter.common_args_mut().chunk_limit_mut();
            if filter_chunk_limit.is_none() {
                *filter_chunk_limit = Some(global_chunk_limit);
            }
        }
    }

    // Get the input and output paths
    let InputsOutputsExists {
        input_paths,
        output_paths,
        exists,
    } = get_input_output_paths(&filter_commands, tmp_dir.path())?;

    // Handle an existing output
    match cli.exists {
        OutputExists::Exit => {
            if exists.iter().any(|i| *i) {
                Err(FilterError::Other("Output exists, exiting".to_string()))?;
            }
        }
        OutputExists::Erase => {}
    }

    // Instantiate the filters
    let filters: Vec<Box<dyn FilterTraits>> = filter_commands
        .iter()
        .map(|filter| filter.init())
        .try_collect()?;

    // Collect filters/input/outputs and check compatibility
    let direct_io = cli.direct_io;
    let filter_input_output: Vec<_> = itertools::izip!(
        &filter_commands,
        &bars,
        &filters,
        &input_paths,
        &output_paths,
        &exists
    )
    .enumerate()
    .map(
        |(i, (filter_command, bar, filter, input, output, exists))| {
            let (array_input, array_output) = get_array_input_output(
                filter,
                input.path(),
                output.path(),
                filter_command.common_args().reencode(),
                direct_io,
            )?;
            bar.println(format!(
            "{}{}\n\targs:   {}\n\tencode: {}\n\tinput:  {} {:?} {:?}\n\toutput: {} {:?} {:?}{}",
            if filters.len() == 1 {
                "".to_string()
            } else {
                format!("{i} ")
            },
            filter_command.name(),
            filter_command.args_str(),
            filter_command.reencode_str(),
            array_input.data_type(),
            array_input.shape(),
            input.path(),
            array_output.data_type(),
            array_output.shape(),
            output.path(),
            if *exists { " (overwrite)" } else { "" },
        ));
            array_output.store_metadata()?; // erased before filter run

            filter.is_compatible(
                &array_input.chunk_array_representation(&vec![0; array_input.dimensionality()])?,
                &array_output
                    .chunk_array_representation(&vec![0; array_output.dimensionality()])?,
            )?;
            Ok::<_, FilterError>((
                filter_command.name(),
                filter,
                array_input,
                array_output,
                output.path(),
            ))
        },
    )
    .try_collect()?;

    // Erase output metadata to imply indicating that filter has not run
    filter_input_output
        .iter()
        .try_for_each(|(_, _, _, array_output, _)| array_output.erase_metadata())?;

    // Run the filters
    std::iter::zip(filter_input_output, bars).try_for_each(
        |((_name, filter, array_input, mut array_output, output_path), bar)| {
            bar.reset();

            let progress_callback = |stats: ProgressStats| progress_callback(stats, &bar);
            let progress_callback = ProgressCallback::new(&progress_callback);
            // Run the filter
            filter.apply(&array_input, &mut array_output, &progress_callback)?;

            // Write metadata to indicate that filter is finished
            array_output.store_metadata()?;

            bar.set_style(bar_style_finish());
            bar.set_prefix(format!(
                "{} {}",
                bar.prefix(),
                output_path.to_string_lossy()
            ));
            bar.abandon();
            Ok::<(), FilterError>(())
        },
    )?;

    let duration_s = start.elapsed().as_secs_f32();
    println!("Completed in {duration_s:.2}s");

    Ok(())
}

struct InputsOutputsExists {
    input_paths: Vec<PathOrTempPath>,
    output_paths: Vec<PathOrTempPath>,
    exists: Vec<bool>,
}

fn get_input_output_paths(
    filters: &[FilterCommand],
    tmp_dir: &Path,
) -> Result<InputsOutputsExists, FilterError> {
    let mut id_to_path = HashMap::<String, Arc<TempDir>>::new();
    let mut input_paths = Vec::<PathOrTempPath>::with_capacity(filters.len());
    let mut output_paths = Vec::<PathOrTempPath>::with_capacity(filters.len());
    let mut last_output: Option<std::path::PathBuf> = None;
    let mut exists = Vec::<bool>::with_capacity(filters.len());
    for filter in filters {
        if let Some(PathOrIdentifier::Path(output_path)) = filter.io_args().output() {
            exists.push(output_path.exists());
        } else {
            exists.push(false);
        }

        // println!("{filter:#?}");
        let input_path = get_path(
            filter.io_args().input(),
            tmp_dir,
            &mut id_to_path,
            &last_output,
        )?;
        let output_path = get_path(filter.io_args().output(), tmp_dir, &mut id_to_path, &None)?;
        if input_paths.is_empty() {
            if let PathOrTempPath::TempPath(_) = input_path {
                Err(std::io::Error::other(
                    "the first filter must have a valid input path",
                ))?
            }
        }
        last_output = Some(output_path.path().to_path_buf());
        // println!("{:?} -> {:?}", input_path.path(), output_path.path());

        input_paths.push(input_path);
        output_paths.push(output_path);
    }

    Ok(InputsOutputsExists {
        input_paths,
        output_paths,
        exists,
    })
}
