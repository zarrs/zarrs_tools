use std::{error::Error, sync::Arc};

use clap::{Parser, Subcommand};
use rayon::current_num_threads;
use serde::Serialize;
use serde_json::Number;
use zarrs::{
    array::{Array, ArrayMetadataOptions, DimensionName, FillValueMetadataV3},
    filesystem::FilesystemStore,
    group::{Group, GroupMetadataOptions},
    metadata::v3::MetadataV3,
    node::{Node, NodeMetadata},
};

/// Get information about a Zarr array or group.
///
/// Outputs are JSON encoded.
#[derive(Parser)]
#[command(author, version=zarrs_tools::ZARRS_TOOLS_VERSION_WITH_ZARRS)]
struct Cli {
    /// The maximum number of chunks concurrently processed.
    ///
    /// Defaults to the RAYON_NUM_THREADS environment variable or the number of logical CPUs.
    /// Consider reducing this for images with large chunk sizes or on systems with low memory availability.
    #[arg(long, default_value_t = current_num_threads())]
    chunk_limit: usize,

    /// Path to the Zarr input array or group.
    path: std::path::PathBuf,

    #[command(subcommand)]
    command: InfoCommand,
}

#[derive(Parser, Debug)]
struct HistogramParams {
    n_bins: usize,
    min: f64,
    max: f64,
}

#[derive(Subcommand, Debug)]
enum InfoCommand {
    /// Get the array/group metadata.
    Metadata,
    /// Get the array/group metadata (interpreted as V3).
    MetadataV3,
    /// Get the array/group attributes.
    Attributes,
    /// Get the array shape.
    Shape,
    /// Get the array data type.
    DataType,
    /// Get the array fill value.
    FillValue,
    /// Get the array dimension names.
    DimensionNames,
    /// Get the array data range.
    Range,
    /// Get the array data histogram.
    Histogram(HistogramParams),
}

fn main() -> std::process::ExitCode {
    if let Err(err) = run() {
        println!("{err}");
        std::process::ExitCode::FAILURE
    } else {
        std::process::ExitCode::SUCCESS
    }
}

fn group_metadata_options_v3() -> GroupMetadataOptions {
    let mut metadata_options = GroupMetadataOptions::default();
    metadata_options.set_metadata_convert_version(zarrs::config::MetadataConvertVersion::V3);
    metadata_options
}

fn array_metadata_options_v3() -> ArrayMetadataOptions {
    let mut metadata_options = ArrayMetadataOptions::default();
    metadata_options.set_metadata_convert_version(zarrs::config::MetadataConvertVersion::V3);
    metadata_options.set_include_zarrs_metadata(false);
    metadata_options
}

fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let storage = Arc::new(FilesystemStore::new(&cli.path)?);

    let node = Node::open(&storage, "/")?;
    if let NodeMetadata::Group(_) = node.metadata() {
        // Group handling
        let group = Group::open(storage.clone(), "/")?;
        match cli.command {
            InfoCommand::Metadata => {
                println!("{}", serde_json::to_string_pretty(group.metadata())?);
            }
            InfoCommand::MetadataV3 => {
                let metadata = group.metadata_opt(&group_metadata_options_v3());
                println!("{}", serde_json::to_string_pretty(&metadata)?);
            }
            InfoCommand::Attributes => {
                println!("{}", serde_json::to_string_pretty(group.attributes())?);
            }
            _ => {
                println!("The {:?} command is not supported for a group", cli.command)
            }
        }
    } else {
        // Array handling
        let array = Array::open(storage.clone(), "/")?;
        match cli.command {
            InfoCommand::Metadata => {
                println!("{}", serde_json::to_string_pretty(array.metadata())?);
            }
            InfoCommand::MetadataV3 => {
                let metadata = array.metadata_opt(&array_metadata_options_v3());
                println!("{}", serde_json::to_string_pretty(&metadata)?);
            }
            InfoCommand::Attributes => {
                println!("{}", serde_json::to_string_pretty(array.attributes())?);
            }
            InfoCommand::Shape => {
                #[derive(Serialize)]
                struct Shape {
                    shape: Vec<u64>,
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&Shape {
                        shape: array.shape().to_vec()
                    })?
                );
            }
            InfoCommand::DataType => {
                #[derive(Serialize)]
                struct DataType {
                    data_type: MetadataV3,
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&DataType {
                        data_type: array.data_type().metadata()
                    })?
                );
            }
            InfoCommand::FillValue => {
                #[derive(Serialize)]
                struct FillValue {
                    fill_value: FillValueMetadataV3,
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&FillValue {
                        fill_value: array.data_type().metadata_fill_value(array.fill_value())?
                    })?
                );
            }
            InfoCommand::DimensionNames => {
                #[derive(Serialize)]
                struct DimensionNames {
                    dimension_names: Option<Vec<DimensionName>>,
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&DimensionNames {
                        dimension_names: array.dimension_names().clone()
                    })?
                );
            }
            InfoCommand::Range => {
                let (min, max) = zarrs_tools::info::calculate_range(&array, cli.chunk_limit)?;
                #[derive(Serialize)]
                struct MinMax {
                    min: Number,
                    max: Number,
                }
                println!("{}", serde_json::to_string_pretty(&MinMax { min, max })?);
            }
            InfoCommand::Histogram(histogram_params) => {
                let (bin_edges, hist) = zarrs_tools::info::calculate_histogram(
                    &array,
                    histogram_params.n_bins,
                    histogram_params.min,
                    histogram_params.max,
                    cli.chunk_limit,
                )?;
                #[derive(Serialize)]
                struct Histogram {
                    bin_edges: Vec<f64>,
                    hist: Vec<u64>,
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&Histogram { bin_edges, hist })?
                );
            }
        }
    }

    Ok(())
}
