# zarrs_tools 

[![Latest Version](https://img.shields.io/crates/v/zarrs_tools.svg)](https://crates.io/crates/zarrs_tools)
![msrv](https://img.shields.io/crates/msrv/zarrs_tools)
[![build](https://github.com/zarrs/zarrs_tools/actions/workflows/ci.yml/badge.svg)](https://github.com/zarrs/zarrs_tools/actions/workflows/ci.yml)

Various tools for creating and manipulating [Zarr v3](https://zarr.dev) data with the [zarrs](https://github.com/zarrs/zarrs) rust crate.

A changelog can be found [here](https://github.com/zarrs/zarrs_tools/blob/main/CHANGELOG.md).

## Tools
All tools support input and output of [Zarr V3](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) data.
Some tools additionally support input of a [V3 compatible subset](https://docs.rs/zarrs/latest/zarrs/#implementation-status) of [Zarr V2](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html).

- [`zarrs_reencode`](https://github.com/zarrs/zarrs_tools/blob/main/docs/zarrs_reencode.md): reencode an array. Manipulate the chunk size, shard size, codecs, fill value, chunk key encoding separator, and attributes.
- [`zarrs_filter`](https://github.com/zarrs/zarrs_tools/blob/main/docs/zarrs_filter.md) (feature `filter`): apply simple image filters (transformations) to an array.
- [`zarrs_ome`](https://github.com/zarrs/zarrs_tools/blob/main/docs/zarrs_ome.md) (feature `ome`): convert an array to an [OME-Zarr](https://ngff.openmicroscopy.org/latest/index.html) 0.5 multi-scale image.
- [`zarrs_info`](https://github.com/zarrs/zarrs_tools/blob/main/docs/zarrs_info.md) (feature `info`): return metadata related info or the range/histogram of an array.
- [`zarrs_validate`](https://github.com/zarrs/zarrs_tools/blob/main/docs/zarrs_validate.md) (feature `validate`): validate that two arrays are equivalent.
- [`zarrs_binary2zarr`](https://github.com/zarrs/zarrs_tools/blob/main/docs/zarrs_binary2zarr.md) (feature `binary2zarr`): create an array from piped binary data.

See [docs/](https://github.com/zarrs/zarrs_tools/blob/main/docs/) for tool documentation.

## `zarrs` Benchmarking
- `zarrs_reencode`: suitable for round trip benchmarking.
- `zarrs_benchmark_read_sync` (feature `benchmark`): benchmark the zarrs sync API.
- `zarrs_benchmark_read_async` (feature `benchmark`): benchmark the zarrs async API.

See the [zarrs/zarr_benchmarks](https://github.com/zarrs/zarr_benchmarks) repository for benchmarks of `zarrs` against other Zarr V3 implementations.

## Install

### Prebuilt Binaries
```bash
# Requires cargo-binstall https://github.com/cargo-bins/cargo-binstall
cargo binstall zarrs_tools
```
Prebuilt binaries are not available on all platforms.

### From [crates.io](https://crates.io/crates/zarrs_tools)
```bash
cargo install --all-features zarrs_tools
```

### From [source](https://github.com/zarrs/zarrs_tools)
```bash
cargo install --all-features --path .
# cargo install --all-features --git https://github.com/zarrs/zarrs_tools
```

### Enabling SIMD intrinsics
Encoding and decoding performance may be improved with `avx2`/`sse2` enabled (if supported).

This can be enabled by compiling with either of:
 - `RUSTFLAGS="-C target-cpu=native"`
 - `RUSTFLAGS="-C target-feature=+avx2,+sse2"`

### Enabling non-default `zarrs` codecs
Non-default `zarrs` codecs (see [`zarrs` crate features](https://docs.rs/zarrs/latest/zarrs/#crate-features)) can be enabled by passing them as feature flags.
For example:
```bash
cargo install zarrs_tools --all-features --features zarrs/bitround,zarrs/zfp,zarrs/bz2,zarrs/pcodec
```

## Licence
`zarrs_tools` is licensed under either of
 - the Apache License, Version 2.0 [LICENSE-APACHE](./LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
 - the MIT license [LICENSE-MIT](./LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
