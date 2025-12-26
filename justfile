toolchain := "nightly"

# Display the available recipes
_help:
    @just --list --unsorted

# Build the project.
build:
    cargo +{{toolchain}} build

# Run tests.
test:
    cargo +{{toolchain}} test --all-features
    cargo +{{toolchain}} test --all-features --examples

# Format the code.
fmt:
    cargo +{{toolchain}} fmt

# Generate documentation.
doc:
    RUSTDOCFLAGS="-D warnings --cfg docsrs" cargo +{{toolchain}} doc -Z unstable-options -Z rustdoc-scrape-examples --no-deps

# Run clippy.
clippy:
    cargo +{{toolchain}} clippy --all-features -- -D warnings

# Run all checks.
check: build test clippy doc
    cargo +{{toolchain}} fmt --all -- --check
    cargo +{{toolchain}} check

# Install coverage tools.
_coverage_install:
    cargo install cargo-llvm-cov --locked

# Generate a coverage report.
_coverage_report:
    cargo +{{toolchain}} llvm-cov --doctests --html

# Generate a coverage file.
_coverage_file:
    cargo +{{toolchain}} llvm-cov --doctests --lcov --output-path lcov.info
