#!/bin/bashifflib-rs/build_wheels.sh
# Build wheels for multiple platforms

echo "Building wheels for multiple platforms..."

# Create dist directory if it doesn't exist
mkdir -p dist

# Build macOS wheel (native)
echo "Building macOS wheel..."
maturin build --release --out dist

# Build Linux x86_64 wheel
echo "Building Linux x86_64 wheel..."
docker run --rm --platform linux/amd64 -v $(pwd):/io ghcr.io/pyo3/maturin build --release --out dist --find-interpreter --compatibility manylinux2014

# Build Linux ARM64 wheel
echo "Building Linux ARM64 wheel..."
docker run --rm --platform linux/arm64 -v $(pwd):/io ghcr.io/pyo3/maturin build --release --out dist --find-interpreter --compatibility manylinux2014

echo "All wheels built successfully!"
ls -la dist/