#!/bin/bash
# Build wheels for multiple platforms and Python versions

echo "Building wheels for multiple platforms and Python versions..."

# Create dist directory if it doesn't exist
mkdir -p dist

# Clean previous builds
rm -rf dist/*

# Python versions to build for
PYTHON_VERSIONS="3.10 3.11 3.12"

echo "Building macOS wheels..."
# Skip macOS x86_64 wheel due to cross-compilation issues
echo "  Skipping macOS x86_64 (cross-compilation issues)..."

# Build macOS ARM64 wheel (Apple Silicon)
echo "  Building macOS ARM64..."
for py_ver in $PYTHON_VERSIONS; do
    echo "    Python $py_ver"
    maturin build --release --out dist --interpreter python$py_ver --target aarch64-apple-darwin || echo "    Warning: Python $py_ver not found for ARM64"
done

echo "Building Linux wheels..."
# Build Linux x86_64 wheels
echo "  Building Linux x86_64..."
docker run --rm --platform linux/amd64 -v $(pwd):/io ghcr.io/pyo3/maturin build --release --out dist --find-interpreter --compatibility manylinux2014

# Build Linux ARM64 wheels
echo "  Building Linux ARM64..."
docker run --rm --platform linux/arm64 -v $(pwd):/io ghcr.io/pyo3/maturin build --release --out dist --find-interpreter --compatibility manylinux2014

# Build Linux i686 wheels
echo "  Building Linux i686..."
docker run --rm --platform linux/386 -v $(pwd):/io ghcr.io/pyo3/maturin build --release --out dist --find-interpreter --compatibility manylinux2014 --target i686-unknown-linux-gnu || echo "    Warning: i686 build failed"

echo "Building Windows wheels..."
# Note: Windows wheels require a Windows environment or cross-compilation setup
# For now, we'll note this limitation
echo "  Windows wheels require Windows environment or GitHub Actions CI"
echo "  Consider using the GitHub Actions workflow for Windows builds"

echo "Building source distribution..."
maturin sdist --out dist

echo "All available wheels built successfully!"
echo "Built files:"
ls -la dist/

echo ""
echo "To upload to PyPI:"
echo "  twine upload dist/*"
echo ""
echo "To upload to TestPyPI first:"
echo "  twine upload --repository testpypi dist/*"