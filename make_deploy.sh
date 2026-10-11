#!/bin/bash
set -e

echo "[1/5] Cleaning up temporary and junk files..."
rm -rf tts_training_data
rm -f generate_tts_data.py test_ort_cuda.rs test_pitch.rs test_tdpsola.rs "Linux Voice Changer Alternatives.md" launch_virtual_mic.sh

echo "[2/5] Compiling standalone release binary with embedded RPATH..."
export PKG_CONFIG_PATH=$HOME/.local/usr/lib/x86_64-linux-gnu/pkgconfig
export RUSTFLAGS="-C link-arg=-Wl,-rpath,\$ORIGIN"
cargo build --release

echo "[3/5] Assembling deployment directory..."
DEPLOY_DIR="hevoicechanger_deploy"
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

# Copy standalone binary
cp target/release/hevoicechanger "$DEPLOY_DIR/"

# Copy necessary shared libraries with symlink dereferencing (e.g. libwebgpu_dawn.so)
if ls target/release/*.so 1> /dev/null 2>&1; then
    cp -L target/release/*.so "$DEPLOY_DIR/"
fi

# Copy ONNX models folder
cp -r models "$DEPLOY_DIR/"

# Copy documentation and license
cp README.md LICENSE "$DEPLOY_DIR/" 2>/dev/null || true

echo "[4/5] Archiving into hevoicechanger_deploy.tar.gz..."
tar -czvf hevoicechanger_deploy.tar.gz "$DEPLOY_DIR"

echo "[5/5] Deployment package created successfully: hevoicechanger_deploy.tar.gz"
echo "The binary is standalone and finds bundled models and shared libraries automatically via \$ORIGIN."
