#!/bin/bash
set -e

echo "🧹 Cleaning up project junk files..."
rm -rf tts_training_data
rm -f generate_tts_data.py test_ort_cuda.rs test_pitch.rs test_tdpsola.rs "Linux Voice Changer Alternatives.md" launch_virtual_mic.sh

echo "⚙️ Compiling standalone release binary..."
export PKG_CONFIG_PATH=$HOME/.local/usr/lib/x86_64-linux-gnu/pkgconfig
export RUSTFLAGS="-C link-arg=-Wl,-rpath,\$ORIGIN"
cargo build --release

echo "📦 Assembling deployment folder..."
DEPLOY_DIR="hevoicechanger_deploy"
rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR

# Copy standalone binary
cp target/release/hevoicechanger $DEPLOY_DIR/

# Copy necessary shared libraries (e.g. libwebgpu_dawn.so)
if ls target/release/*.so 1> /dev/null 2>&1; then
    cp target/release/*.so $DEPLOY_DIR/
fi

# Copy ONNX models folder
cp -r models $DEPLOY_DIR/

echo "🗜️ Packing into hevoicechanger_deploy.tar.gz..."
tar -czvf hevoicechanger_deploy.tar.gz $DEPLOY_DIR

echo "✅ Done! Deployment package is ready: hevoicechanger_deploy.tar.gz"
echo "Inside the extracted folder, the binary is perfectly standalone and will find the models and .so files automatically."
