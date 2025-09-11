#!/bin/bash
# Cross-compilation script for memexd binary
# 
# Builds memexd for multiple platforms:
# - macOS Intel (x86_64-apple-darwin)
# - macOS Apple Silicon (aarch64-apple-darwin)  
# - Linux Intel (x86_64-unknown-linux-gnu)
# - Windows Intel (x86_64-pc-windows-msvc)
# - Windows ARM (aarch64-pc-windows-msvc)

set -e  # Exit on error

# Change to the daemon directory
cd "$(dirname "$0")/src/rust/daemon"

# Create release builds directory
mkdir -p "target/release-builds"

# Define our target platforms
TARGETS=(
    "x86_64-apple-darwin"      # macOS Intel
    "aarch64-apple-darwin"     # macOS Apple Silicon  
    "x86_64-unknown-linux-gnu" # Linux Intel
    "x86_64-pc-windows-msvc"   # Windows Intel
    "aarch64-pc-windows-msvc"  # Windows ARM
)

echo "🚀 Starting cross-compilation of memexd for ${#TARGETS[@]} targets..."
echo ""

# Function to build for a target
build_target() {
    local target=$1
    local binary_name="memexd"
    
    # Add .exe extension for Windows targets
    if [[ $target == *"windows"* ]]; then
        binary_name="memexd.exe"
    fi
    
    echo "🔨 Building for $target..."
    
    # Build the target
    if cargo build --bin memexd --release --target "$target" --quiet; then
        # Copy the binary to our release builds directory with target suffix
        local source_path="target/$target/release/$binary_name"
        local dest_name="memexd-$target"
        
        if [[ $target == *"windows"* ]]; then
            dest_name="memexd-$target.exe"
        fi
        
        if [ -f "$source_path" ]; then
            cp "$source_path" "target/release-builds/$dest_name"
            
            # Get file size
            local size=$(stat -f%z "target/release-builds/$dest_name" 2>/dev/null || stat -c%s "target/release-builds/$dest_name" 2>/dev/null || echo "unknown")
            echo "   ✅ Success: $dest_name (${size} bytes)"
        else
            echo "   ❌ Error: Binary not found at $source_path"
            return 1
        fi
    else
        echo "   ❌ Build failed for $target"
        return 1
    fi
    
    echo ""
}

# Build for each target
success_count=0
total_count=${#TARGETS[@]}

for target in "${TARGETS[@]}"; do
    if build_target "$target"; then
        ((success_count++))
    fi
done

echo "📊 Build Summary:"
echo "   Success: $success_count/$total_count targets"
echo ""

if [ $success_count -eq $total_count ]; then
    echo "🎉 All targets built successfully!"
else
    echo "⚠️  Some targets failed to build"
fi

# List all built binaries
echo ""
echo "📦 Built binaries in target/release-builds/:"
ls -lah target/release-builds/ | grep memexd || echo "   No binaries found"

echo ""
echo "✨ Cross-compilation complete!"