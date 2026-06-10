# Homebrew formula TEMPLATE for workspace-qdrant-mcp
#
# The release workflow stamps PLACEHOLDER_VERSION and the per-platform
# sha256 placeholders, then attaches the result to the GitHub release
# (see .github/workflows/release.yml "Stamp package manifests"). The
# in-repo copy is NOT directly installable.
#
# For tap-based installation (future):
#   brew tap ChrisGVE/workspace-qdrant-mcp
#   brew install workspace-qdrant-mcp

class WorkspaceQdrantMcp < Formula
  desc "Project-scoped vector database for AI assistants with hybrid search"
  homepage "https://github.com/ChrisGVE/workspace-qdrant-mcp"
  version "PLACEHOLDER_VERSION"
  license "MIT"

  # Platform-specific binary downloads
  on_macos do
    on_arm do
      url "https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/download/v#{version}/workspace-qdrant-mcp-darwin-arm64.tar.gz"
      # sha256 "PLACEHOLDER_DARWIN_ARM64"
    end
    on_intel do
      url "https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/download/v#{version}/workspace-qdrant-mcp-darwin-x64.tar.gz"
      # sha256 "PLACEHOLDER_DARWIN_X64"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/download/v#{version}/workspace-qdrant-mcp-linux-arm64.tar.gz"
      # sha256 "PLACEHOLDER_LINUX_ARM64"
    end
    on_intel do
      url "https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/download/v#{version}/workspace-qdrant-mcp-linux-x64.tar.gz"
      # sha256 "PLACEHOLDER_LINUX_X64"
    end
  end

  # No runtime dependencies - binaries are self-contained with ONNX Runtime statically linked

  def install
    bin.install "wqm"
    bin.install "memexd"
    bin.install "workspace-qdrant-mcp"

    # Install shell completions if available
    if File.exist?("completions/wqm.bash")
      bash_completion.install "completions/wqm.bash" => "wqm"
    end
    if File.exist?("completions/wqm.zsh")
      zsh_completion.install "completions/wqm.zsh" => "_wqm"
    end
    if File.exist?("completions/wqm.fish")
      fish_completion.install "completions/wqm.fish"
    end
  end

  def post_install
    # Create default config directory
    (var/"workspace-qdrant").mkpath
  end

  def caveats
    <<~EOS
      To start using workspace-qdrant-mcp:

      1. Start Qdrant (if not already running):
         docker run -d -p 6333:6333 qdrant/qdrant

      2. Check connectivity:
         wqm admin health

      3. Install the daemon service:
         wqm service install
         wqm service start

      For MCP server integration with Claude, see:
        https://github.com/ChrisGVE/workspace-qdrant-mcp#configure-mcp
    EOS
  end

  service do
    run [opt_bin/"memexd"]
    keep_alive true
    working_dir var/"workspace-qdrant"
    log_path var/"log/memexd.log"
    error_log_path var/"log/memexd.error.log"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/wqm --version")
    assert_match "memexd", shell_output("#{bin}/memexd --help")
    assert_match "workspace-qdrant-mcp", shell_output("#{bin}/workspace-qdrant-mcp --version")
  end
end
