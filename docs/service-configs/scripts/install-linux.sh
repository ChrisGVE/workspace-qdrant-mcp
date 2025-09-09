#!/bin/bash

# Linux systemd service installation script for memexd
# Run with: sudo bash install-linux.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="memexd"
SERVICE_USER="memexd"
SERVICE_GROUP="memexd"
BINARY_PATH="/usr/local/bin/memexd"
CONFIG_DIR="/etc/memexd"
CONFIG_FILE="$CONFIG_DIR/config.toml"
DATA_DIR="/var/lib/memexd"
LOG_DIR="/var/log/memexd"
SYSTEMD_SERVICE="/etc/systemd/system/memexd.service"

echo "Installing memexd as systemd service..."

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo "Error: This script must be run as root (use sudo)" >&2
    exit 1
fi

# Check if memexd binary exists
if [[ ! -f "$BINARY_PATH" ]]; then
    echo "Error: memexd binary not found at $BINARY_PATH"
    echo "Please build and install the binary first:"
    echo "  cd rust-engine"
    echo "  cargo build --release --bin memexd"
    echo "  sudo cp target/release/memexd $BINARY_PATH"
    echo "  sudo chmod +x $BINARY_PATH"
    exit 1
fi

# Create service user and group
echo "Creating service user and group..."
if ! getent group "$SERVICE_GROUP" >/dev/null 2>&1; then
    groupadd --system "$SERVICE_GROUP"
    echo "Created group: $SERVICE_GROUP"
fi

if ! getent passwd "$SERVICE_USER" >/dev/null 2>&1; then
    useradd --system --gid "$SERVICE_GROUP" --shell /bin/false \
            --home-dir "$DATA_DIR" --create-home \
            --comment "memexd service user" "$SERVICE_USER"
    echo "Created user: $SERVICE_USER"
fi

# Create directories
echo "Creating directories..."
mkdir -p "$CONFIG_DIR" "$DATA_DIR" "$LOG_DIR"

# Set ownership and permissions
chown root:root "$CONFIG_DIR"
chmod 755 "$CONFIG_DIR"

chown "$SERVICE_USER:$SERVICE_GROUP" "$DATA_DIR"
chmod 750 "$DATA_DIR"

chown "$SERVICE_USER:$SERVICE_GROUP" "$LOG_DIR"
chmod 750 "$LOG_DIR"

# Create configuration file if it doesn't exist
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Creating default configuration..."
    cat > "$CONFIG_FILE" << 'EOF'
# memexd Daemon Configuration for Linux systemd

[daemon]
port = 8765
log_level = "info"
worker_threads = 4
enable_file_watching = true
batch_size = 50

[qdrant]
url = "http://localhost:6333"
api_key = ""
timeout = 30
prefer_grpc = false

[embedding]
model = "sentence-transformers/all-MiniLM-L6-v2"
enable_sparse_vectors = true
chunk_size = 800
chunk_overlap = 120
batch_size = 50

[workspace]
collections = ["project"]
global_collections = []
max_collections = 100

[workspace.file_patterns]
include = ["*.md", "*.txt", "*.py", "*.js", "*.ts", "*.json", "*.yaml", "*.yml", "*.toml", "*.rs"]
exclude = ["node_modules/**", ".git/**", "target/**", "build/**", "dist/**", "*.pyc"]

[logging]
file_logging = true
log_file_path = "/var/log/memexd/memexd.log"
json_format = true
max_file_size_mb = 100
max_backup_files = 5
metrics_enabled = true

[security]
bind_address = "127.0.0.1"
max_request_size_mb = 10
rate_limit_requests_per_minute = 1000

[performance]
max_memory_mb = 2048
document_timeout_seconds = 300
health_check_interval_seconds = 60
EOF

    chown root:"$SERVICE_GROUP" "$CONFIG_FILE"
    chmod 640 "$CONFIG_FILE"
    echo "Created configuration file: $CONFIG_FILE"
fi

# Create systemd service file
echo "Creating systemd service file..."
cat > "$SYSTEMD_SERVICE" << EOF
[Unit]
Description=Memory eXchange Daemon - Document processing service
Documentation=https://github.com/ChrisGVE/workspace-qdrant-mcp
After=network-online.target
Wants=network-online.target
RequiresMountsFor=$DATA_DIR

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$DATA_DIR
ExecStart=$BINARY_PATH --config=$CONFIG_FILE --pid-file=/var/run/memexd.pid
ExecReload=/bin/kill -HUP \$MAINPID

# Restart configuration
Restart=always
RestartSec=10
StartLimitInterval=300
StartLimitBurst=3

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$DATA_DIR $LOG_DIR
PrivateTmp=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictNamespaces=true
RestrictSUIDSGID=true

# Resource limits
MemoryHigh=1G
MemoryMax=2G
TasksMax=100
CPUQuota=80%

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=memexd

[Install]
WantedBy=multi-user.target
EOF

echo "Created systemd service file: $SYSTEMD_SERVICE"

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable service for automatic startup
echo "Enabling service for automatic startup..."
systemctl enable memexd.service

# Test configuration
echo "Testing configuration..."
if sudo -u "$SERVICE_USER" "$BINARY_PATH" --config="$CONFIG_FILE" --foreground --log-level=debug &
then
    DAEMON_PID=$!
    sleep 3
    if kill -0 "$DAEMON_PID" 2>/dev/null; then
        echo "✅ Daemon started successfully in test mode"
        kill "$DAEMON_PID"
        wait "$DAEMON_PID" 2>/dev/null || true
    else
        echo "❌ Daemon failed to start in test mode"
        exit 1
    fi
else
    echo "❌ Failed to start daemon in test mode"
    exit 1
fi

# Start the service
echo "Starting memexd service..."
systemctl start memexd.service

# Wait a moment and check status
sleep 3
if systemctl is-active --quiet memexd.service; then
    echo "✅ Service installed and started successfully!"
    echo ""
    echo "Service Details:"
    systemctl status memexd.service --no-pager
    echo ""
    echo "Management Commands:"
    echo "  Start:   sudo systemctl start memexd.service"
    echo "  Stop:    sudo systemctl stop memexd.service"
    echo "  Restart: sudo systemctl restart memexd.service"
    echo "  Status:  sudo systemctl status memexd.service"
    echo "  Logs:    sudo journalctl -u memexd.service -f"
    echo ""
    echo "Configuration:"
    echo "  Config:  $CONFIG_FILE"
    echo "  Data:    $DATA_DIR"
    echo "  Logs:    $LOG_DIR"
else
    echo "❌ Service installed but failed to start. Check logs:"
    echo "  sudo journalctl -u memexd.service --no-pager"
    exit 1
fi

echo ""
echo "Installation complete!"