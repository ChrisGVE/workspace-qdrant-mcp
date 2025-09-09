#!/bin/bash

# macOS LaunchDaemon installation script for memexd
# Run with: sudo bash install-macos.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="com.workspace-qdrant.memexd"
SERVICE_USER="_memexd"
SERVICE_GROUP="_memexd"
SERVICE_UID=499
SERVICE_GID=499
BINARY_PATH="/usr/local/bin/memexd"
CONFIG_DIR="/usr/local/etc/memexd"
CONFIG_FILE="$CONFIG_DIR/config.toml"
DATA_DIR="/usr/local/var/lib/memexd"
LOG_DIR="/usr/local/var/log/memexd"
RUN_DIR="/usr/local/var/run"
LAUNCHD_PLIST="/Library/LaunchDaemons/$SERVICE_NAME.plist"

echo "Installing memexd as macOS LaunchDaemon..."

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

# Function to check if group exists
group_exists() {
    dscl . -read /Groups/"$1" >/dev/null 2>&1
}

# Function to check if user exists
user_exists() {
    dscl . -read /Users/"$1" >/dev/null 2>&1
}

# Create service group
echo "Creating service group..."
if ! group_exists "$SERVICE_GROUP"; then
    dscl . -create /Groups/"$SERVICE_GROUP"
    dscl . -create /Groups/"$SERVICE_GROUP" PrimaryGroupID $SERVICE_GID
    dscl . -create /Groups/"$SERVICE_GROUP" RealName "memexd service group"
    echo "Created group: $SERVICE_GROUP (GID: $SERVICE_GID)"
else
    echo "Group $SERVICE_GROUP already exists"
fi

# Create service user
echo "Creating service user..."
if ! user_exists "$SERVICE_USER"; then
    dscl . -create /Users/"$SERVICE_USER"
    dscl . -create /Users/"$SERVICE_USER" UserShell /usr/bin/false
    dscl . -create /Users/"$SERVICE_USER" RealName "memexd daemon"
    dscl . -create /Users/"$SERVICE_USER" UniqueID $SERVICE_UID
    dscl . -create /Users/"$SERVICE_USER" PrimaryGroupID $SERVICE_GID
    dscl . -create /Users/"$SERVICE_USER" NFSHomeDirectory "$DATA_DIR"
    # Hide user from login window
    dscl . -create /Users/"$SERVICE_USER" IsHidden 1
    echo "Created user: $SERVICE_USER (UID: $SERVICE_UID)"
else
    echo "User $SERVICE_USER already exists"
fi

# Create directories
echo "Creating directories..."
mkdir -p "$CONFIG_DIR" "$DATA_DIR" "$LOG_DIR" "$RUN_DIR"

# Set ownership and permissions
chown root:wheel "$CONFIG_DIR"
chmod 755 "$CONFIG_DIR"

chown "$SERVICE_USER:$SERVICE_GROUP" "$DATA_DIR"
chmod 750 "$DATA_DIR"

chown "$SERVICE_USER:$SERVICE_GROUP" "$LOG_DIR"
chmod 750 "$LOG_DIR"

chown root:wheel "$RUN_DIR"
chmod 755 "$RUN_DIR"

# Create configuration file if it doesn't exist
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Creating default configuration..."
    cat > "$CONFIG_FILE" << 'EOF'
# memexd Daemon Configuration for macOS LaunchDaemon

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
log_file_path = "/usr/local/var/log/memexd/memexd.log"
json_format = false
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

# Stop existing service if running
if launchctl list | grep -q "$SERVICE_NAME"; then
    echo "Stopping existing service..."
    launchctl unload "$LAUNCHD_PLIST" 2>/dev/null || true
    sleep 2
fi

# Create LaunchDaemon plist
echo "Creating LaunchDaemon plist..."
cat > "$LAUNCHD_PLIST" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$SERVICE_NAME</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>$BINARY_PATH</string>
        <string>--config</string>
        <string>$CONFIG_FILE</string>
        <string>--pid-file</string>
        <string>$RUN_DIR/memexd.pid</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>$DATA_DIR</string>
    
    <key>StandardOutPath</key>
    <string>$LOG_DIR/memexd.log</string>
    
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/memexd.error.log</string>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
        <key>Crashed</key>
        <true/>
    </dict>
    
    <key>ThrottleInterval</key>
    <integer>10</integer>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>ProcessType</key>
    <string>Background</string>
    
    <key>UserName</key>
    <string>$SERVICE_USER</string>
    
    <key>GroupName</key>
    <string>$SERVICE_GROUP</string>
    
    <key>Nice</key>
    <integer>10</integer>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
    
    <key>SoftResourceLimits</key>
    <dict>
        <key>NumberOfProcesses</key>
        <integer>100</integer>
        <key>NumberOfFiles</key>
        <integer>1024</integer>
    </dict>
</dict>
</plist>
EOF

chown root:wheel "$LAUNCHD_PLIST"
chmod 644 "$LAUNCHD_PLIST"
echo "Created LaunchDaemon plist: $LAUNCHD_PLIST"

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

# Load and start the service
echo "Loading LaunchDaemon..."
launchctl load "$LAUNCHD_PLIST"

# Wait a moment and check status
sleep 3
if launchctl list "$SERVICE_NAME" >/dev/null 2>&1; then
    SERVICE_INFO=$(launchctl list "$SERVICE_NAME")
    echo "✅ Service installed and loaded successfully!"
    echo ""
    echo "Service Details:"
    echo "$SERVICE_INFO"
    echo ""
    echo "Management Commands:"
    echo "  Start:   sudo launchctl start $SERVICE_NAME"
    echo "  Stop:    sudo launchctl stop $SERVICE_NAME"
    echo "  Restart: sudo launchctl stop $SERVICE_NAME && sudo launchctl start $SERVICE_NAME"
    echo "  Status:  sudo launchctl list $SERVICE_NAME"
    echo "  Logs:    tail -f $LOG_DIR/memexd.log"
    echo "  Errors:  tail -f $LOG_DIR/memexd.error.log"
    echo ""
    echo "Configuration:"
    echo "  Config:  $CONFIG_FILE"
    echo "  Data:    $DATA_DIR"
    echo "  Logs:    $LOG_DIR"
    echo "  Plist:   $LAUNCHD_PLIST"
    
    # Check if service is actually running
    if echo "$SERVICE_INFO" | grep -q '"PID"'; then
        echo ""
        echo "✅ Service is running!"
    else
        echo ""
        echo "⚠️  Service loaded but may not be running. Check logs:"
        echo "  tail $LOG_DIR/memexd.error.log"
    fi
else
    echo "❌ Service installed but failed to load. Check system log:"
    echo "  sudo log show --predicate 'subsystem == \"com.apple.launchd\"' --last 5m"
    exit 1
fi

echo ""
echo "Installation complete!"