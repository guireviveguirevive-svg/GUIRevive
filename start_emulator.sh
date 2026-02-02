#!/bin/bash
set -u

export ANDROID_HOME=~/android-sdk
export ANDROID_SDK_ROOT=$ANDROID_HOME
export PATH=$ANDROID_HOME/cmdline-tools/latest/bin:$PATH
export PATH=$PATH:$ANDROID_HOME/platform-tools
export PATH=$PATH:$ANDROID_HOME/emulator

# ===== Configuration =====
AVD_NAME="Android10.0"
START_PORT=5554
NUM_EMULATORS=${1:-16}
LOG_DIR="./emulator_logs"
MAX_RETRY=3

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] $*"; }

cleanup_device() {
  local port=$1
  adb -s emulator-$port emu kill 2>/dev/null || true
  pkill -f "emulator.*-port $port" 2>/dev/null || true
}

cleanup_all() {
  log "Cleaning up existing emulators..."
  for ((i=0; i<NUM_EMULATORS; i++)); do
    port=$((START_PORT + i*2))
    cleanup_device "$port"
  done
  pkill -f "emulator.*-avd $AVD_NAME" 2>/dev/null || true
  if [[ "${1:-}" == "from_trap" ]]; then
    exit 0
  fi
}
trap 'cleanup_all from_trap' SIGINT SIGTERM

check_avd() {
  if ! avdmanager list avd | grep -q "Name: $AVD_NAME"; then
    echo "Error: AVD '$AVD_NAME' does not exist"
    exit 1
  fi
}

start_emulator() {
  local port=$1
  emulator -avd "$AVD_NAME" \
    -port "$port" \
    -no-window \
    -read-only \
    -no-snapshot \
    -no-snapshot-save \
    -no-boot-anim \
    -accel on \
    -gpu swiftshader_indirect \
    >/dev/null 2>&1 &
  echo $!
}

wait_for_device() {
  local device_serial=$1
  local timeout_sec=${2:-120}
  if timeout "$timeout_sec" adb -s "$device_serial" wait-for-device 2>/dev/null; then
    return 0
  else
    return 1
  fi
}

wait_for_boot() {
  local device_serial=$1
  local timeout_sec=${2:-180}
  local start=$(date +%s)
  while (( $(date +%s) - start < timeout_sec )); do
    local boot_completed
    boot_completed=$(adb -s "$device_serial" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r\n')
    if [[ "$boot_completed" == "1" ]]; then
      return 0
    fi
    sleep 2
  done
  return 1
}

try_boot_emulator() {
  local port=$1
  local device="emulator-$port"
  local attempt=1

  while (( attempt <= MAX_RETRY )); do
    log "[$device] Startup attempt #$attempt ..."
    cleanup_device "$port"
    sleep 2
    start_emulator "$port" >/dev/null
    sleep 3

    if ! wait_for_device "$device" 180; then
      log "[$device] Connection failed (attempt #$attempt)"
      attempt=$((attempt+1))
      continue
    fi

    if wait_for_boot "$device" 300; then
      log "[$device] Startup successful (attempt #$attempt)"
      echo "$device" >> "$LOG_DIR/ready_devices.txt"
      return 0
    else
      log "[$device] Startup timeout (attempt #$attempt)"
    fi

    attempt=$((attempt+1))
  done

  log "[$device] Startup failed (after $MAX_RETRY retries)"
  echo "$device" >> "$LOG_DIR/failed_devices.txt"
  return 1
}

main() {
  check_avd
  cleanup_all || true
  sleep 2

  : > "$LOG_DIR/ready_devices.txt"
  : > "$LOG_DIR/failed_devices.txt"

  local start_time=$(date +%s)
  log "Starting $NUM_EMULATORS emulators..."

  # Start concurrently
  for ((i=0; i<NUM_EMULATORS; i++)); do
    port=$((START_PORT + i*2))
    try_boot_emulator "$port" &
    sleep 1
  done
  
  # ✅ Key improvement: Wait for all background tasks to complete
  log "Waiting for all emulators to start..."
  wait

  local end_time=$(date +%s)
  local total_time=$((end_time - start_time))

  echo
  log "=== Startup Completion Report ==="
  log "End time: $(date)"
  log "Total time: ${total_time} seconds"
  
  local success_count
  success_count=$(wc -l < "$LOG_DIR/ready_devices.txt" 2>/dev/null || echo 0)
  local fail_count
  fail_count=$(wc -l < "$LOG_DIR/failed_devices.txt" 2>/dev/null || echo 0)
  
  log "Successfully ready: $success_count/$NUM_EMULATORS"
  log "Failed: $fail_count/$NUM_EMULATORS"

  if (( success_count > 0 )); then
    echo "Ready devices:"
    cat "$LOG_DIR/ready_devices.txt"
  fi
  if (( fail_count > 0 )); then
    echo "Failed devices:"
    cat "$LOG_DIR/failed_devices.txt"
  fi
  echo

  log "Log directory: $LOG_DIR/"
  
  if (( success_count == NUM_EMULATORS )); then
    log "✓ All $NUM_EMULATORS emulators started successfully"
    exit 0
  elif (( success_count > 0 )); then
    log "⚠ Partially successful: $success_count/$NUM_EMULATORS"
    exit 0  # Partial success also returns 0, allowing Python to continue
  else
    log "✗ All emulators failed to start"
    exit 1
  fi
}

main "$@"