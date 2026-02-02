#!/bin/bash
export API_KEY="sk-proj-x"

target_apps=(
    "com.activitymanager"
)


for pkg in "${target_apps[@]}"; do
    echo "=========================================="
    echo "Processing: $pkg"
    echo "=========================================="
    # Repair
    python3 start_repair_bash.py \
    --apk-base "$pkg" \
    --max-parallel 5
    sleep 5
done
