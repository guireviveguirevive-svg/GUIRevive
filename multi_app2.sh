#!/bin/bash

CSV_DIR="droidbot/select_apks"


for csv in "$CSV_DIR"/*.csv; do
    pkg=$(basename "$csv" .csv)  # Extract package name
    echo "============================"
    echo " Processing $pkg"
    echo "============================"
    
    # Replay new
    python3 start_bash.py replay_new \
        --csv-file "$csv" \
        --apk-base "$CSV_DIR/$pkg" \
        --max-parallel 8 \
        --run-count 3 \
        --parent-dir "$pkg"
done
