#!/bin/bash

CSV_DIR="droidbot/select_apks"


for csv in "$CSV_DIR"/*.csv; do
    

    pkg=$(basename "$csv" .csv)  # Extract package name
    echo "============================"
    echo " Processing $pkg"
    echo "============================"

    # if [[ " ${skiped_apps[@]} " =~ " $pkg " ]]; then
    #     echo "Skipping $pkg because it is in the skiped_apps list"
    #     continue
    # fi

    # Record
    python3 start_bash.py record \
        --csv-file "$csv" \
        --apk-base "$CSV_DIR/$pkg" \
        --max-parallel 8 \
        --run-count 3 \
        --parent-dir "$pkg"

    # Replay original
    python3 start_bash.py replay_original \
        --csv-file "$csv" \
        --apk-base "$CSV_DIR/$pkg" \
        --max-parallel 8 \
        --run-count 3 \
        --parent-dir "$pkg"
done
