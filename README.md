# GUIRevive
GUIRevive is an automated approach for repairing obsolete GUI tests caused by UI evolution in mobile applications.

## Overview

This repository is organized into two parts: **Data** and **Tool**.

- [**Dataset**](#1-dataset) -- 736 obsolete test cases across 36 open-source Android apps (`All_cases.csv`)
- [**GUIRevive**](#3-test-case-repair-guirevive) -- Our automated approach for repairing obsolete GUI tests using semantic-aware widget localization, functionality-preserving validation, and goal-guided exploration

## Prerequisites

- Python 3.8+
- Android SDK with emulator
- Android Virtual Device (AVD) configured (default: `Android10.0`)

Install dependencies:

```bash
pip install -r requirements.txt
```

Download and prepare the data:

1. **Obsolete Test Cases**: Download from [Zenodo](https://zenodo.org/records/18538552) and extract to the project root
2. **Historical APKs**: Download from [Google Drive](https://drive.google.com/file/d/1aMffj_-6WWQbdI_xV02fYKKpmYKbfSem/view?usp=drive_link) and extract to `droidbot/select_apks/`

## Project Structure

```
GUIRevive/
├── start.py                        # Single APK runner entry point
├── start_bash.py                   # Batch data collection runner (record/replay)
├── start_repair_bash.py            # Batch repair runner (GUIRevive)
├── droidbot/
│   ├── droidbot.py                 # Core DroidBot engine
│   ├── input_policy.py             # Policy implementations (Obsolete Test Collection / GUIRevive / Guider baseline)
│   ├── UIMatch/                    # Semantic-Aware Widget Localization module in GUIRevive
│   └── Coser/                      # Coser baseline
├── All_cases.csv                   # Full obsolete case list
└── prompts.md                      # LLM prompt catalog for GUIRevive
```

## 1. Dataset

The full dataset of 736 obsolete test cases across 36 open-source Android apps is listed in `All_cases.csv`:

| Column | Description |
|--------|-------------|
| App | Android package name |
| Record App | Base app version (recorded) |
| Replay App | Target app version (replayed) |
| Run Count | Test run index |
| Mismatch | Type of UI mismatch |

Each test case includes three directories:
- `record_output_*` -- Original recorded test sequence
- `replay_output_*` -- Failed replay on the new version
- `repair_output_*` -- Ground truth repair result

The 36 apps and their GitHub repositories are listed below:

| App | Category | GitHub |
|-----|----------|--------|
| app.familygem | People Manager | [FamilyGem](https://github.com/michelesalvador/FamilyGem) |
| com.quran.labs.androidquran | Bible Reader | [quran_android](https://github.com/quran/quran_android) |
| de.markusfisch.android.libra | Decision Maker | [Libra](https://github.com/markusfisch/Libra) |
| code.name.monkey.retromusic | Music | [RetroMusicPlayer](https://github.com/RetroMusicPlayer/RetroMusicPlayer) |
| universe.constellation.orion.viewer | File Manager | [orion-viewer](https://github.com/max-kammerer/orion-viewer) |
| me.hackerchick.catima | Wallet | [Catima](https://github.com/CatimaLoyalty/Android) |
| org.billthefarmer.editor | Text Editor | [editor](https://github.com/billthefarmer/editor) |
| com.mkulesh.micromath.plus | Calculator | [microMathematics](https://github.com/mkulesh/microMathematics) |
| io.github.muntashirakon.AppManager | App Manager | [AppManager](https://github.com/MuntashirAkon/AppManager) |
| com.atul.musicplayer | Music | [music_player_lite](https://github.com/ap-atul/music_player_lite) |
| org.zephyrsoft.trackworktime | Schedule | [trackworktime](https://github.com/mathisdt/trackworktime) |
| com.vrem.wifianalyzer | WiFi Analyzer | [WiFiAnalyzer](https://github.com/VREMSoftwareDevelopment/WiFiAnalyzer) |
| org.isoron.uhabits | Habit Tracker | [uhabits](https://github.com/iSoron/uhabits) |
| com.mirfatif.permissionmanagerx | Permission Manager | [PermissionManagerX](https://github.com/mirfatif/PermissionManagerX) |
| me.tsukanov.counter | Counter | [counter](https://github.com/gentlecat/counter) |
| org.michaelbel.moviemade | Movie | [movies](https://github.com/michaelbel/movies) |
| free.rm.skytube.oss | Player | [SkyTube](https://github.com/SkyTubeTeam/SkyTube) |
| com.jlindemann.science | Chemistry | [Atomic-Periodic-Table.Android](https://github.com/JLindemann42/Atomic-Periodic-Table.Android) |
| org.secuso.privacyfriendlynotes | Notes | [privacy-friendly-notes](https://github.com/SecUSo/privacy-friendly-notes) |
| eu.faircode.email | Email | [FairEmail](https://github.com/M66B/FairEmail) |
| com.michaldrabik.showly2 | Movie | [showly](https://github.com/michaldrabik/showly) |
| com.mxt.anitrend | Reader | [anitrend-app](https://github.com/AniTrend/anitrend-app) |
| com.best.deskclock | Clock | [Clock](https://github.com/BlackyHawky/Clock) |
| me.zhanghai.android.files | File Manager | [MaterialFiles](https://github.com/zhanghai/MaterialFiles) |
| org.gateshipone.odyssey | Music | [odyssey](https://github.com/gateship-one/odyssey) |
| com.arn.scrobble | Music | [InnerTune](https://github.com/z-huang/InnerTune) |
| de.grobox.liberario | Transport | [Transportr](https://github.com/grote/Transportr) |
| com.kgurgul.cpuinfo | System Info | [cpu-info](https://github.com/kamgurgul/cpu-info) |
| com.github.anrimian.musicplayer | Music | [music-player](https://github.com/Anrimian/music-player) |
| com.activitymanager | Activity Manager | [ActivityManager](https://github.com/sdex/ActivityManager) |
| hu.vmiklos.plees_tracker | Sleep Manager | [plees-tracker](https://github.com/vmiklos/plees-tracker) |
| com.amaze.filemanager | File Manager | [AmazeFileManager](https://github.com/TeamAmaze/AmazeFileManager) |
| org.secuso.privacyfriendlytodolist | To-Do List | [privacy-friendly-todo-list](https://github.com/SecUSo/privacy-friendly-todo-list) |
| org.billthefarmer.diary | Diary | [diary](https://github.com/billthefarmer/diary) |
| it.feio.android.omninotes | Notes | [Omni-Notes](https://github.com/federicoiosue/Omni-Notes) |
| xyz.zedler.patrick.tack | Beat Manager | [tack-android](https://github.com/patzly/tack-android) |

Historical APK versions for each app are available in [Google Drive](https://drive.google.com/file/d/1aMffj_-6WWQbdI_xV02fYKKpmYKbfSem/view?usp=drive_link). Extract to `droidbot/select_apks/` with the following structure:

```
droidbot/select_apks/
├── {app_package_name}.csv       # Version list with download links
└── {app_package_name}/
    ├── {version_1}.apk
    ├── {version_2}.apk
    └── ...
```

Note: The CSV files list app versions from newest to oldest.

## 2. Obsolete Test Collection

If you want to collect new obsolete test cases, you can use our data collection tool to record GUI interactions on a base app version and replay them on different versions.

### Three Modes

| Mode | Description | Output Directory |
|------|-------------|-----------------|
| `record` | Random exploration on base version | `record_output_{base_app}_run{N}/` |
| `replay_original` | Replay on same version (sanity check) | `replay_output_{base_app}_run{N}/` |
| `replay_new` | Replay on new version (detect failures) | `replay_output_{new_app}_run{N}_for_{base_app}/` |

### Usage

**Step 1: Record** -- Capture test sequences on base app versions.

```bash
python3 start_bash.py record \
    --csv-file "droidbot/select_apks/<app_name>.csv" \
    --apk-base "droidbot/select_apks/<app_name>" \
    --max-parallel 8 \
    --run-count 3 \
    --parent-dir <app_name>
```

`--run-count 3` performs 3 random explorations, each with 100 events by default. Use `--count` to change the number of events per run.

**Step 2: Replay on Original Version** -- Verify recordings are reproducible.

```bash
python3 start_bash.py replay_original \
    --csv-file "droidbot/select_apks/<app_name>.csv" \
    --apk-base "droidbot/select_apks/<app_name>" \
    --max-parallel 8 \
    --run-count 3 \
    --parent-dir <app_name>
```

**Step 3: Replay on New Versions** -- Detect failures caused by UI changes.

```bash
python3 start_bash.py replay_new \
    --csv-file "droidbot/select_apks/<app_name>.csv" \
    --apk-base "droidbot/select_apks/<app_name>" \
    --max-parallel 8 \
    --run-count 3 \
    --parent-dir <app_name>
```

A test case is marked as obsolete if the replay event count is less than the original recording event count.

### Output Structure

Each output directory (`record_output_*`, `replay_output_*`) contains:

```
{record,replay}_output_{base_app}_run{N} or replay_output_{new_app}_run{N}_for_{base_app}/
├── states/              # Screenshots of each UI state
├── events/              # Recorded UI events (JSON)
├── xmls/                # UI hierarchy XML dumps
├── views/               # View hierarchy data
├── user_input.txt       # Event sequence
└── index.html           # Visual report
```

## 3. GUIRevive

GUIRevive automatically repairs obsolete GUI test cases by locating the corresponding widget on the updated app version. It reads test cases from `All_cases.csv` and outputs repaired event sequences.

For details on the LLM prompts used in GUIRevive's three-stage pipeline, see [`prompts.md`](prompts.md).

### Usage

```bash
python3 start_repair_bash.py \
    --apk-base <app_name> \
    --repair-output-dir-suffix _guirevive \
    --max-parallel 4
```

- `--apk-base`: App package name to process (e.g., `com.activitymanager`).
- `--repair-output-dir-suffix`: Suffix appended to output directory name. Use this to avoid overwriting existing `repair_output_*` directories (which contain ground truth data).
- `--max-parallel`: Number of parallel emulator instances.

### Output Structure

Output is saved to `{app_name}/repair_output_{new_app_version}_run{N}_for_{base_app_version}{suffix}/`:

```
repair_output_{new_app}_run{N}_for_{base_app}/
├── states/              # Screenshots during repair
├── events/              # UI events (JSON)
├── xmls/                # UI hierarchy XML dumps
├── views/               # View hierarchy data
├── user_input.txt       # Event sequence
├── index.html           # Visual report
└── exploration_tmp/     # Ground truth repair results
    ├── states/          # Repaired screenshots
    ├── events/          # Repaired event sequence
    ├── xmls/            # Repaired UI hierarchy XML dumps
    ├── images/          # Ground truth widget images (matched_view*.png)
    └── repair_logs/     # Repair traces (repair_trace_event_*.json)
                         # Contains "action": "match_found" with ground truth widget info
```

### Ablation Experiments

Example: disable semantic matching with `-without_llm`:

```bash
python3 start_repair_bash.py \
    --apk-base <app_name> \
    --repair-output-dir-suffix _without_llm \
    -without_llm \
    --max-parallel 4
```

Available ablation flags:
- `-without_taxonomy` — Disable role-aware validation
- `-without_rule` — Disable rule-based matching
- `-without_llm` — Disable semantic matching
- `-without_next_screen_summary` — Disable widget effect summarization
- `-without_history_summary` — Disable past exploration summarization

## 4. Baseline: Guider

[Guider](https://dl.acm.org/doi/10.1145/3460319.3464830) (ISSTA 2021) is a baseline approach that uses a heuristic method for repair.

### Usage

```bash
python3 start.py \
    -a droidbot/select_apks/com.activitymanager/5.4.7.apk \
    -o com.activitymanager/repair_output_5_4_7_run2_for_4_1_3_guider \
    -replay_output com.activitymanager/record_output_4_1_3_run2 \
    -failed_replay_output com.activitymanager/replay_output_5_4_7_run2_for_4_1_3 \
    -policy guider \
    -count 100 \
    -is_emulator \
    -d emulator-5554
```

- `-a`: New version APK path (the version to repair to).
- `-o`: Repair output directory.
- `-replay_output`: Original recording directory (`record_output_*`).
- `-failed_replay_output`: Failed replay directory (`replay_output_*`).

## 5. Baseline: Coser

[Coser](https://dl.acm.org/doi/abs/10.1145/3597503.3639108) (ICSE 2024) uses external semantic and internal semantic information for repair.

### Usage

```bash
python droidbot/Coser/start.py \
  --record_path com.activitymanager/record_output_4_1_3_run2 \
  --replay_path com.activitymanager/replay_output_5_4_7_run2_for_4_1_3 \
  --old_source /path/to/ActivityManager-4.1.3-source \
  --new_source /path/to/ActivityManager-5.4.7-source
```

- `--record_path`: Original recording directory (`record_output_*`).
- `--replay_path`: Failed replay directory (`replay_output_*`).
- `--old_source`: Source code directory of the base app version.
- `--new_source`: Source code directory of the new app version.

Output is saved to `coser_output_{case_name}/` containing `trace.json` and `repair.log`.
