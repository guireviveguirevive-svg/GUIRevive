# GUIRevive

GUIRevive is an automated GUI test migration and repair tool for Android applications. When app UI changes across versions break existing test cases, GUIRevive automatically identifies and repairs the broken test steps using a combination of rule-based matching, visual similarity analysis, and LLM-based semantic understanding.

## Overview

The toolkit consists of three main components:

1. **Data Collection** -- Record and replay GUI test cases across different app versions to identify broken test steps
2. **Test Case Repair (GUIRevive)** -- Automatically repair broken test cases using a 3-tier matching strategy
3. **Baseline: Guider** -- A baseline repair approach (ISSTA 2021) for comparison

## Prerequisites

- Python 3.8+
- Android SDK with emulator
- Android Virtual Device (AVD) configured (default: `Android10.0`)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
GUIRevive/
├── start.py                        # Single APK runner entry point
├── start_bash.py                   # Batch data collection runner (record/replay)
├── start_repair_bash.py            # Batch repair runner (GUIRevive)
├── start_repair_bash_guider.py     # Batch repair runner (Guider baseline)
├── droidbot/
│   ├── droidbot.py                 # Core DroidBot engine
│   ├── input_policy.py             # Policy implementations (Replay/Matching/Guider)
│   └── UIMatch/
│       ├── Matcher.py              # 3-tier matching algorithm
│       ├── Parser.py               # UI hierarchy parser
│       └── utils.py                # Image similarity, LLM calls
└── All_cases.csv                   # Full test case list
```

## 1. Data Collection

The data collection pipeline records GUI interactions on a base app version and replays them on different versions to detect UI migration failures.

### Three Modes

| Mode | Description | Output Directory |
|------|-------------|-----------------|
| `record` | Random exploration on base version | `record_output_{app}_run{N}/` |
| `replay_original` | Replay on same version (sanity check) | `replay_output_{app}_run{N}/` |
| `replay_new` | Replay on new version (detect failures) | `replay_output_{new_app}_run{N}_for_{base_app}/` |

### Usage

**Step 1: Record** -- Capture test sequences on base app versions.

```bash
python3 start_bash.py record \
    --csv-file "droidbot/select_apks/<package>.csv" \
    --apk-base "droidbot/select_apks/<package>" \
    --max-parallel 8 \
    --run-count 3 \
    --parent-dir <package>
```

**Step 2: Replay on Original Version** -- Verify recordings are reproducible.

```bash
python3 start_bash.py replay_original \
    --csv-file "droidbot/select_apks/<package>.csv" \
    --apk-base "droidbot/select_apks/<package>" \
    --max-parallel 8 \
    --run-count 3 \
    --parent-dir <package>
```

**Step 3: Replay on New Versions** -- Detect failures caused by UI changes.

```bash
python3 start_bash.py replay_new \
    --csv-file "droidbot/select_apks/<package>.csv" \
    --apk-base "droidbot/select_apks/<package>" \
    --max-parallel 8 \
    --run-count 3 \
    --parent-dir <package>
```

After replay, an HTML report is generated at `html_report/matching_report.html` listing all failed test cases.

### Output Structure

Each output directory contains:

```
record_output_{app}_run{N}/
├── states/              # Screenshots of each UI state
├── events/              # Recorded UI events (JSON)
├── xmls/                # UI hierarchy XML dumps
├── views/               # View hierarchy data
├── user_input.txt       # Event sequence
└── index.html           # Visual report
```

## 2. Test Case Repair (GUIRevive)

GUIRevive repairs failed test cases using a **3-tier matching strategy** to find equivalent UI elements in the updated app version:

| Priority | Method | Description |
|----------|--------|-------------|
| 1 | **Exact Matching** | Match by `resource-id`, `content-desc`, or `text` attributes |
| 2 | **Similarity Matching** | Combine visual similarity (SSIM), attribute similarity, and spatial similarity |
| 3 | **LLM-based Matching** | Use LLM with vision capabilities to semantically match UI elements |

### Usage

```bash
python3 start_repair_bash.py \
    --apk-base /path/to/<package> \
    --max-parallel 4
```

The script automatically:
1. Parses `html_report/matching_report.html` to identify failed cases
2. For each failed case, attempts repair using the 3-tier matching strategy
3. Outputs results to `repair_output_{new_app}_run{N}_for_{base_app}/`

### Single Case Repair

```bash
python3 start.py \
    -a <apk_path> \
    -o <repair_output_dir> \
    -replay_output <record_output_dir> \
    -failed_replay_output <failed_replay_dir> \
    -policy matching \
    -count 100 \
    -is_emulator \
    -d emulator-5554
```

### Ablation Experiments

GUIRevive supports ablation flags to evaluate individual components:

```bash
python3 start_repair_bash.py \
    --apk-base /path/to/<package> \
    --without_taxonomy       # Disable UI change taxonomy
    --without_rule           # Disable rule-based matching
    --without_llm            # Disable LLM-based matching
    --without_next_screen_summary   # Disable next screen summary
    --without_history_summary       # Disable exploration history summary
```

## 3. Baseline: Guider

Guider is a baseline approach from ISSTA 2021 that uses a simpler 3-tier matching strategy for comparison:

| Type | Description |
|------|-------------|
| **alpha-typed (Sure match)** | Unique identity property match |
| **beta-typed (Close match)** | Multiple identity matches, ranked by visual similarity |
| **gamma-typed (Remote match)** | No identity match, visual similarity only |

Unlike GUIRevive, Guider does **not** use LLM-based matching.

### Usage

```bash
python3 start_repair_bash_guider.py \
    --apk-base /path/to/<package> \
    --max-parallel 4
```

Output is saved to `repair_output_{new_app}_run{N}_for_{base_app}_guider/`.

## Dataset

The full dataset of 735 test cases across 25 open-source Android apps is listed in `All_cases.csv`:

| Column | Description |
|--------|-------------|
| Server | Server where data is stored |
| App | Android package name |
| Record App | Base app version (recorded) |
| Replay App | Target app version (replayed) |
| Run Count | Test run index |

Each test case includes three directories:
- `record_output_*` -- Original recorded test sequence
- `replay_output_*` -- Failed replay on the new version
- `repair_output_*` -- Ground truth repair result
