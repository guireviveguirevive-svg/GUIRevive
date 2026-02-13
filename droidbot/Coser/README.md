## COSER

COSER (Comprehensive Semantic Repair) is an automated Android GUI test script repair tool. It uses external semantic (text, content-desc) and internal semantic (source code) information to match GUI elements across app versions, and navigates cross-page changes via a Transition Graph.

### Requirements

- Python 3.8+
- Java 8 (for javalang parsing)

### Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Project Structure

```
Coser/
├── start.py                       # Entry point
├── requirements.txt
└── Reproduce/
    ├── method_code_extractor.py   # Step 1: Extract method code -> MethodCode.xml
    ├── layout_analyzer.py         # Step 2: Map GUI elements to code -> ElementCode.xml
    ├── transition_analyzer.py     # Step 3: Build transition graph -> Transition.xml
    ├── code_match_handler.py      # Semantic matching (SentenceBERT + UniXcoder)
    └── model.bin                  # Fine-tuned UniXcoder weights
```

### Pipeline

```
Source Code
    │
    ▼
Step 1: MethodCodeExtractor
    Input:  source code directory
    Output: MethodCode.xml  (<method, code>)
    │
    ▼
Step 2: LayoutAnalyzer
    Input:  source code + MethodCode.xml
    Output: ElementCode.xml  (<element, attributes, code>)
    │
    ▼
Step 3: TransitionAnalyzer  (new version only)
    Input:  source code + ElementCode.xml
    Output: Transition.xml  (<Activity_A, Activity_B, trigger>)
    │
    ▼
Step 4: Element Matching
    Stage 1: External Semantic Matching (text/content-desc similarity via SentenceBERT)
    Stage 2: Internal Semantic Matching (code similarity via UniXcoder)
    │
    ▼
Step 5: Navigation (if target is on a different page)
    Uses Transition Graph to find path from current to target activity
    │
    ▼
Step 6: Output trace.json
```

### Usage

Each repair case requires the following input data:

```
your_case/
├── record_output_{base}/        # DroidBot recording on the base version
│   ├── events/                  #   event_*.json (user actions)
│   ├── states/                  #   state_*.json + screenshots
│   └── xmls/                    #   xml_*.xml (UI hierarchy dumps)
├── replay_output_{new}_for_{base}/  # DroidBot replay on the updated version (failed)
│   ├── events/
│   ├── states/
│   └── xmls/
└── source_code/
    ├── {base_version}/          # Java/Kotlin source of the base version
    └── {new_version}/           # Java/Kotlin source of the updated version
```

Run:

```bash
python start.py \
  --record_path <record_output_dir> \
  --replay_path <replay_output_dir> \
  --old_source <base_version_source> \
  --new_source <updated_version_source>
```

Parameters:
- `--record_path`: Recording output directory (base version)
- `--replay_path`: Failed replay output directory (updated version)
- `--old_source`: Source code of the base version
- `--new_source`: Source code of the updated version
- `--output_dir`: (optional) Directory for intermediate XML files, can be reused across runs
- `--result_dir`: (optional) Directory for results (trace.json + repair.log)

Output: `coser_output_{case_name}/` containing `trace.json` and `repair.log`

### Example

An example case is included in `Reproduce/failed_test_case_1/` (PermissionManagerX v1.11 → v1.14):

```bash
python start.py \
  --record_path Reproduce/failed_test_case_1/record_output_v1_11_run3 \
  --replay_path Reproduce/failed_test_case_1/replay_output_v1_14_PMX_v1_14_run3_for_v1_11 \
  --old_source Reproduce/failed_test_case_1/source_code/v1_11 \
  --new_source Reproduce/failed_test_case_1/source_code/v1_14
```

Results will be written to `Reproduce/failed_test_case_1/coser_output_v1_14_PMX_v1_14_run3_for_v1_11/`.

### Output Format

`trace.json`:

```json
{
  "repair_success": true,
  "failed_event_number": 3,
  "original_element": {
    "class": "android.widget.CheckBox",
    "resource_id": "com.mirfatif.permissionmanagerx:id/action_dark_theme",
    "text": null
  },
  "matching": {
    "method": "internal_semantic",
    "score": 0.7822
  },
  "navigation": {
    "required": false,
    "steps": []
  },
  "matched_element": {
    "class": "MenuItem",
    "resource_id": "com.mirfatif.permissionmanagerx:id/action_adb",
    "text": "adb_menu_item",
    "content_description": "ADB Access",
    "activity": "Main"
  }
}
```

