"""
COSER - GUI Test Script Repair Tool
Complete repair pipeline for a single failed test case.

Usage:
    python start.py --record_path <path> --replay_path <path>
                    --old_source <path> --new_source <path>
                    [--output trace.json]

Input:
    - record_path: Path to the original recording output (base version)
    - replay_path: Path to the failed replay output (updated version)
    - old_source: Path to the base version source code
    - new_source: Path to the updated version source code

Output:
    - trace.json: Complete repair trace with matched element info
"""

import os
import sys
import json
import glob
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any
from contextlib import contextmanager
from datetime import datetime

# Add Reproduce directory to path
REPRODUCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Reproduce')
sys.path.insert(0, REPRODUCE_DIR)


class TeeLogger:
    """Write output to both console and log file."""

    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output."""
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout


class COSERRepair:
    """Main class for COSER test script repair."""

    def __init__(self, record_path: str, replay_path: str,
                 old_source: str = None, new_source: str = None,
                 output_dir: str = None):
        """
        Initialize COSER repair.

        Args:
            record_path: Path to record output (base version)
            replay_path: Path to replay output (updated version)
            old_source: Path to base version source code
            new_source: Path to updated version source code
            output_dir: Directory for intermediate outputs
        """
        self.record_path = os.path.abspath(record_path)
        self.replay_path = os.path.abspath(replay_path)
        self.old_source = os.path.abspath(old_source) if old_source else None
        self.new_source = os.path.abspath(new_source) if new_source else None

        # Output directory for intermediate files
        if output_dir:
            self.output_dir = os.path.abspath(output_dir)
        else:
            self.output_dir = os.path.join(REPRODUCE_DIR, 'output')
        os.makedirs(self.output_dir, exist_ok=True)

        # Failed event information
        self.failed_event_number = None
        self.failed_event_json = None
        self.target_widget = None

        # XML trees
        self.record_xml_tree = None
        self.replay_xml_tree = None

        # Static analysis outputs
        self.old_method_code_xml = None
        self.old_element_code_xml = None
        self.old_transition_xml = None
        self.new_method_code_xml = None
        self.new_element_code_xml = None
        self.new_transition_xml = None

        # Parsed data
        self.old_element_code_map = {}  # resource_id -> {element_info, code_snippets}
        self.new_element_code_map = {}
        self.old_text_code_map = {}  # textContent (lowercase) -> {element_info, code_snippets}
        self.new_text_code_map = {}
        self.transition_graph = {}  # activity -> [(target_activity, trigger_element, method)]

        # Matcher
        self.matcher = None

    def _print_step(self, step_num: int, title: str):
        """Print a step header."""
        print(f"\n{'=' * 60}")
        print(f"Step {step_num}: {title}")
        print('=' * 60)

    # =========================================================================
    # Step 1: Load Failed Event
    # =========================================================================
    def load_failed_event(self) -> Dict:
        """Load the failed event information from replay and record paths."""
        self._print_step(1, "Loading failed event")

        # Find max event number in replay
        replay_events_dir = os.path.join(self.replay_path, "events")
        event_files = glob.glob(os.path.join(replay_events_dir, "event_*.json"))

        if not event_files:
            raise ValueError(f"No event files found in {replay_events_dir}")

        event_numbers = []
        for f in event_files:
            basename = os.path.basename(f)
            num = int(basename.split('_')[1].split('.')[0])
            event_numbers.append(num)

        max_replay_event = max(event_numbers)
        self.failed_event_number = max_replay_event + 1

        print(f"  Last successful replay event: {max_replay_event}")
        print(f"  Failed event number: {self.failed_event_number}")

        # Load failed event from record
        failed_event_path = os.path.join(
            self.record_path, "events", f"event_{self.failed_event_number}.json"
        )

        if not os.path.exists(failed_event_path):
            raise FileNotFoundError(f"Failed event not found: {failed_event_path}")

        with open(failed_event_path, 'r', encoding='utf-8') as f:
            self.failed_event_json = json.load(f)

        if 'event' in self.failed_event_json and 'view' in self.failed_event_json['event']:
            self.target_widget = self.failed_event_json['event']['view']

        print(f"  Event type: {self.failed_event_json.get('event', {}).get('event_type', 'unknown')}")
        if self.target_widget:
            print(f"  Target widget class: {self.target_widget.get('class', 'unknown')}")
            print(f"  Target widget resource_id: {self.target_widget.get('resource_id', 'None')}")
            print(f"  Target widget text: {self.target_widget.get('text', 'None')}")
            print(f"  Target widget content_desc: {self.target_widget.get('content_description', 'None')}")

        return self.failed_event_json

    # =========================================================================
    # Step 2: Load XML Trees
    # =========================================================================
    def load_xml_trees(self):
        """Load XML trees for matching."""
        self._print_step(2, "Loading XML trees")

        # Record XML: xml_{failed_event_number - 1}.xml
        record_xml_path = os.path.join(
            self.record_path, "xmls", f"xml_{self.failed_event_number - 1}.xml"
        )
        if os.path.exists(record_xml_path):
            self.record_xml_tree = ET.parse(record_xml_path)
            print(f"  Loaded record XML: xml_{self.failed_event_number - 1}.xml")

        # Replay XML: the last xml
        replay_xmls_dir = os.path.join(self.replay_path, "xmls")
        xml_files = glob.glob(os.path.join(replay_xmls_dir, "xml_*.xml"))
        if xml_files:
            def get_xml_num(path):
                return int(os.path.basename(path).split('_')[1].split('.')[0])
            xml_files.sort(key=get_xml_num)
            self.replay_xml_tree = ET.parse(xml_files[-1])
            print(f"  Loaded replay XML: {os.path.basename(xml_files[-1])}")

    # =========================================================================
    # Step 3: Run Static Analysis
    # =========================================================================
    def _parse_version_from_path(self, path: str, is_record: bool = True) -> str:
        """
        Parse version name from record/replay path.

        Examples:
        - record_output_v1_11_run3 -> v1_11
        - replay_output_v1_14_PMX_v1_14_run3_for_v1_11 -> v1_14_PMX_v1_14
        """
        import re
        basename = os.path.basename(path.rstrip('/'))

        if is_record:
            # record_output_{version}_run{N}
            match = re.match(r'record_output_(.+)_run\d+', basename)
            if match:
                return match.group(1)
        else:
            # replay_output_{version}_run{N}_for_{base_version}
            match = re.match(r'replay_output_(.+)_run\d+_for_.+', basename)
            if match:
                return match.group(1)

        # Fallback to basename
        return basename

    def run_static_analysis(self):
        """Run MethodCodeExtractor, LayoutAnalyzer, TransitionAnalyzer on source code."""
        self._print_step(3, "Running static analysis")

        if not self.old_source or not self.new_source:
            print("  Skipping: Source code paths not provided")
            return False

        if not os.path.exists(self.old_source):
            print(f"  Warning: Old source not found: {self.old_source}")
            return False
        if not os.path.exists(self.new_source):
            print(f"  Warning: New source not found: {self.new_source}")
            return False

        # Extract version names from record/replay paths
        old_version = self._parse_version_from_path(self.record_path, is_record=True)
        new_version = self._parse_version_from_path(self.replay_path, is_record=False)

        print(f"  Old version: {old_version}")
        print(f"  New version: {new_version}")

        # Run analysis on both versions
        # OLD version: only need MethodCode + ElementCode (for Internal Semantic Matching)
        # NEW version: need all three (MethodCode + ElementCode + Transition for navigation)
        print("\n  [3.1] Analyzing OLD version (MethodCode + ElementCode only)...")
        self.old_method_code_xml, self.old_element_code_xml, self.old_transition_xml = \
            self._run_analysis_pipeline(self.old_source, old_version, skip_transition=True)

        print("\n  [3.2] Analyzing NEW version (full analysis)...")
        self.new_method_code_xml, self.new_element_code_xml, self.new_transition_xml = \
            self._run_analysis_pipeline(self.new_source, new_version, skip_transition=False)

        # Parse ElementCode.xml to build element-code maps (id-based and text-based)
        if self.old_element_code_xml and os.path.exists(self.old_element_code_xml):
            self.old_element_code_map, self.old_text_code_map = self._parse_element_code_xml(self.old_element_code_xml)
            print(f"  Loaded {len(self.old_element_code_map)} elements from old ElementCode.xml")
            print(f"  Loaded {len(self.old_text_code_map)} text-indexed elements from old ElementCode.xml")

        if self.new_element_code_xml and os.path.exists(self.new_element_code_xml):
            self.new_element_code_map, self.new_text_code_map = self._parse_element_code_xml(self.new_element_code_xml)
            print(f"  Loaded {len(self.new_element_code_map)} elements from new ElementCode.xml")
            print(f"  Loaded {len(self.new_text_code_map)} text-indexed elements from new ElementCode.xml")

        # Parse Transition.xml to build transition graph
        if self.new_transition_xml and os.path.exists(self.new_transition_xml):
            self.transition_graph = self._parse_transition_xml(self.new_transition_xml)
            print(f"  Loaded {len(self.transition_graph)} activities in transition graph")

        return True

    def _run_analysis_pipeline(self, source_path: str, app_name: str,
                                skip_transition: bool = False) -> Tuple[str, str, str]:
        """Run the complete analysis pipeline on a source code directory."""
        method_code_xml = None
        element_code_xml = None
        transition_xml = None

        try:
            # Step 1: MethodCodeExtractor
            print(f"    Running MethodCodeExtractor...")
            method_code_xml = self._run_method_code_extractor(source_path, app_name)

            if method_code_xml and os.path.exists(method_code_xml):
                # Step 2: LayoutAnalyzer
                print(f"    Running LayoutAnalyzer...")
                element_code_xml = self._run_layout_analyzer(source_path, method_code_xml, app_name)

                if element_code_xml and os.path.exists(element_code_xml) and not skip_transition:
                    # Step 3: TransitionAnalyzer (only for new version)
                    print(f"    Running TransitionAnalyzer...")
                    transition_xml = self._run_transition_analyzer(source_path, element_code_xml, app_name)

        except Exception as e:
            print(f"    Error in analysis pipeline: {e}")
            import traceback
            traceback.print_exc()

        return method_code_xml, element_code_xml, transition_xml

    def _run_method_code_extractor(self, source_path: str, app_name: str) -> Optional[str]:
        """Run MethodCodeExtractor on source code."""
        try:
            output_path = os.path.join(self.output_dir, f"{app_name}_MethodCode.xml")

            # Import the module
            import method_code_extractor as mce

            # Get source files (suppress verbose output)
            with suppress_stdout():
                java_files, kotlin_files = mce.get_source_files(source_path)
            source_files = [(f, p, 'java') for f, p in java_files] + \
                          [(f, p, 'kotlin') for f, p in kotlin_files]

            print(f"      Found {len(java_files)} Java files, {len(kotlin_files)} Kotlin files")

            if not source_files:
                return None

            # Generate output (suppress verbose output)
            with suppress_stdout():
                mce.generate_method_code_xml(source_files, output_path)
            print(f"      Output: {os.path.basename(output_path)}")

            return output_path

        except Exception as e:
            print(f"      Error: {e}")
            return None

    def _run_layout_analyzer(self, source_path: str, method_code_xml: str,
                             app_name: str) -> Optional[str]:
        """Run LayoutAnalyzer on source code."""
        try:
            output_path = os.path.join(self.output_dir, f"{app_name}_ElementCode.xml")

            from layout_analyzer import AndroidLayoutAnalyzer, write_xml

            # Find resource path (with AndroidManifest.xml)
            resource_path = self._find_resource_path(source_path)
            if not resource_path:
                print(f"      Warning: Could not find AndroidManifest.xml in {source_path}")
                return None

            print(f"      Resource path: {resource_path}")

            # Create analyzer and run (suppress verbose output)
            with suppress_stdout():
                analyzer = AndroidLayoutAnalyzer(resource_path, method_code_xml, source_path)
                analyzer.build_layout_list_by_activity()
                root = analyzer.build_ui_elements()

            # Print diagnostic info
            print(f"      Found {len(analyzer.activity_list)} activities, "
                  f"{len(analyzer.layout_names)} layouts, "
                  f"{len(analyzer.menu_names)} menus")

            # Count UIElements in the output
            ui_element_count = len(root.findall('.//UIElement')) if root is not None else 0
            print(f"      Generated {ui_element_count} UIElements")

            # Write output
            with suppress_stdout():
                write_xml(root, output_path)
            print(f"      Output: {os.path.basename(output_path)}")

            return output_path

        except Exception as e:
            print(f"      Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _run_transition_analyzer(self, source_path: str, element_code_xml: str,
                                  app_name: str) -> Optional[str]:
        """Run TransitionAnalyzer on source code."""
        try:
            output_path = os.path.join(self.output_dir, f"{app_name}_Transition.xml")

            from transition_analyzer import AndroidTransitionAnalyzer, find_android_paths

            # Find resource path (suppress verbose output)
            with suppress_stdout():
                resource_path, pkg = find_android_paths(source_path)
            if not resource_path:
                print(f"      Warning: Could not find resource path")
                return None

            # Get method code xml path
            method_code_xml = os.path.join(
                self.output_dir, f"{app_name}_MethodCode.xml"
            )

            # Create analyzer and run (suppress verbose output)
            with suppress_stdout():
                analyzer = AndroidTransitionAnalyzer(
                    element_code_path=element_code_xml,
                    app_name=app_name,
                    resource_path=resource_path,
                    code_path=source_path,
                    pkg=pkg,
                    method_code_path=method_code_xml
                )

                # Analyze
                analyzer.analyze_activity_transition()
                analyzer.analyze_implicit_intents()
                analyzer.analyze_explicit_intents_from_all_methods()
                analyzer.analyze_more_options()
                analyzer.analyze_side_bar()

                # Generate output
                analyzer.generate_xml(output_path)
            print(f"      Output: {os.path.basename(output_path)}")

            return output_path

        except Exception as e:
            print(f"      Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _find_resource_path(self, source_path: str) -> Optional[str]:
        """Find the path containing AndroidManifest.xml (main app module)."""
        # Common locations for standard projects
        candidates = [
            source_path,
            os.path.join(source_path, 'app', 'src', 'main'),
            os.path.join(source_path, 'src', 'main'),
            os.path.join(source_path, 'app'),
        ]

        for path in candidates:
            manifest_path = os.path.join(path, 'AndroidManifest.xml')
            if os.path.exists(manifest_path):
                return path

        # For multi-module projects, find all AndroidManifest.xml files
        # and prefer the one with LAUNCHER activity (main app module)
        all_manifests = []
        for root, dirs, files in os.walk(source_path):
            if 'AndroidManifest.xml' in files:
                # Skip test directories
                if 'androidTest' in root or 'test' in root.split(os.sep):
                    continue
                all_manifests.append(root)

        if not all_manifests:
            return None

        # If only one manifest found, return it
        if len(all_manifests) == 1:
            return all_manifests[0]

        # Find the main app module (with LAUNCHER intent filter)
        for manifest_dir in all_manifests:
            manifest_path = os.path.join(manifest_dir, 'AndroidManifest.xml')
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Main app has LAUNCHER category
                    if 'android.intent.category.LAUNCHER' in content:
                        return manifest_dir
            except Exception:
                pass

        # Fallback: prefer paths with 'src/main' pattern
        for manifest_dir in all_manifests:
            if 'src/main' in manifest_dir or 'src\\main' in manifest_dir:
                # Also check it has res/layout
                layout_dir = os.path.join(manifest_dir, 'res', 'layout')
                if os.path.exists(layout_dir):
                    return manifest_dir

        # Last fallback: return first one with res/layout
        for manifest_dir in all_manifests:
            layout_dir = os.path.join(manifest_dir, 'res', 'layout')
            if os.path.exists(layout_dir):
                return manifest_dir

        # Ultimate fallback
        return all_manifests[0]

    def _parse_element_code_xml(self, xml_path: str) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """Parse ElementCode.xml to extract element-code mappings.

        Returns:
            Tuple of (id_map, text_map):
            - id_map: resource_id -> element info
            - text_map: textContent -> element info (for elements without id)
        """
        id_map = {}
        text_map = {}
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for activity in root.findall('.//Activity'):
                activity_name = activity.get('name', '')

                for ui_element in activity.findall('.//UIElement'):
                    element_id = ui_element.get('id', '')
                    text = ui_element.get('text', '')
                    text_content = ui_element.get('textContent', '')

                    # Collect all method bodies
                    code_snippets = []
                    for method in ui_element.findall('.//method'):
                        body = method.get('body', '')
                        if body:
                            code_snippets.append(body)

                    element_info = {
                        'activity': activity_name,
                        'id': element_id,
                        'text': text,
                        'textContent': text_content,
                        'tag': ui_element.get('tag', ''),
                        'code_snippets': code_snippets,
                        'internal_semantic': '\n'.join(code_snippets)
                    }

                    # Index by id if available
                    if element_id:
                        id_map[element_id] = element_info

                    # Also index by textContent for text-based lookup
                    # This helps match elements without resource_id (like menu items)
                    if text_content and code_snippets:
                        # Use textContent as key (normalized)
                        text_key = text_content.strip().lower()
                        if text_key and text_key not in text_map:
                            text_map[text_key] = element_info

        except Exception as e:
            print(f"      Error parsing ElementCode.xml: {e}")

        return id_map, text_map

    def _parse_transition_xml(self, xml_path: str) -> Dict[str, List[Dict]]:
        """Parse Transition.xml to build transition graph."""
        graph = {}
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for path in root.findall('.//Path'):
                source = path.find('source')
                target = path.find('target')
                method = path.find('method')

                if source is not None and target is not None:
                    source_name = source.get('name', '')
                    target_name = target.get('name', '')
                    method_name = method.get('name', '') if method is not None else ''

                    if source_name not in graph:
                        graph[source_name] = []

                    graph[source_name].append({
                        'target': target_name,
                        'method': method_name
                    })

        except Exception as e:
            print(f"      Error parsing Transition.xml: {e}")

        return graph

    # =========================================================================
    # Step 4: Element Matching
    # =========================================================================
    def match_element(self) -> Tuple[Optional[Dict], str, float]:
        """Match the target element using External + Internal semantics."""
        self._print_step(4, "Matching target element")

        if self.target_widget is None:
            print("  Error: No target widget to match")
            return None, "none", 0.0

        # Initialize matcher
        from code_match_handler import CodeMatchHandler
        self.matcher = CodeMatchHandler(
            external_threshold=0.7,
            use_sentence_bert=True,
            use_unixcoder=True
        )

        # Parse current screen elements
        current_elements = self._parse_xml_elements(self.replay_xml_tree)
        print(f"  Current screen has {len(current_elements)} elements")

        # Prepare old element
        old_element = self._prepare_old_element()
        old_semantic = self.matcher._get_external_semantic(old_element)
        print(f"  Target element external semantic: '{old_semantic}'")

        # Add internal semantic from ElementCode
        target_resource_id = self.target_widget.get('resource_id', '')
        print(f"  Target resource_id: '{target_resource_id}'")
        print(f"  Old ElementCode map has {len(self.old_element_code_map)} keys")

        # Debug: show some keys from old_element_code_map
        if self.old_element_code_map:
            sample_keys = list(self.old_element_code_map.keys())[:5]
            print(f"  Sample keys from old ElementCode: {sample_keys}")

        # Try exact match first (by resource_id)
        if target_resource_id and target_resource_id in self.old_element_code_map:
            old_element['internal_semantic'] = \
                self.old_element_code_map[target_resource_id].get('internal_semantic', '')
            print(f"  Target element has internal semantic: {len(old_element.get('internal_semantic', ''))} chars")
        else:
            # Try partial match (just the id part)
            print(f"  Exact match not found, trying partial match...")
            found_match = False
            if target_resource_id:
                target_id_part = target_resource_id.split('/')[-1] if '/' in target_resource_id else target_resource_id
                for key in self.old_element_code_map.keys():
                    key_id_part = key.split('/')[-1] if '/' in key else key
                    if target_id_part and key_id_part == target_id_part:
                        old_element['internal_semantic'] = \
                            self.old_element_code_map[key].get('internal_semantic', '')
                        print(f"  Found via partial match: '{key}'")
                        print(f"  Target element has internal semantic: {len(old_element.get('internal_semantic', ''))} chars")
                        found_match = True
                        break

            # If no resource_id or no match found, try text-based lookup
            if not found_match:
                target_text = self.target_widget.get('text', '')
                if target_text and self.old_text_code_map:
                    text_key = target_text.strip().lower()
                    print(f"  Trying text-based lookup with key: '{text_key}'")
                    print(f"  Old text_code_map has {len(self.old_text_code_map)} keys")

                    # Debug: show some keys from old_text_code_map
                    if self.old_text_code_map:
                        sample_text_keys = list(self.old_text_code_map.keys())[:5]
                        print(f"  Sample keys from old text_code_map: {sample_text_keys}")

                    if text_key in self.old_text_code_map:
                        old_element['internal_semantic'] = \
                            self.old_text_code_map[text_key].get('internal_semantic', '')
                        print(f"  Found via text-based match: '{text_key}'")
                        print(f"  Target element has internal semantic: {len(old_element.get('internal_semantic', ''))} chars")
                        found_match = True
                    else:
                        # Try partial text match (substring)
                        for key in self.old_text_code_map.keys():
                            if text_key in key or key in text_key:
                                old_element['internal_semantic'] = \
                                    self.old_text_code_map[key].get('internal_semantic', '')
                                print(f"  Found via partial text match: '{key}'")
                                print(f"  Target element has internal semantic: {len(old_element.get('internal_semantic', ''))} chars")
                                found_match = True
                                break

                if not found_match:
                    print(f"  No internal semantic found for target element (tried id and text)")

        # Initialize models
        print("  Initializing matching models...")
        self.matcher.initialize_models()

        # Stage 1: External Semantic Matching (current screen)
        print("\n  [Stage 1] External Semantic Matching (current screen)...")
        matched, score = self.matcher.get_external_semantic_match(old_element, current_elements)

        if matched and score >= self.matcher.external_threshold:
            print(f"  Match found! Score: {score:.4f}")
            return matched, "external_semantic", score

        print(f"  No match above threshold (best score: {score:.4f})")

        # Stage 2: Internal Semantic Matching (all elements in new version)
        print("\n  [Stage 2] Internal Semantic Matching (all elements)...")
        print(f"  Old element has internal_semantic: {bool(old_element.get('internal_semantic'))}")
        print(f"  New element_code_map size: {len(self.new_element_code_map)}")
        print(f"  New text_code_map size: {len(self.new_text_code_map)}")

        if old_element.get('internal_semantic') and (self.new_element_code_map or self.new_text_code_map):
            # Prepare all new elements with internal semantics
            all_new_elements = []
            seen_ids = set()

            # Add elements from id-based map
            for elem_id, elem_info in self.new_element_code_map.items():
                all_new_elements.append({
                    'resource-id': elem_id,
                    'text': elem_info.get('text', ''),
                    'content-desc': elem_info.get('textContent', ''),
                    'class': elem_info.get('tag', ''),
                    'internal_semantic': elem_info.get('internal_semantic', ''),
                    'activity': elem_info.get('activity', '')
                })
                seen_ids.add(elem_id)

            # Add elements from text-based map (that aren't already included)
            for text_key, elem_info in self.new_text_code_map.items():
                elem_id = elem_info.get('id', '')
                if elem_id and elem_id in seen_ids:
                    continue  # Already added from id-based map
                all_new_elements.append({
                    'resource-id': elem_id or f'text:{text_key}',  # Use text as identifier if no id
                    'text': elem_info.get('text', ''),
                    'content-desc': elem_info.get('textContent', ''),
                    'class': elem_info.get('tag', ''),
                    'internal_semantic': elem_info.get('internal_semantic', ''),
                    'activity': elem_info.get('activity', '')
                })

            print(f"  Searching in {len(all_new_elements)} elements from ElementCode (id + text maps)...")

            internal_matched, internal_score = self.matcher.get_internal_semantic_match(
                old_element, all_new_elements
            )

            if internal_matched:
                print(f"  Match found via internal semantics!")
                print(f"  Matched element: {internal_matched.get('resource-id', 'N/A')}")
                print(f"  Located in activity: {internal_matched.get('activity', 'unknown')}")
                print(f"  Internal semantic score: {internal_score:.4f}")

                # Print code snippets for comparison
                print(f"\n  --- Old Element Code (internal semantic) ---")
                old_code = old_element.get('internal_semantic', '')
                if old_code:
                    # Truncate if too long, show first 500 chars
                    if len(old_code) > 500:
                        print(f"  {old_code[:500]}...")
                        print(f"  ... (total {len(old_code)} chars)")
                    else:
                        print(f"  {old_code}")
                else:
                    print(f"  (no code)")

                print(f"\n  --- Matched Element Code (internal semantic) ---")
                matched_code = internal_matched.get('internal_semantic', '')
                if matched_code:
                    if len(matched_code) > 500:
                        print(f"  {matched_code[:500]}...")
                        print(f"  ... (total {len(matched_code)} chars)")
                    else:
                        print(f"  {matched_code}")
                else:
                    print(f"  (no code)")

                return internal_matched, "internal_semantic", internal_score

        print("  No match found via internal semantics")

        # Fallback: return best external match
        if matched:
            print(f"\n  [Fallback] Returning best external match (score: {score:.4f})")
            return matched, "external_semantic_fallback", score

        return None, "none", 0.0

    def _prepare_old_element(self) -> Dict:
        """Prepare old element dict for matching."""
        return {
            'text': self.target_widget.get('text', '') or '',
            'content-desc': self.target_widget.get('content_description', '') or '',
            'resource-id': self.target_widget.get('resource_id', '') or '',
            'class': self.target_widget.get('class', '') or '',
            'bounds': self.target_widget.get('bounds', []),
        }

    def _parse_bounds_string(self, bounds_str: str) -> Dict:
        """
        Parse bounds string like '[0,0][1080,1920]' to dict format.

        Returns:
            {'x': int, 'y': int, 'width': int, 'height': int}
        """
        import re
        match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds_str)
        if match:
            x1, y1, x2, y2 = map(int, match.groups())
            return {
                'x': x1,
                'y': y1,
                'width': x2 - x1,
                'height': y2 - y1
            }
        return {}

    def _parse_xml_elements(self, xml_tree: ET.ElementTree) -> List[Dict]:
        """Parse XML tree and extract all GUI elements."""
        if xml_tree is None:
            return []

        elements = []
        for node in xml_tree.getroot().iter():
            if 'bounds' not in node.attrib:
                continue

            bounds_str = node.attrib.get('bounds', '')
            bounds_dict = self._parse_bounds_string(bounds_str)

            elements.append({
                'class': node.attrib.get('class', ''),
                'text': node.attrib.get('text', ''),
                'content-desc': node.attrib.get('content-desc', ''),
                'resource-id': node.attrib.get('resource-id', ''),
                'bounds': bounds_dict,  # 现在是字典格式
                'bounds_str': bounds_str,  # 保留原始字符串
                'clickable': node.attrib.get('clickable', 'false') == 'true',
                'enabled': node.attrib.get('enabled', 'false') == 'true',
            })

        return elements

    # =========================================================================
    # Step 5: Find Navigation Path
    # =========================================================================
    def find_navigation_path(self, matched_element: Dict) -> List[Dict]:
        """Find navigation path using Transition Graph."""
        self._print_step(5, "Finding navigation path")

        if not matched_element:
            print("  No matched element, skipping navigation")
            return []

        target_activity = matched_element.get('activity', '')
        if not target_activity:
            print("  Matched element on current screen (same page)")
            return []

        if not self.transition_graph:
            print("  No transition graph available")
            return []

        # Get current activity from replay state
        current_activity = self._get_current_activity()
        if not current_activity:
            print("  Could not determine current activity")
            return []

        print(f"  Current activity: {current_activity}")
        print(f"  Target activity: {target_activity}")

        # Check if same activity (using normalized names)
        if self._activities_match(current_activity, target_activity):
            print("  Same activity (normalized), no navigation needed")
            return []

        # BFS to find path (use normalized names)
        path = self._find_path_bfs(current_activity, target_activity)

        if path:
            print(f"  Found navigation path with {len(path)} steps:")
            for i, step in enumerate(path):
                print(f"    {i+1}. {step['from']} -> {step['to']} via {step['method']}")
        else:
            print("  No path found in transition graph")

        return path

    def _get_current_activity(self) -> Optional[str]:
        """Get current activity from the failed event's event_str."""
        # Parse activity from event_str like:
        # "TouchEvent(state=xxx, view=xxx(MainActivity/CheckBox-))"
        if self.failed_event_json:
            event_str = self.failed_event_json.get('event_str', '')
            # Extract activity name from pattern like (ActivityName/ClassName-)
            # Find the last occurrence of (Word/ pattern
            import re
            matches = re.findall(r'\((\w+)/', event_str)
            if matches:
                return matches[-1]  # Return the last match (the activity name)
        return None

    def _extract_simple_activity_name(self, name: str) -> str:
        """
        Extract simple class name from full activity component name.

        Examples:
        - "com.mirfatif.permissionmanagerx/.fwk.MainActivityM" -> "MainActivityM"
        - "MainActivityM" -> "MainActivityM"
        - "com.pkg.MainActivity" -> "MainActivity"
        """
        # Handle component name format: pkg/.relative.ClassName
        if '/' in name:
            name = name.split('/')[-1]
        # Remove leading dot if present
        if name.startswith('.'):
            name = name[1:]
        # Get the last part after the last dot (the class name)
        if '.' in name:
            name = name.split('.')[-1]
        return name

    def _activities_match(self, activity1: str, activity2: str) -> bool:
        """Check if two activity names refer to the same activity."""
        if not activity1 or not activity2:
            return False
        # Extract simple class names and compare directly
        simple1 = self._extract_simple_activity_name(activity1)
        simple2 = self._extract_simple_activity_name(activity2)
        return simple1 == simple2

    def _find_graph_key(self, activity_name: str) -> Optional[str]:
        """Find the matching key in transition graph using normalized names."""
        for key in self.transition_graph.keys():
            if self._activities_match(key, activity_name):
                return key
        return None

    def _find_path_bfs(self, start: str, end: str) -> List[Dict]:
        """BFS to find shortest path in transition graph."""
        from collections import deque

        if self._activities_match(start, end):
            return []

        # Find the actual graph key that matches our start activity
        start_key = self._find_graph_key(start)
        if not start_key:
            # Start activity not in transition graph, can't navigate from here
            return []

        visited = {start_key}
        visited_normalized = {self._extract_simple_activity_name(start_key)}
        queue = deque([(start_key, [])])

        while queue:
            current, path = queue.popleft()

            if current not in self.transition_graph:
                continue

            for edge in self.transition_graph[current]:
                target = edge['target']
                target_normalized = self._extract_simple_activity_name(target)

                if self._activities_match(target, end):
                    return path + [{
                        'from': current,
                        'to': target,
                        'method': edge['method']
                    }]

                if target_normalized not in visited_normalized:
                    visited.add(target)
                    visited_normalized.add(target_normalized)
                    queue.append((target, path + [{
                        'from': current,
                        'to': target,
                        'method': edge['method']
                    }]))

        return []

    # =========================================================================
    # Step 6: Generate Trace
    # =========================================================================
    def generate_trace(self, matched_element: Dict, matching_method: str,
                       score: float, navigation_path: List[Dict]) -> Dict:
        """Generate the final repair trace."""
        self._print_step(6, "Generating repair trace")

        trace = {
            "repair_success": matched_element is not None,
            "failed_event_number": self.failed_event_number,
            "original_element": {
                "class": self.target_widget.get('class', ''),
                "resource_id": self.target_widget.get('resource_id', ''),
                "text": self.target_widget.get('text'),
                "content_description": self.target_widget.get('content_description'),
                "bounds": self.target_widget.get('bounds', []),
            },
            "matching": {
                "method": matching_method,
                "score": round(score, 4),
            },
            "navigation": {
                "required": len(navigation_path) > 0,
                "steps": navigation_path,
            },
            "matched_element": None,
        }

        if matched_element:
            trace["matched_element"] = {
                "class": matched_element.get('class', ''),
                "resource_id": matched_element.get('resource-id', ''),
                "text": matched_element.get('text', ''),
                "content_description": matched_element.get('content-desc', ''),
                "bounds": matched_element.get('bounds', ''),
                "activity": matched_element.get('activity', ''),
            }

        return trace

    # =========================================================================
    # Main Repair Process
    # =========================================================================
    def repair(self) -> Dict:
        """Run the complete repair process."""
        print("\n" + "=" * 60)
        print("COSER GUI Test Script Repair")
        print("=" * 60)
        print(f"Record path: {self.record_path}")
        print(f"Replay path: {self.replay_path}")
        if self.old_source:
            print(f"Old source: {self.old_source}")
        if self.new_source:
            print(f"New source: {self.new_source}")

        # Step 1: Load failed event
        self.load_failed_event()

        # Step 2: Load XML trees
        self.load_xml_trees()

        # Step 3: Run static analysis
        self.run_static_analysis()

        # Step 4: Match element
        matched_element, matching_method, score = self.match_element()

        # Step 5: Find navigation path
        navigation_path = []
        if matched_element:
            navigation_path = self.find_navigation_path(matched_element)

        # Step 6: Generate trace
        trace = self.generate_trace(matched_element, matching_method, score, navigation_path)

        # Print summary
        print("\n" + "=" * 60)
        print("REPAIR SUMMARY")
        print("=" * 60)
        print(f"  Success: {trace['repair_success']}")
        print(f"  Matching method: {trace['matching']['method']}")
        print(f"  Matching score: {trace['matching']['score']}")
        print(f"  Navigation required: {trace['navigation']['required']}")

        if trace['matched_element']:
            print(f"\n  Matched Element:")
            print(f"    class: {trace['matched_element']['class']}")
            print(f"    resource_id: {trace['matched_element']['resource_id']}")
            print(f"    text: {trace['matched_element']['text']}")
            if trace['matched_element']['activity']:
                print(f"    activity: {trace['matched_element']['activity']}")

        return trace


def _parse_case_name_from_replay(replay_path: str) -> str:
    """
    Parse case name from replay path for output folder naming.

    Examples:
    - replay_output_v1_14_PMX_v1_14_run3_for_v1_11 -> v1_14_PMX_v1_14_run3_for_v1_11
    """
    import re
    basename = os.path.basename(replay_path.rstrip('/'))
    # Remove 'replay_output_' prefix if present
    if basename.startswith('replay_output_'):
        return basename[14:]  # len('replay_output_') = 14
    return basename


def main():
    parser = argparse.ArgumentParser(description='COSER - GUI Test Script Repair Tool')
    parser.add_argument('--record_path', required=True, help='Path to record output (base version)')
    parser.add_argument('--replay_path', required=True, help='Path to replay output (updated version)')
    parser.add_argument('--old_source', default=None, help='Path to base version source code')
    parser.add_argument('--new_source', default=None, help='Path to updated version source code')
    parser.add_argument('--output_dir', default=None, help='Directory for intermediate XML outputs (can be reused)')
    parser.add_argument('--result_dir', default=None, help='Directory for case results (log + trace.json)')

    args = parser.parse_args()

    # Determine result directory (for log and trace.json)
    case_name = _parse_case_name_from_replay(args.replay_path)
    if args.result_dir:
        result_dir = args.result_dir
    else:
        # Default: coser_output_{case_name} in same parent directory as input
        # e.g., Reproduce/failed_test_case_1/record_output_xxx -> Reproduce/failed_test_case_1/coser_output_xxx
        parent_dir = os.path.dirname(os.path.abspath(args.replay_path.rstrip('/')))
        result_dir = os.path.join(parent_dir, f'coser_output_{case_name}')

    os.makedirs(result_dir, exist_ok=True)

    # Set up logging
    log_path = os.path.join(result_dir, 'repair.log')
    trace_path = os.path.join(result_dir, 'trace.json')

    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        print(f"Case: {case_name}")
        print(f"Result dir: {result_dir}")
        print(f"Log file: {log_path}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Create repairer and run
        repairer = COSERRepair(
            record_path=args.record_path,
            replay_path=args.replay_path,
            old_source=args.old_source,
            new_source=args.new_source,
            output_dir=args.output_dir
        )

        trace = repairer.repair()

        # Save trace (convert numpy types to Python native types for JSON serialization)
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types."""
            import numpy as np
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

        with open(trace_path, 'w', encoding='utf-8') as f:
            json.dump(trace, f, indent=2, ensure_ascii=False, default=convert_to_serializable)

        print(f"\nTrace saved to: {trace_path}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return 0 if trace['repair_success'] else 1

    finally:
        # Restore stdout and close log file
        if logger:
            sys.stdout = logger.terminal
            logger.close()
            print(f"Results saved to: {result_dir}/")


if __name__ == "__main__":
    sys.exit(main())
