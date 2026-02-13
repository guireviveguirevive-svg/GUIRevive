"""
Parse test case directory, extract data pairs of original app and new version app
"""

import os
import re
import glob
import json
from lxml import etree as ET
from typing import Dict, List, Tuple, Optional
import concurrent.futures
import csv
from datetime import datetime
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Import matching module
from .Matcher import Matcher
from .utils import draw_element_on_image, read_image, compute_ssim
from .Logger import get_logger


class Parser:
    """Test case parser"""
    
    def __init__(self, dataset_dir: str, base_app_filter: str = None, run_count_filter: str = None):
        """
        Initialize the parser

        Args:
            dataset_dir: Test case directory, e.g. "test_case1"
            base_app_filter: Base app filter, e.g. "v6_4_1", used to only process test cases of a specific base version
            run_count_filter: Run count filter, e.g. "3", used to only process test cases of a specific run count
        """
        self.dataset_dir = dataset_dir
        self.base_app_filter = base_app_filter
        self.run_count_filter = run_count_filter
        self.test_cases_pairs = self.generate_test_cases_pairs()
        print("before deduplicate test_cases_pairs", len(self.test_cases_pairs))
        self.deduplicate_failed_events()
        print("after deduplicate test_cases_pairs", len(self.test_cases_pairs))
        
    
    def generate_test_cases_pairs(self) -> Dict[Tuple[str, str, str], Dict]:
        """
        1. Parse test case directory, extract folder pairs
        2. For each test case pair, find the failure point and retrieve the png, xml, events and other info
        """
        # Find all replay folders
        replay_folders = self.find_replay_folders()
        
        # Parse data pairs
        pairs = {}
        for replay_folder in replay_folders:
            replay_name = os.path.basename(replay_folder)
            replay_info = self.parse_replay_dir_name(replay_name)
            
            if not replay_info:
                continue
            
            # If run_count_filter is specified, only process matching ones
            if self.run_count_filter and str(replay_info['run_count']) != str(self.run_count_filter):
                continue

            # If base_app_filter is specified, only process matching ones
            if self.base_app_filter and str(replay_info['base_version']) != str(self.base_app_filter):
                continue
                
            # Derive the corresponding record folder
            record_name = self.derive_record_folder(replay_info)
            if not record_name:
                continue
                        
            # Create data pair - use triplet as key (base_version, new_version, run_count)
            pair_id = (replay_info['base_version'], replay_info['new_version'], replay_info['run_count'])
            
            record_path = os.path.join(self.dataset_dir, record_name)
            replay_events_count =  self.count_events(os.path.join(replay_folder, "events"))
            if replay_events_count == 100: # Indicates no failed events
                continue
            pairs[pair_id] = {
                "record_path": record_path,
                "replay_path": replay_folder,
                "replay_events_count":replay_events_count
            }
        
        return pairs
    
    def find_replay_folders(self) -> List[str]:
        """
        Find all replay_output_*_for_* folders under dataset_dir.
        If base_app_filter is provided (e.g. "v6_4_1"), only return directories matching *_for_{base_app_filter}.

        Returns:
            List of replay folder paths
        """
        pattern = os.path.join(self.dataset_dir, "replay_output_*_for_*")
        replay_folders = glob.glob(pattern)
        
        if self.base_app_filter:
            suffix = f"_for_{self.base_app_filter}"
            replay_folders = [p for p in replay_folders if os.path.basename(p).endswith(suffix)]
        
        return replay_folders
    
    def derive_record_folder(self, replay_info: Dict) -> Optional[str]:
        """
        Derive the corresponding record folder name from replay info.
        Example: for replay_info = {"new_version": "v6_6_2", "run_count": "3", "base_version": "v6_4_1"}
        returns "record_output_v6_4_1_run3"

        Args:
            replay_info: Replay info dictionary containing new_version, run_count, base_version

        Returns:
            Corresponding record folder name, or None if it cannot be derived
        """
        if not replay_info or 'base_version' not in replay_info or 'run_count' not in replay_info:
            return None
        
        base_app = replay_info['base_version']
        run_count = replay_info['run_count']
        
        # Build record folder name
        record_name = f"record_output_{base_app}_run{run_count}"
        
        # Check if it exists
        record_path = os.path.join(self.dataset_dir, record_name)
        if os.path.exists(record_path):
            return record_name
        
        return None
    
    def parse_replay_dir_name(self, dir_name: str) -> Optional[Dict]:
        """
        Parse replay directory name

        Args:
            dir_name: Directory name, e.g. "replay_output_V3_0_9-F-DROID_run3_for_V3_0_2-F-DROID"

        Returns:
            Parsed result, e.g. {"new_version": "V3_0_9-F-DROID", "run_count": "3", "base_version": "V3_0_2-F-DROID"}
        """
        match = re.match(r'^replay_output_(?P<new_version>.+?)_run(?P<run_count>\d+)_for_(?P<base_version>.+)$', dir_name)
        if match:
            return match.groupdict()
        return None
    
    def count_events(self, events_dir: str) -> int:
        """
        Count the number of .json files in the events directory

        Args:
            events_dir: Events directory path

        Returns:
            Number of events
        """
        if not os.path.isdir(events_dir):
            return 0
        
        json_paths = glob.glob(os.path.join(events_dir, "*.json"))
        return len(json_paths)
    
    def find_original_element(self, event_path, xml_tree) -> ET.Element:
        """
        Find the corresponding element in xml_tree based on information from the event

        Args:
            event_path: Event file path
            xml_tree: XML tree object

        Returns:
            The found element object, or None
        """
        import json
        
        # 1. Extract bounds and class info from the event file
        try:
            with open(event_path, 'r', encoding='utf-8') as f:
                event = json.load(f)
            
            # Extract target bounds and class
            verified_bounds = None
            verified_class = None
            verified_text = None
            verified_resource_id = None
            verified_content_description = None
            
            
            if 'event' in event and 'view' in event['event']:
                view = event['event']['view']
                if 'bounds' in view:
                    verified_bounds = view['bounds']
                if 'class' in view:
                    verified_class = view['class']
                if 'text' in view:
                    verified_text = view['text']
                if 'resource_id' in view:
                    verified_resource_id = view['resource_id']
                if 'content_description' in view:
                    verified_content_description = view['content_description']
            
                
            # Convert bounds to string format [x1,y1][x2,y2]
            verified_bounds_str = f"[{verified_bounds[0][0]},{verified_bounds[0][1]}][{verified_bounds[1][0]},{verified_bounds[1][1]}]"
            
        except Exception as e:
            print(f"Error reading event file: {e}")
            return None
        
        # 2. Find the node with matching attributes in the XML tree
        if xml_tree is None:
            print("XML tree is None")
            return None
            
        root = xml_tree.getroot()

        for node in root.iter():
            attrs = node.attrib

            def norm(v):
                # Normalize None or "" to None
                return v if v not in (None, "") else None

            current_bounds = norm(attrs.get('bounds'))
            current_class = norm(attrs.get('class'))
            current_text = norm(attrs.get('text'))
            current_resource_id = norm(attrs.get('resource-id'))
            current_content_desc = norm(attrs.get('content-desc'))

            match = True

            if verified_bounds_str is not None and current_bounds != verified_bounds_str:
                match = False
            if verified_class is not None and current_class != verified_class:
                match = False
            if verified_text is not None and current_text != verified_text:
                match = False
            if verified_resource_id is not None and current_resource_id != verified_resource_id:
                match = False
            if verified_content_description is not None and current_content_desc != verified_content_description:
                match = False

            if match:
                return node

        
        # Return the found element object
        return None
    
    
    def parse_xml_file(self, xml_path: str) -> Optional[ET.ElementTree]:
        """
        Parse XML file using lxml

        Args:
            xml_path: XML file path

        Returns:
            Parsed XML tree object, or None if parsing fails

        ElementTree = the tree wrapping the entire XML document
        Element = a node in the XML
        Each Element can have:
            - tag (node name)
            - attrib (attribute dictionary)
            - text (text content)
            - children (child nodes)

        lxml's ElementTree provides richer functionality than the standard library:
            - getparent(): get parent node
            - xpath(): supports more powerful XPath queries
            - getprevious()/getnext(): get previous/next sibling node
        """
        if not os.path.exists(xml_path):
            print(f"XML file does not exist: {xml_path}")
            return None
            
        result_tree = None
        
        try:
            # Use lxml parser, preserve comments
            parser = ET.XMLParser(remove_blank_text=True, remove_comments=False)
            result_tree = ET.parse(xml_path, parser)
        except Exception as e:
            print(f"Failed to parse XML file {xml_path}: {e}")
        
        return result_tree
    
    def process_single_pair(self, pair_data):
        """
        Process a single test case pair

        Args:
            pair_data: (pair_id, pair_info) tuple

        Returns:
            (pair_id, matching_result) tuple
        """
        pair_id, pair_info = pair_data
        failed_event_number = pair_info['replay_events_count'] + 1
        record_path = pair_info['record_path']
        replay_path = pair_info['replay_path']
        
        # Create independent log directory and logger for each process
        log_dir = os.path.join(self.dataset_dir, f"logs/logs_pair_{pair_id}")
        os.makedirs(log_dir, exist_ok=True)
        # Use a unique logger name and force creation of new handlers
        logger_name = f"UIMatch_pair_{pair_id}"
        logger = get_logger(name=logger_name, log_dir=log_dir, force_new_handlers=True)
        
        # find png + xml + event in record
        original_event = f"{record_path}/events/event_{failed_event_number}.json"
        original_png = f"{record_path}/states/screen_{failed_event_number-1}.png"
        original_xml = f"{record_path}/xmls/xml_{failed_event_number-1}.xml"
        original_tree = self.parse_xml_file(original_xml)
        original_element = self.find_original_element(original_event, original_tree)
        print(f"original_element: {original_element}")

        # find png + xml in replay
        replay_png = f"{replay_path}/states/screen_{failed_event_number-1}.png"
        replay_xml = f"{replay_path}/xmls/xml_{failed_event_number-1}.xml"
        replay_tree = self.parse_xml_file(replay_xml)

        # check if has success png in replay
        if os.path.exists(f"{replay_path}/states/screen_{failed_event_number-1}_success.png"):
            return pair_id, {
                "success": False,
                "matching_method": "Exists Success",
            }

        # Call the algorithm to match the original and replay
        matcher = Matcher(original_png, original_tree, original_element, replay_png, replay_tree, logger)
        matching_result = matcher.matching()
        logger.info(f"matching_result: {matching_result}")
        if matching_result["success"]:
            # Annotate in the replay png
            if matching_result['matched_element'] is not None:
                logger.info(f"matching_result: {matching_result['matched_element'].attrib}")
                replay_img = read_image(replay_png)
                replay_img = draw_element_on_image(replay_img, matching_result["matched_element"].attrib.get("bounds", ""))
                replay_img.save(replay_png.replace(".png", "_success.png"))
                # Convert lxml.etree._Attrib object to a regular Python dictionary
                matching_result['matched_element'] = dict(matching_result['matched_element'].attrib)
                # save as a json file for matched element
                with open(f"{replay_path}/matched_element_{failed_event_number-1}.json", "w", encoding="utf-8") as f:
                    json.dump(matching_result['matched_element'], f, ensure_ascii=False, indent=4)

                # save as a json file for matched method
                with open(f"{replay_path}/matched_method_{failed_event_number-1}.json", "w", encoding="utf-8") as f:
                    json.dump(matching_result['matching_method'], f, ensure_ascii=False, indent=4)

            else:
                matching_result['success'] = False

        
        return pair_id, matching_result
    
    def read_event_attributes(self, event_path: str) -> Dict:
        """
        Read event file
        """
        try:
            with open(event_path, 'r', encoding='utf-8') as f:
                event = json.load(f)
        except Exception as e:
            print(f"Error reading event file: {e}")
            return None
        
        bounds = ""
        class_name = ""
        text = ""
        resource_id = ""
        content_description = ""
        if 'event' in event and 'view' in event['event']:
                view = event['event']['view']
                if 'bounds' in view:
                    bounds = view['bounds']
                if 'class' in view:
                    class_name = view['class']
                if 'text' in view:
                    text = view['text']
                if 'resource_id' in view:
                    resource_id = view['resource_id']
                if 'content_description' in view:
                    content_description = view['content_description']
        
        result_str = f"{str(bounds)}|{class_name}|{text}|{resource_id}|{content_description}"
        return result_str
    
    def deduplicate_failed_events(self) -> List[Dict]:
        """
        Deduplicate failed events
        self.test_cases_pairs

        1. If base app, run count, and replay_events_count are the same, it means the failure occurred in the same scenario
        2. If new app and event content are the same, it means the failure occurred on the same element
        """

        visited_pairs_1 = set()
        cleaned_test_cases_pairs_1 = {}

        for pair_id, pair_info in self.test_cases_pairs.items():
            # paire_id: base_version, run_count, replay_events_count
            pair_key = (pair_id[0], pair_id[2], pair_info['replay_events_count'])

            if pair_key in visited_pairs_1:
                continue
            visited_pairs_1.add(pair_key)
            cleaned_test_cases_pairs_1[pair_id] = pair_info

        visited_pairs_2 = set()
        cleaned_test_cases_pairs_2 = {}
        for pair_id, pair_info in cleaned_test_cases_pairs_1.items():
            failed_event_number = pair_info['replay_events_count'] + 1
            current_event_path = f"{pair_info['record_path']}/events/event_{failed_event_number}.json"

            if failed_event_number == 101:
                continue # Indicates no failed events
            current_event_attributes_str = self.read_event_attributes(current_event_path)
            if current_event_attributes_str is None:
                continue
            new_version = pair_id[1]
            pair_key = (current_event_attributes_str, new_version)
            if pair_key in visited_pairs_2:
                continue
            visited_pairs_2.add(pair_key)
            cleaned_test_cases_pairs_2[pair_id] = pair_info
        self.test_cases_pairs = cleaned_test_cases_pairs_2

        
    
    def read_and_matching_failed_events_sequential(self) -> Dict:
        """
        Read failed events, process sequentially using a for loop (for debugging)

        Args:
            debug: Whether to enable verbose debug output

        Returns:
            Test case dictionary containing matching results
        """
        import time
        
        total_count = len(self.test_cases_pairs)
        print(f"Starting sequential processing of {total_count} test case pairs...")
        
        # Processing statistics
        success_count = 0
        error_count = 0
        start_time = time.time()
        
        # Iterate and process each test case
        for idx, (pair_id, pair_info) in enumerate(self.test_cases_pairs.items(), 1):
            try:
                # Display current test case info being processed
                record_app, replay_app, run_count = pair_id
                print(f"\n[{idx}/{total_count}] Processing: {record_app} -> {replay_app} (Run {run_count})")
                
                
                # Process test case
                process_start = time.time()
                processed_pair_id, matching_result = self.process_single_pair((pair_id, pair_info))
                process_time = time.time() - process_start
                
                # Update results
                self.test_cases_pairs[pair_id]['matching_result'] = matching_result
                
                # Display results
                success = matching_result.get('success', False)
                method = matching_result.get('matching_method', 'unknown') if success else 'failed'
                
                if success:
                    success_count += 1
                    print(f"Success: matched using method '{method}' (time: {process_time:.2f}s)")
                else:
                    print(f"Failed: {method} (time: {process_time:.2f}s)")
                
                # Display progress
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                remaining = avg_time * (total_count - idx)
                print(f"Progress: {idx}/{total_count} ({idx/total_count*100:.1f}%) - Elapsed: {elapsed:.1f}s - Estimated remaining: {remaining:.1f}s")
                
            except Exception as exc:
                error_count += 1
                print(f"Error occurred while processing test case {pair_id}:")
                import traceback
                traceback.print_exc()
                
                # Record the error but continue processing other test cases
                self.test_cases_pairs[pair_id]['matching_result'] = {
                    "success": False, 
                    "error": str(exc),
                    "matching_method": "error"
                }
        
        # Display summary
        total_time = time.time() - start_time
        print(f"\nProcessing complete, {total_count} test cases total, time: {total_time:.1f}s")
        print(f"Success: {success_count} ({success_count/total_count*100:.1f}%)")
        print(f"Failed: {total_count - success_count - error_count} ({(total_count - success_count - error_count)/total_count*100:.1f}%)")
        print(f"Errors: {error_count} ({error_count/total_count*100:.1f}%)")
        
        return self.test_cases_pairs
    
    
    def read_and_matching_failed_events_parallel(self, max_workers=None) -> Dict:
        """
        Read failed events, process in parallel using multithreading
        """

        if max_workers is None:
            max_workers = os.cpu_count() or 4

        total = len(self.test_cases_pairs)
        print(f"Starting multithreaded processing of {total} test case pairs using {max_workers} threads...")

        results_lock = threading.Lock()
        results = {}

        progress_queue = queue.Queue()
        SENTINEL = object()  # Exit signal

        def progress_monitor():
            completed = 0
            while True:
                try:
                    msg = progress_queue.get(timeout=1)
                except queue.Empty:
                    continue
                try:
                    if msg is SENTINEL:
                        # Mark and exit
                        return
                    print(msg)
                    if ("Processing complete" in msg) or ("Error occurred" in msg):
                        completed += 1
                        print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
                finally:
                    # Ensure task_done is called for every message
                    progress_queue.task_done()

        monitor_thread = threading.Thread(target=progress_monitor)
        monitor_thread.start()

        def process_pair(pair_id, pair_info):
            try:
                progress_queue.put(f"Processing pair_id: {pair_id}")
                processed_pair_id, matching_result = self.process_single_pair((pair_id, pair_info))
                with results_lock:
                    results[pair_id] = matching_result
                success = matching_result.get('success', False)
                method = matching_result.get('matching_method', 'unknown') if success else 'failed'
                progress_queue.put(f"Processing complete: pair_id={pair_id}, success={success}, method={method}")
                return pair_id, matching_result
            except Exception as exc:
                progress_queue.put(f"Error occurred: pair_id={pair_id}, err={exc}")
                progress_queue.put(traceback.format_exc())
                with results_lock:
                    results[pair_id] = {"success": False, "error": str(exc)}
                return pair_id, results[pair_id]

        # Execute and wait
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_pair, pid, info): pid for pid, info in self.test_cases_pairs.items()}
            for _ in as_completed(futures):
                pass  # Just ensure all tasks are completed

        # Update matching results in the original dictionary
        for pair_id, matching_result in results.items():
            self.test_cases_pairs[pair_id]['matching_result'] = matching_result

        # Notify the progress thread to exit and ensure all messages are consumed
        progress_queue.put(SENTINEL)
        progress_queue.join()
        monitor_thread.join()

        print(f"Multithreaded processing complete, processed {len(results)} test case pairs")
        return self.test_cases_pairs


    
    def generate_html_content(self, original_png_path, original_element, matched_element_png_path, matched_element) -> str:
        """
        Generate HTML content

        Args:
            original_png_path: Relative path of original PNG file in tmp_images
            original_element: Original element
            matched_element_png_path: Relative path of matched element PNG file in tmp_images
            matched_element: Matched element
        """
        # Safely get attributes to avoid KeyError
        original_resource_id = original_element.attrib.get('resource-id', 'N/A')
        matched_resource_id = matched_element.get('resource-id', 'N/A')

        original_text = original_element.attrib.get('text', 'N/A')
        matched_text = matched_element.get('text', 'N/A')

        original_content_description = original_element.attrib.get('content-desc', 'N/A')
        matched_content_description = matched_element.get('content-desc', 'N/A')

        original_class = original_element.attrib.get('class', 'N/A')
        matched_class = matched_element.get('class', 'N/A')

        # Add CSS styles
        css_style = """
        <style>
            .container {
                display: flex;
                justify-content: space-between;
                width: 100%;
                margin-bottom: 30px;
            }
            .column {
                width: 48%;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 10px;
            }
            .attribute {
                background-color: #f5f5f5;
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 5px;
            }
            .attribute p {
                margin: 5px 0;
                font-family: Arial, sans-serif;
            }
            .image {
                text-align: center;
            }
            .image img {
                object-fit: contain;
                height: 580px;
                width: 300px;
                border: 1px solid #ddd;
            }
            h2 {
                color: #333;
                font-family: Arial, sans-serif;
                margin-bottom: 10px;
            }
            .success {
                color: green;
                font-weight: bold;
            }
            .failure {
                color: red;
                font-weight: bold;
            }
        </style>
        """
        
        # Generate HTML content
        html = css_style
        html += f"<h2>Matching Result: <span class='success'>Success</span></h2>"
        html += f"<div class='container'>"
        html += f"<div class='column'>"
        html += f"<h3>Original Element</h3>"
        html += f"<div class='attribute'>"
        html += f"<p><strong>Resource ID:</strong> {original_resource_id}</p>"
        html += f"<p><strong>Text:</strong> {original_text}</p>"
        html += f"<p><strong>Content Desc:</strong> {original_content_description}</p>"
        html += f"<p><strong>Class:</strong> {original_class}</p>"
        html += f"</div>"
        html += f"<div class='image'>"
        html += f"<img src='{original_png_path}' alt='Original Element'>"
        html += f"</div>"
        html += f"</div>"
        html += f"<div class='column'>"
        html += f"<h3>Matched Element</h3>"
        html += f"<div class='attribute'>"
        html += f"<p><strong>Resource ID:</strong> {matched_resource_id}</p>"
        html += f"<p><strong>Text:</strong> {matched_text}</p>"
        html += f"<p><strong>Content Desc:</strong> {matched_content_description}</p>"
        html += f"<p><strong>Class:</strong> {matched_class}</p>"
        html += f"</div>"
        html += f"<div class='image'>"
        html += f"<img src='{matched_element_png_path}' alt='Matched Element'>"
        html += f"</div>"
        html += f"</div>"
        html += f"</div>"
        return html
    
    
    
    
    def generate_html_report(self) -> None:
        """
        Generate HTML report - put all test cases in a single HTML file

        1. Iterate test_cases_pairs, add all test case results to the same HTML file
        2. Each test case includes: original element, new version element, matching result, matching method
        3. Copy PNG files to tmp_images directory and reference relative paths in HTML
        """
        import shutil
        from datetime import datetime
        
        # Create report directory
        html_report_dir = os.path.join(self.dataset_dir, "html_report")
        tmp_image_dir = os.path.join(html_report_dir, "tmp_images")
        os.makedirs(tmp_image_dir, exist_ok=True)
        os.makedirs(html_report_dir, exist_ok=True)
        
        # Prepare HTML header
        report_html_path = os.path.join(html_report_dir, "matching_report.html")
        
        # HTML header and styles
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>UI Matching Report</title>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    padding-bottom: 50px;
                }}
                h1, h2, h3 {{
                    color: #333;
                    font-family: Arial, sans-serif;
                    margin-bottom: 10px;
                }}
                .summary {{
                    background-color: #f8f8f8;
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin-bottom: 30px;
                    border-radius: 5px;
                }}
                .summary table {{
                    border-collapse: collapse;
                    width: 100%;
                    font-size: 14px;
                }}
                .summary th, .summary td {{
                    border: 1px solid #ddd;
                    padding: 6px;
                    text-align: center;
                }}
                .summary th {{
                    background-color: #f2f2f2;
                }}
                /* Set column widths for a more compact table */
                .summary table th:nth-child(1), .summary table th:nth-child(2) {{
                    width: 10%;
                }}
                .summary table th:nth-child(3), .summary table th:nth-child(4), .summary table th:nth-child(5) {{
                    width: 6%;
                }}
                /* Use more compact widths for metric columns */
                .summary table th:nth-child(n+6) {{
                    width: 5%;
                }}
                .case-container {{
                    border: 1px solid #ddd;
                    margin-bottom: 40px;
                    padding: 15px;
                    border-radius: 5px;
                    background-color: #fff;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .case-header {{
                    background-color: #f5f5f5;
                    padding: 10px;
                    margin-bottom: 15px;
                    border-radius: 5px;
                }}
                .container {{
                    display: flex;
                    justify-content: space-between;
                    width: 100%;
                    margin-bottom: 30px;
                }}
                .column {{
                    width: 48%;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    padding: 10px;
                }}
                .attribute {{
                    background-color: #f5f5f5;
                    padding: 10px;
                    margin-bottom: 10px;
                    border-radius: 5px;
                }}
                .attribute p {{
                    margin: 5px 0;
                    font-family: Arial, sans-serif;
                }}
                .image {{
                    text-align: center;
                }}
                .image img {{
                    object-fit: contain;
                    height: 580px;
                    width: 300px;
                    border: 1px solid #ddd;
                }}
                .success {{
                    color: green;
                    font-weight: bold;
                }}
                .failure {{
                    color: red;
                    font-weight: bold;
                }}
                .navigation {{
                    position: fixed;
                    top: 10px;
                    right: 10px;
                    background-color: #fff;
                    border: 1px solid #ddd;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    max-height: 300px;
                    overflow-y: auto;
                    z-index: 1000;
                }}
                .navigation h3 {{
                    margin-top: 0;
                }}
                .navigation ul {{
                    list-style-type: none;
                    padding-left: 0;
                    margin: 0;
                }}
                .navigation li {{
                    margin-bottom: 5px;
                }}
                .navigation a {{
                    text-decoration: none;
                    color: #0066cc;
                }}
                .navigation a:hover {{
                    text-decoration: underline;
                }}
                #toc-toggle {{
                    display: block;
                    margin-bottom: 10px;
                    cursor: pointer;
                }}
                .hidden {{
                    display: none;
                }}
            </style>
            <script>
                function toggleTOC() {{
                    var toc = document.getElementById('toc-list');
                    var toggle = document.getElementById('toc-toggle');
                    if (toc.classList.contains('hidden')) {{
                        toc.classList.remove('hidden');
                        toggle.textContent = 'Hide TOC';
                    }} else {{
                        toc.classList.add('hidden');
                        toggle.textContent = 'Show TOC';
                    }}
                }}
                
                // Add anchor navigation after page load
                document.addEventListener('DOMContentLoaded', function() {{
                    var cases = document.querySelectorAll('.case-container');
                    var tocList = document.getElementById('toc-list');
                    
                    cases.forEach(function(caseElement, index) {{
                        var caseId = 'case-' + (index + 1);
                        caseElement.id = caseId;
                        
                        var caseHeader = caseElement.querySelector('.case-header');
                        var recordApp = caseHeader.getAttribute('data-record-app');
                        var replayApp = caseHeader.getAttribute('data-replay-app');
                        
                        var listItem = document.createElement('li');
                        var link = document.createElement('a');
                        link.href = '#' + caseId;
                        link.textContent = recordApp + ' → ' + replayApp;
                        listItem.appendChild(link);
                        tocList.appendChild(listItem);
                    }});
                }});
            </script>
        </head>
        <body>
            <h1>UI Matching Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <!-- Navigation table of contents -->
            <div class="navigation">
                <span id="toc-toggle" onclick="toggleTOC()">Hide TOC</span>
                <ul id="toc-list">
                    <!-- JavaScript will populate this list -->
                </ul>
            </div>
            
            <!-- Summary section -->
            <div class="summary">
                <h2>Summary</h2>
                <table>
                    <tr>
                        <th>Record App</th>
                        <th>Replay App</th>
                        <th>Run Count</th>
                        <th>Result</th>
                        <th>Method</th>
                        <th>Resource ID</th>
                        <th>Text</th>
                        <th>Content Desc</th>
                        <th>Class</th>
                        <th>Visual</th>
                        <th>Bounds Strict</th>
                        <th>Bounds Loose</th>
                    </tr>
        """
        
        # Process each test case pair
        success_count = 0
        total_count = 0
        case_contents = ""
        
        for pair_id, pair_info in self.test_cases_pairs.items():
            try:
                total_count += 1
                record_app, replay_app, run_count = pair_id
                failed_event_number = pair_info['replay_events_count'] + 1
                record_path = pair_info['record_path']
                replay_path = pair_info['replay_path']

                # 1. Get original element info
                original_event = f"{record_path}/events/event_{failed_event_number}.json"
                original_png = f"{record_path}/states/screen_{failed_event_number-1}_marked_original_element.png"
                original_xml = f"{record_path}/xmls/xml_{failed_event_number-1}.xml"
                
                # Check if files exist
                if not os.path.exists(original_png) or not os.path.exists(original_event) or not os.path.exists(original_xml):
                    print(f"Skipping {pair_id}: original files do not exist")
                    continue
                
                # Parse original element
                original_tree = self.parse_xml_file(original_xml)
                original_element = self.find_original_element(original_event, original_tree)
                if original_element is None:
                    print(f"Skipping {pair_id}: unable to find original element")
                    continue

                # 2. Get matched element info
                matched_element_png = f"{replay_path}/states/screen_{failed_event_number-1}_success.png"
                matching_result = pair_info.get('matching_result', {})
                matched_element = matching_result.get('matched_element', {})
                
                # Show in report even if matching failed
                if not os.path.exists(matched_element_png) or not matched_element:
                    # Matching failed, set default values
                    success = False
                    method = "failed"
                    matched_element = {}  # Use empty dict to indicate no matched element
                    # Use a default "matching failed" image
                    matched_element_png = matched_element_png = f"{replay_path}/states/screen_{failed_event_number-1}.png"
                else:
                    success = True
                    method = matching_result.get('matching_method', 'unknown')
                
                if success:
                    success_count += 1
                
                # 3. Copy images to tmp_images directory
                # Generate unique file names
                original_png_filename = f"original_{record_app}_{replay_app}_{run_count}_{failed_event_number-1}.png"
                matched_png_filename = f"matched_{record_app}_{replay_app}_{run_count}_{failed_event_number-1}.png"
                
                original_png_dest = os.path.join(tmp_image_dir, original_png_filename)
                matched_png_dest = os.path.join(tmp_image_dir, matched_png_filename)
                
                # Copy image files
                shutil.copy2(original_png, original_png_dest)
                
                # If matching succeeded, copy matched image; if failed, create a "failure" image
                if os.path.exists(matched_element_png):
                    shutil.copy2(matched_element_png, matched_png_dest)
                else:
                    # Create a simple "matching failed" image
                    try:
                        from PIL import Image, ImageDraw, ImageFont
                        # Create a blank image with the same size as the original
                        try:
                            original_img = Image.open(original_png)
                            width, height = original_img.size
                        except:
                            width, height = 300, 580  # Default size
                        
                        # Create an image with a red background
                        failed_img = Image.new('RGB', (width, height), (255, 240, 240))
                        draw = ImageDraw.Draw(failed_img)
                        
                        # Add text
                        text = "Match Failed"
                        # Try to use system font, fall back to default if it fails
                        try:
                            font = ImageFont.truetype("Arial", 40)
                        except:
                            font = ImageFont.load_default()
                        
                        # Calculate text position to center it
                        text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (150, 40)
                        position = ((width - text_width) // 2, (height - text_height) // 2)
                        
                        # Draw text
                        draw.text(position, text, fill=(255, 0, 0), font=font)
                        
                        # Save image
                        failed_img.save(matched_png_dest)
                    except Exception as e:
                        print(f"Error creating failure image: {e}")
                        # If creating the failure image fails, copy the original image as fallback
                        shutil.copy2(original_png, matched_png_dest)
                
                # Relative image paths (relative to the HTML file)
                original_png_rel_path = f"tmp_images/{original_png_filename}"
                matched_png_rel_path = f"tmp_images/{matched_png_filename}"
                
                # 4. Add to summary table
                result_class = "success" if success else "failure"
                result_text = "Success" if success else "Failure"
                
                # Calculate matching metrics
                metrics = {}
                if success:
                    # Only calculate metrics when matching succeeded
                    metrics = self.calculate_metrics(original_png, original_element, matched_element_png, matched_element)
                else:
                    # When matching fails, set all metrics to N/A
                    metrics = {
                        "Resouce ID": "N/A",
                        "Text": "N/A",
                        "Content Desc": "N/A",
                        "Class": "N/A",
                        "Visual": "N/A",
                        "Bounds Strict": "N/A",
                        "Bounds Loose": "N/A",
                    }
                
                # Set CSS class for each metric
                resource_id_class = "success" if metrics["Resouce ID"] is True else ("failure" if metrics["Resouce ID"] is False else "")
                text_class = "success" if metrics["Text"] is True else ("failure" if metrics["Text"] is False else "")
                content_desc_class = "success" if metrics["Content Desc"] is True else ("failure" if metrics["Content Desc"] is False else "")
                class_class = "success" if metrics["Class"] is True else ("failure" if metrics["Class"] is False else "")
                visual_class = "success" if metrics["Visual"] is True else ("failure" if metrics["Visual"] is False else "")
                bounds_strict_class = "success" if metrics["Bounds Strict"] is True else ("failure" if metrics["Bounds Strict"] is False else "")
                bounds_loose_class = "success" if metrics["Bounds Loose"] is True else ("failure" if metrics["Bounds Loose"] is False else "")
                
                html_content += f"""
                <tr>
                    <td>{record_app}</td>
                    <td>{replay_app}</td>
                    <td>{run_count}</td>
                    <td class="{result_class}">{result_text}</td>
                    <td>{method}</td>
                    <td class="{resource_id_class}">{metrics["Resouce ID"]}</td>
                    <td class="{text_class}">{metrics["Text"]}</td>
                    <td class="{content_desc_class}">{metrics["Content Desc"]}</td>
                    <td class="{class_class}">{metrics["Class"]}</td>
                    <td class="{visual_class}">{metrics["Visual"]}</td>
                    <td class="{bounds_strict_class}">{metrics["Bounds Strict"]}</td>
                    <td class="{bounds_loose_class}">{metrics["Bounds Loose"]}</td>
                </tr>
                """
                
                # 5. Generate detailed content for each test case
                # Safely get attributes to avoid KeyError
                original_resource_id = original_element.attrib.get('resource-id', 'N/A')
                matched_resource_id = matched_element.get('resource-id', 'N/A')

                original_text = original_element.attrib.get('text', 'N/A')
                matched_text = matched_element.get('text', 'N/A')

                original_content_description = original_element.attrib.get('content-desc', 'N/A')
                matched_content_description = matched_element.get('content-desc', 'N/A')

                original_class = original_element.attrib.get('class', 'N/A')
                matched_class = matched_element.get('class', 'N/A')
                
                case_contents += f"""
                <div class="case-container">
                    <div class="case-header" data-record-app="{record_app}" data-replay-app="{replay_app}">
                        <h2>Test Case: {record_app} → {replay_app} (Run {run_count})</h2>
                        <p><strong>Event Number:</strong> {failed_event_number}</p>
                        <p><strong>Result:</strong> <span class="{result_class}">{result_text}</span></p>
                        <p><strong>Method:</strong> {method}</p>
                    </div>
                    
                    <div class="container">
                        <div class="column">
                            <h3>Original Element</h3>
                            <div class="attribute">
                                <p><strong>Resource ID:</strong> {original_resource_id}</p>
                                <p><strong>Text:</strong> {original_text}</p>
                                <p><strong>Content Desc:</strong> {original_content_description}</p>
                                <p><strong>Class:</strong> {original_class}</p>
                            </div>
                            <div class="image">
                                <img src="{original_png_rel_path}" alt="Original Element">
                            </div>
                        </div>
                        <div class="column">
                            <h3>Matched Element</h3>
                            <div class="attribute">
                                <p><strong>Resource ID:</strong> {matched_resource_id}</p>
                                <p><strong>Text:</strong> {matched_text}</p>
                                <p><strong>Content Desc:</strong> {matched_content_description}</p>
                                <p><strong>Class:</strong> {matched_class}</p>
                            </div>
                            <div class="image">
                                <img src="{matched_png_rel_path}" alt="Matched Element">
                            </div>
                        </div>
                    </div>
                </div>
                """
                
            except Exception as e:
                print(f"Error processing {pair_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Finalize HTML content
        success_rate = 0 if total_count == 0 else (success_count / total_count * 100)
        
        html_content += f"""
                </table>
                <p><strong>Summary:</strong> {success_count}/{total_count} successful matches ({success_rate:.1f}% success rate)</p>
            </div>
            
            <!-- Detailed test cases -->
            <h2>Detailed Test Cases</h2>
            {case_contents}
        </body>
        </html>
        """
        
        # Write HTML file
        with open(report_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        print(f"HTML report generated at {report_html_path}")
        print(f"Successful matches: {success_count}/{total_count} ({success_rate:.1f}%)")


            
        

    def calculate_metrics(self, original_png_path, original_element, matched_png_path, matched_element):
        """
        Calculate various metrics between matched element and original element

        Args:
            original_png_path: Screenshot path of the original element
            original_element: XML node of the original element
            matched_png_path: Screenshot path of the matched element
            matched_element: Attribute dictionary of the matched element

        Returns:
            Dictionary containing 7 metrics:
            - Resource ID: Whether resource IDs match
            - Text: Whether text matches
            - Content Desc: Whether content descriptions match
            - Class: Whether class names match
            - Visual: Whether visually similar
            - Bounds Strict: Whether bounds are exactly the same
            - Bounds Loose: Whether bounds have overlap
        """
        print(f"original_png_path: {original_png_path}")
        print(f"matched_png_path: {matched_png_path}")
        original_png_path = original_png_path.replace("_marked_original_element", "")
        matched_png_path = matched_png_path.replace("_success", "")


        if not os.path.exists(original_png_path) or not os.path.exists(matched_png_path):
            return {
                "Resouce ID": 'NA',
                "Text": 'NA',
                "Content Desc": 'NA',
                "Class": 'NA',
                "Visual": 'NA',
                "Bounds Strict": 'NA',
                "Bounds Loose": 'NA',
            }

        # 1. Compare basic attributes, calculate first 4 metrics
        original_resource_id = original_element.attrib.get('resource-id', '')
        matched_resource_id = matched_element.get('resource-id', '')
        resource_id_match = original_resource_id == matched_resource_id
        
        # If both are empty, treat as NA
        if original_resource_id == '' and matched_resource_id == '':
            resource_id_match = 'NA'
            
        original_text = original_element.attrib.get('text', '')
        matched_text = matched_element.get('text', '')
        text_match = original_text == matched_text
        
        # If both are empty, treat as NA
        if original_text == '' and matched_text == '':
            text_match = 'NA'
            
        original_content_desc = original_element.attrib.get('content-desc', '')
        matched_content_desc = matched_element.get('content-desc', '')
        content_desc_match = original_content_desc == matched_content_desc
        
        # If both are empty, treat as NA
        if original_content_desc == '' and matched_content_desc == '':
            content_desc_match = 'NA'
            
        original_class = original_element.attrib.get('class', '')
        matched_class = matched_element.get('class', '')
        class_match = original_class == matched_class

        # If both are empty, treat as NA
        if original_class == '' and matched_class == '':
            class_match = 'NA'
        
        # 2. Calculate bounds-related metrics
        bounds_strict_match = False
        bounds_loose_match = False

        original_bounds_str = original_element.attrib.get('bounds', '')
        matched_bounds_str = matched_element.get('bounds', '')
        
        # Create Matcher instance to call instance methods
        matcher = Matcher(None, None, None, None, None, None)  # Pass required parameters
        
        original_bounds = matcher._parse_bounds(original_bounds_str)
        matched_bounds = matcher._parse_bounds(matched_bounds_str)

        # 2.1 Calculate bounds overlap, bounds strict
        if original_bounds == matched_bounds:
            bounds_strict_match = True
        
        # 2.2 Calculate bounds overlap, bounds loose
        overlap = matcher._calculate_overlap(original_bounds, matched_bounds)
        if overlap > 0:
            bounds_loose_match = True

        # 3. Calculate visual similarity
        visual_match = False
        print(f"original_bounds: {original_bounds}")
        print(f"matched_bounds: {matched_bounds}")
        try:
            # 3.1 Read images
            original_img = read_image(original_png_path)
            replay_img = read_image(matched_png_path)
            print(f"original_img: {original_img}")
            print(f"replay_img: {replay_img}")
                
            # 3.2 Crop element images
            original_x1, original_y1, original_x2, original_y2 = original_bounds
            candidate_x1, candidate_y1, candidate_x2, candidate_y2 = matched_bounds
            
            # Use PIL's crop method to crop images
            try:
                original_crop = original_img.crop((original_x1, original_y1, original_x2, original_y2))
                candidate_crop = replay_img.crop((candidate_x1, candidate_y1, candidate_x2, candidate_y2))
                
                # Check if cropped images are empty
                if original_crop.width <= 1 or original_crop.height <= 1 or candidate_crop.width <= 1 or candidate_crop.height <= 1:
                    visual_match = 'NA'
            except Exception as e:
                print(f"Image cropping error: {e}")
        
            similarity = compute_ssim(original_crop, candidate_crop)
            print(f"similarity: {similarity}")
            if similarity > 0.9:
                visual_match = True
            
        except Exception as e:
            print(f"Error calculating visual similarity: {e}")
                    
        return {
            "Resouce ID": resource_id_match,
            "Text": text_match,
            "Content Desc": content_desc_match,
            "Class": class_match,
            "Visual": visual_match,
            "Bounds Strict": bounds_strict_match,
            "Bounds Loose": bounds_loose_match,
        }


def main():
    """Test function"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse test case directory")
    parser.add_argument("-dataset_dir", help="Test case directory path", default="../test_case1")
    parser.add_argument("-base-app", help="Base app filter, e.g. v6_4_1", default=None) # Can filter to view only a specific base app
    parser.add_argument("-run-count", help="Run count filter, e.g. 3", default=None)
    parser.add_argument("-max-workers", type=int, help="Max worker threads, defaults to None which uses CPU core count", default=30)
    parser.add_argument("-log-dir", help="Log directory path, defaults to 'logs'", default="logs")
    parser.add_argument("-only-report-failed", help="Only report failed test case pairs", action="store_true")
    parser.add_argument("-debug", help="Enable verbose debug output", action="store_true", default=False)
    args = parser.parse_args()
    
    # Create the main process logger
    main_logger = get_logger(name="UIMatch_main", log_dir=args.log_dir, force_new_handlers=True)
    main_logger.info(f"Starting test case processing, args: dataset_dir={args.dataset_dir}, base_app={args.base_app}, run_count={args.run_count}, max_workers={args.max_workers}")
    
    test_parser = Parser(args.dataset_dir, args.base_app, args.run_count)
    # Pass the logger to the Parser instance
    test_parser.logger = main_logger

    if args.only_report_failed == False:
        # Match failed events
        if args.debug == False:
            test_parser.read_and_matching_failed_events_parallel(max_workers=args.max_workers)
        else:
            test_parser.read_and_matching_failed_events_sequential()
    else:
        # When only generating report, load matching info from existing result files
        for pair_id, pair_info in test_parser.test_cases_pairs.items():
            record_app, replay_app, run_count = pair_id
            replay_path = pair_info['replay_path']
            failed_event_number = pair_info.get('replay_events_count', 0) + 1
            
            # Try to load matching result file
            result_json_path = f"{replay_path}/matched_element_{failed_event_number-1}.json"
            if os.path.exists(result_json_path):
                try:
                    with open(result_json_path, 'r', encoding='utf-8') as f:
                        matched_element = json.load(f)
                        matching_result = pair_info.get('matching_result', {})
                        matching_result['success'] = True
                        
                        # Try to load matching method
                        method_json_path = f"{replay_path}/matched_method_{failed_event_number-1}.json"
                        if os.path.exists(method_json_path):
                            try:
                                with open(method_json_path, 'r', encoding='utf-8') as mf:
                                    matching_method = json.load(mf)
                                    matching_result['matching_method'] = matching_method
                            except Exception as me:
                                print(f"Failed to load matching method {pair_id}: {me}")
                                matching_result['matching_method'] = "Unknown"
                        else:
                            matching_result['matching_method'] = "Unknown"
                            
                        matching_result['matched_element'] = matched_element
                        pair_info['matching_result'] = matching_result
                        print(f"Loaded matching result: {pair_id}")
                except Exception as e:
                    print(f"Failed to load matching result {pair_id}: {e}")
            else:
                print(f"Matching result file does not exist: {result_json_path}")   

    # Generate HTML report
    test_parser.generate_html_report()
    
    

if __name__ == "__main__":
    main()