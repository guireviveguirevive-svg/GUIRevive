#!/usr/bin/env python3
"""
DroidBot Repair Batch Runner

Supports one running mode:
- repair: Fix failed test cases (parse failed cases from html_report)

Repair mode parameters:
- -replay_output: Original recording directory, e.g., record_output_{record_app}_run{N}
- -failed_replay_output: Failed replay directory, e.g., replay_output_{replay_app}_run{N}_for_{record_app}
- -o: Repair output directory, e.g., repair_output_{replay_app}_run{N}_for_{record_app}

Directory structure:
{apk-base}/
├── html_report/
│   └── matching_report.html  <- Parse failed cases from here
├── record_output_{app}_run{N}/
├── replay_output_{app}_run{N}_for_{base}/
├── repair_output_{app}_run{N}_for_{base}/  <- Output here
└── select_apks/
    └── {package_name}/
        └── v{version}.apk

Usage example:
python3 start_repair_bash.py \
    --apk-base /data/shiwensong/droidbot/com.byagowi.persiancalendar \
    --max-parallel 4
"""

import os
import sys
import re
import csv
import time
import logging
import argparse
import subprocess
import signal
import glob
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Set, Tuple
from queue import Queue

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# Log configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_skip_cases(csv_path: str) -> Set[Tuple[str, str, str]]:
    """
    Load the list of cases to skip

    Args:
        csv_path: Path to Need_to_delete.csv file

    Returns:
        Set of (record_app, replay_app, run_count) tuples
    """
    skip_cases = set()
    if not os.path.exists(csv_path):
        logger.warning(f"CSV file does not exist: {csv_path}")
        return skip_cases

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            record_app = row.get('Record App', '').strip()
            replay_app = row.get('Replay App', '').strip()
            run_count = row.get('Run Count', '').strip()
            if record_app and replay_app and run_count:
                skip_cases.add((record_app, replay_app, run_count))

    logger.info(f"Loaded {len(skip_cases)} cases to skip")
    return skip_cases


def should_skip_case(record_app: str, replay_app: str, run_count: str,
                     skip_cases: Set[Tuple[str, str, str]]) -> bool:
    """Check if this case should be skipped"""
    return (record_app, replay_app, run_count) in skip_cases


class DroidBotRepairRunner:
    """DroidBot Repair Batch Runner"""

    def __init__(self, args):
        self.apk_base = args.apk_base  # Data directory, e.g., /data/.../com.xxx
        self.android_home = args.android_home
        self.avd_name = args.avd_name
        self.start_py = args.start_py
        self.count = args.count
        self.max_parallel = args.max_parallel
        self.per_task_timeout = args.per_task_timeout
        self.test_mode = args.test_mode

        # Load cases to skip
        self.skip_delete_cases = args.skip_delete_cases
        self.skip_cases = set()
        if self.skip_delete_cases:
            self.skip_cases = load_skip_cases(args.delete_csv)

        # Repair output directory suffix
        self.repair_output_dir_suffix = args.repair_output_dir_suffix

        # Ablation experiment parameters
        self.without_taxonomy = args.without_taxonomy
        self.without_rule = args.without_rule
        self.without_llm = args.without_llm
        self.without_next_screen_summary = args.without_next_screen_summary
        self.without_history_summary = args.without_history_summary

        # HTML report path automatically derived from apk_base
        self.html_report = os.path.join(self.apk_base, 'html_report', 'matching_report.html')

        # select_apks directory (can be specified by parameter, otherwise derived from apk_base)
        # Default path: ../droidbot/select_apks/{package_name}
        if args.select_apks:
            self.select_apks_dir = args.select_apks
        else:
            package_name = os.path.basename(os.path.abspath(self.apk_base))
            parent_dir = os.path.dirname(os.path.abspath(self.apk_base))
            self.select_apks_dir = os.path.join(parent_dir, 'droidbot', 'select_apks', package_name)

        self.log_dir = args.log_dir or self._default_log_dir()

        # Port configuration
        self.base_port = args.base_port
        self.port_step = 2

        self._setup_environment()
        os.makedirs(self.log_dir, exist_ok=True)

    def _default_log_dir(self) -> str:
        return os.path.join(self.apk_base, 'logs_repair')

    def _setup_environment(self):
        """Set up Android SDK environment variables"""
        os.environ['ANDROID_HOME'] = self.android_home
        os.environ['ANDROID_SDK_ROOT'] = self.android_home
        paths = [
            f"{self.android_home}/cmdline-tools/latest/bin",
            f"{self.android_home}/platform-tools",
            f"{self.android_home}/emulator"
        ]
        os.environ['PATH'] = ':'.join(paths) + ':' + os.environ.get('PATH', '')

    def start_emulators_batch(self, num_emulators: int) -> bool:
        """Start multiple emulators in batch"""
        try:
            logger.info(f"Starting {num_emulators} emulators...")
            result = subprocess.run(
                ['./start_emulator.sh', str(num_emulators)],
                check=True,
                capture_output=True,
                text=True,
                timeout=600
            )
            logger.info(f"✓ Started {num_emulators} emulators")
            return True
        except subprocess.TimeoutExpired:
            logger.error("Timeout starting emulators")
            return False
        except Exception as e:
            logger.error(f"Failed to start emulators: {e}")
            return False

    def kill_emulator(self, serial: str, force: bool = True):
        """Kill the emulator"""
        port = serial.split('-')[1]
        logger.info(f"Killing emulator {serial}...")

        try:
            subprocess.run(
                ['adb', '-s', serial, 'emu', 'kill'],
                capture_output=True,
                timeout=3
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"adb kill timeout for {serial}, using force kill")
        except Exception as e:
            logger.debug(f"adb kill error: {e}")

        if force:
            try:
                subprocess.run(
                    ['pkill', '-9', '-f', f'emulator.*-port {port}'],
                    capture_output=True,
                    timeout=2
                )
            except:
                pass

            pid_file = f'emulator_{port}.pid'
            if os.path.exists(pid_file):
                try:
                    with open(pid_file, 'r') as f:
                        pid = f.read().strip()
                    os.kill(int(pid), signal.SIGKILL)
                except:
                    pass
                try:
                    os.unlink(pid_file)
                except:
                    pass

        logger.info(f"✓ Killed {serial}")

    def cleanup_all_emulators(self):
        """Clean up all emulators"""
        logger.info("Cleaning up all emulators...")
        try:
            subprocess.run(
                ['pkill', '-9', '-f', f'emulator.*-avd {self.avd_name}'],
                capture_output=True,
                timeout=5
            )
        except:
            pass
        time.sleep(2)
        logger.info("✓ Cleanup completed")

    def parse_html_report(self, html_report_path: str) -> List[Dict]:
        """
        Parse HTML report, extract failed test cases

        HTML table columns:
        - Record App: Base version (APK version during recording)
        - Replay App: New version (APK version during replay)
        - Run Count: Number of runs
        - Result: Matching result (Success/Failure)
        - Method: Matching method

        Returns:
            List of failed test cases
        """
        if not HAS_BS4:
            logger.error("Need to install beautifulsoup4: pip install beautifulsoup4")
            return []

        if not os.path.exists(html_report_path):
            logger.error(f"HTML report file does not exist: {html_report_path}")
            return []

        with open(html_report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # Find the summary table
        summary_div = soup.find('div', class_='summary')
        if not summary_div:
            logger.error("Summary table not found")
            return []

        table = summary_div.find('table')
        if not table:
            logger.error("Table not found")
            return []

        # Get all rows (skip header)
        rows = table.find_all('tr')[1:]

        failed_cases = []
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 5:
                continue

            record_app = cells[0].get_text(strip=True)
            replay_app = cells[1].get_text(strip=True)
            run_count = cells[2].get_text(strip=True)
            result = cells[3].get_text(strip=True)
            method = cells[4].get_text(strip=True)

            # Only collect failed cases
            if result.lower() == 'failure' or result.lower() == 'success':
                failed_cases.append({
                    'record_app': record_app,
                    'replay_app': replay_app,
                    'run_count': run_count,
                    'result': result,
                    'method': method
                })

        logger.info(f"Parsed {len(failed_cases)} failed test cases from HTML report")
        return failed_cases

    def sanitize_suffix(self, name: str) -> str:
        """
        Generate safe directory name (consistent with start_bash.py)

        Processing logic:
        1. Keep only alphanumeric and ._-, replace other characters with _
        2. Replace . with _
        3. Remove trailing _
        """
        s = os.path.splitext(name)[0]  # Remove extension
        s = ''.join(c if c.isalnum() or c in '._-' else '_' for c in s)
        s = s.replace('.', '_').rstrip('_')
        return s if s else 'apk'

    def suffix_to_apk_path(self, suffix: str) -> Optional[str]:
        """
        Convert app version name from table to APK path

        The suffix in the HTML table is already sanitized (extracted from directory name)
        So we just need to apply the same sanitize_suffix processing to the APK filename and compare directly

        For example:
        - suffix (from HTML): "Notes__Privacy_Friendly__v1_3_0"
        - APK: "Notes_(Privacy_Friendly)_v1.3.0.apk"
        - after sanitize: "Notes__Privacy_Friendly__v1_3_0"
        - They match, successful match

        Args:
            suffix: Version name (already in sanitized form)

        Returns:
            APK file path, or None if not found
        """
        # Get all APK files
        search_pattern = os.path.join(self.select_apks_dir, "**/*.apk")
        all_apks = glob.glob(search_pattern, recursive=True)

        # Iterate to find matching APK
        for apk_path in all_apks:
            apk_name = os.path.basename(apk_path)
            # Process APK filename with sanitize_suffix
            sanitized_apk = self.sanitize_suffix(apk_name)

            if suffix == sanitized_apk:
                logger.debug(f"Found APK: {suffix} -> {apk_path}")
                return apk_path

        logger.warning(f"APK not found: {suffix}")
        return None

    def generate_tasks(self) -> List[Dict]:
        """
        Generate repair tasks from HTML report

        Directory mapping (all directories under apk_base):
        - record_output: record_output_{record_app}_run{N}
        - failed_replay_output: replay_output_{replay_app}_run{N}_for_{record_app}
        - repair_output: repair_output_{replay_app}_run{N}_for_{record_app}
        """
        failed_cases = self.parse_html_report(self.html_report)
        if not failed_cases:
            return []

        tasks = []
        skipped_count = 0
        for case in failed_cases:
            record_app = case['record_app']
            replay_app = case['replay_app']
            run_count = case['run_count']

            # Check if should skip
            if should_skip_case(record_app, replay_app, run_count, self.skip_cases):
                logger.info(f"Skip: In Need_to_delete.csv - {record_app} <- {replay_app} run{run_count}")
                skipped_count += 1
                continue

            # Build directory names (relative to apk_base)
            record_output_dir = os.path.join(self.apk_base, f"record_output_{record_app}_run{run_count}")
            failed_replay_output_dir = os.path.join(self.apk_base, f"replay_output_{replay_app}_run{run_count}_for_{record_app}")
            repair_output_dir = os.path.join(self.apk_base, f"repair_output_{replay_app}_run{run_count}_for_{record_app}{self.repair_output_dir_suffix}")

            # Check if directories exist
            if not os.path.exists(record_output_dir):
                logger.info(f"Skip: record directory does not exist: {record_output_dir}")
                continue

            if not os.path.exists(failed_replay_output_dir):
                logger.info(f"Skip: failed replay directory does not exist: {failed_replay_output_dir}")
                continue

            # Check if already repaired
            if os.path.exists(repair_output_dir):
                logger.info(f"Skip: repair directory already exists: {repair_output_dir}")
                continue

            # Find APK path (using replay_app, i.e., new version)
            apk_path = self.suffix_to_apk_path(replay_app)
            if not apk_path:
                logger.info(f"Skip: APK not found: {replay_app} (in {self.select_apks_dir})")
                continue

            tasks.append({
                'apk_path': apk_path,
                'suffix': replay_app,
                'run_idx': int(run_count),
                'out_dir': repair_output_dir,
                'record_dir': record_output_dir,
                'failed_replay_dir': failed_replay_output_dir,
                'base_suffix': record_app
            })

        logger.info(f"Generated {len(tasks)} repair tasks")
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} cases (in Need_to_delete.csv)")
        return tasks

    def run_single_task(self, task: Dict, serial: str) -> bool:
        """Run a single repair task"""
        apk_path = task['apk_path']
        out_dir = task['out_dir']
        run_idx = task['run_idx']
        record_dir = task['record_dir']
        failed_replay_dir = task['failed_replay_dir']

        port = serial.split('-')[1]
        log_file = os.path.join(self.log_dir, f"{os.path.basename(out_dir)}_{port}.log")

        logger.info(f"[{serial}] Running repair: {task['suffix']} (run {run_idx})")

        # Build command
        cmd = [
            'python3', self.start_py,
            '-a', apk_path,
            '-o', out_dir,
            '-replay_output', record_dir,
            '-failed_replay_output', failed_replay_dir,
            '-is_emulator',
            '-policy', 'matching',
            '-count', str(self.count),
            '-d', serial
        ]

        # Add ablation experiment parameters if enabled
        if self.without_taxonomy:
            cmd.append('-without_taxonomy')
        if self.without_rule:
            cmd.append('-without_rule')
        if self.without_llm:
            cmd.append('-without_llm')
        if self.without_next_screen_summary:
            cmd.append('-without_next_screen_summary')
        if self.without_history_summary:
            cmd.append('-without_history_summary')

        logger.debug(f"[{serial}] CMD: {' '.join(cmd)}")

        success = False
        try:
            if self.test_mode:
                logger.info(f"[{serial}] TEST MODE → {out_dir}")
                time.sleep(1)
                success = True
            else:
                with open(log_file, 'w') as f:
                    result = subprocess.run(
                        cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        timeout=self.per_task_timeout
                    )
                    success = (result.returncode == 0)

        except subprocess.TimeoutExpired:
            logger.error(f"[{serial}] ✗ TIMEOUT after {self.per_task_timeout}s")
            shutil.rmtree(out_dir, ignore_errors=True)
            success = False

        except Exception as e:
            logger.error(f"[{serial}] ✗ Error: {e}")
            success = False

        finally:
            try:
                self.kill_emulator(serial, force=True)
            except Exception as e:
                logger.error(f"Error killing {serial}: {e}")

        if success:
            logger.info(f"[{serial}] ✓ Success → {out_dir}")
        else:
            logger.error(f"[{serial}] ✗ Failed (see {log_file})")

        return success

    def run_batch(self, tasks: List[Dict]) -> int:
        """Run a batch of tasks"""
        batch_size = len(tasks)
        logger.info("=" * 60)
        logger.info(f"Batch: {batch_size} tasks")
        logger.info("=" * 60)

        if not self.start_emulators_batch(batch_size):
            logger.error("Failed to start emulators")
            return 0

        serial_queue = Queue()
        for i in range(batch_size):
            port = self.base_port + i * self.port_step
            serial = f"emulator-{port}"
            serial_queue.put(serial)

        logger.info(f"Emulators ready: {batch_size}")

        def worker(task_and_serial):
            task, serial = task_and_serial
            return self.run_single_task(task, serial)

        task_assignments = []
        for task in tasks:
            serial = serial_queue.get()
            task_assignments.append((task, serial))

        success_count = 0
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = executor.map(worker, task_assignments)
            for success in results:
                if success:
                    success_count += 1

        logger.info(f"Batch completed: {success_count}/{batch_size} successful")

        time.sleep(2)
        for _, serial in task_assignments:
            try:
                self.kill_emulator(serial, force=True)
            except:
                pass

        return success_count

    def run(self):
        """Main execution logic"""
        logger.info("=" * 60)
        logger.info("REPAIR Mode")
        logger.info(f"Data Dir: {self.apk_base}")
        logger.info(f"HTML Report: {self.html_report}")
        logger.info(f"APK Dir: {self.select_apks_dir}")
        logger.info(f"Max Parallel: {self.max_parallel}")
        logger.info("=" * 60)

        # Check if HTML report exists
        if not os.path.exists(self.html_report):
            logger.error(f"HTML report does not exist: {self.html_report}")
            return

        # Check if select_apks directory exists
        if not os.path.exists(self.select_apks_dir):
            logger.error(f"APK directory does not exist: {self.select_apks_dir}")
            return

        # Clean up environment
        self.cleanup_all_emulators()

        # Generate tasks
        all_tasks = self.generate_tasks()

        if not all_tasks:
            logger.warning("No tasks to run")
            return

        logger.info(f"Total tasks: {len(all_tasks)}")

        # Execute in batches
        total_success = 0
        for i in range(0, len(all_tasks), self.max_parallel):
            batch = all_tasks[i:i + self.max_parallel]
            batch_num = i // self.max_parallel + 1
            total_batches = (len(all_tasks) + self.max_parallel - 1) // self.max_parallel

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Batch {batch_num}/{total_batches}")
            logger.info(f"{'=' * 60}")

            success = self.run_batch(batch)
            total_success += success

            if i + self.max_parallel < len(all_tasks):
                logger.info("Waiting 5s before next batch...")
                time.sleep(5)

        # Statistics
        logger.info(f"\n{'=' * 60}")
        logger.info("REPAIR Completed")
        logger.info(f"Success: {total_success}/{len(all_tasks)}")
        logger.info(f"{'=' * 60}")


def signal_handler(signum, frame):
    """Ctrl+C handler"""
    logger.warning("Interrupted! Cleaning up...")
    subprocess.run(['pkill', '-9', 'emulator'], capture_output=True)
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='DroidBot Repair Batch Runner - Parse failed cases from HTML report and repair them'
    )

    parser.add_argument('--apk-base', required=True,
                        help='Data directory, e.g., /data/shiwensong/droidbot/com.xxx, will automatically read html_report/matching_report.html')
    parser.add_argument('--android-home', default=os.path.expanduser("~/android-sdk"),
                        help='Android SDK path')
    parser.add_argument('--avd-name', '--avd', default='Android10.0',
                        help='AVD name')
    parser.add_argument('--start-py', default='start.py',
                        help='start.py path')
    parser.add_argument('--count', type=int, default=100,
                        help='Number of events per task')
    parser.add_argument('--max-parallel', type=int, default=8,
                        help='Maximum number of parallel tasks')
    parser.add_argument('--per-task-timeout', type=int, default=1800,
                        help='Timeout for each task (seconds), default 30 minutes')
    parser.add_argument('--log-dir', default=None,
                        help='Log directory (default at apk-base/logs_repair)')
    parser.add_argument('--test-mode', action='store_true',
                        help='Test mode, no actual execution')
    parser.add_argument('--base-port', type=int, default=5554,
                        help='Initial emulator port (default 5554)')
    parser.add_argument('--select-apks', default=None,
                        help='APK directory path (default derived from apk-base/select_apks)')
    parser.add_argument('--skip-delete-cases', action='store_true',
                        help='Skip cases listed in Need_to_delete.csv')
    parser.add_argument('--delete-csv', default='Need_to_delete.csv',
                        help='Path to the file listing cases to delete (default: Need_to_delete.csv)')
    parser.add_argument('--repair-output-dir-suffix', default='',
                        help='Repair output directory suffix, appended directly to the directory name (default: empty)')

    # Ablation experiment parameters
    parser.add_argument("--without_taxonomy", action="store_true", dest="without_taxonomy",
                        help="Ablation experiment: disable taxonomy in exploration.")
    parser.add_argument("--without_rule", action="store_true", dest="without_rule",
                        help="Ablation experiment: disable rule-based matching.")
    parser.add_argument("--without_llm", action="store_true", dest="without_llm",
                        help="Ablation experiment: disable LLM-based matching.")
    parser.add_argument("--without_next_screen_summary", action="store_true", dest="without_next_screen_summary",
                        help="Ablation experiment: disable next screen summary for icon understanding.")
    parser.add_argument("--without_history_summary", action="store_true", dest="without_history_summary",
                        help="Ablation experiment: disable exploration history summary.")

    return parser.parse_args()


def main():
    args = parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        runner = DroidBotRepairRunner(args)
        runner.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
