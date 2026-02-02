#!/usr/bin/env python3
"""
DroidBot Test Runner
"""

import os
import sys
import csv
import time
import logging
import argparse
import subprocess
import signal
import glob
import re
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from queue import Queue
import threading

# Log configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DroidBotRunner:
    def __init__(self, args):
        self.mode = args.mode
        self.csv_file = args.csv_file
        self.apk_base = args.apk_base
        self.android_home = args.android_home
        self.avd_name = args.avd_name
        self.start_py = args.start_py
        self.count = args.count
        self.max_parallel = args.max_parallel
        self.run_count = args.run_count
        self.per_task_timeout = args.per_task_timeout
        self.test_mode = args.test_mode
        self.parent_dir = args.parent_dir
        self.log_dir = args.log_dir or self._default_log_dir()
        
        # Port configuration
        self.base_port = 5554
        self.port_step = 2
        
        self._setup_environment()
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _default_log_dir(self) -> str:
        base_dir = {
            'record': './logs',
            'replay_original': './logs_replay',
            'replay_new': './logs_replay_new'
        }.get(self.mode, './logs')
        
        if self.parent_dir:
            return os.path.join(self.parent_dir, base_dir.lstrip('./'))
        return base_dir
    
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
    
    def sanitize_suffix(self, apk_path: str) -> str:
        """Generate safe directory name"""
        s = os.path.basename(apk_path)
        s = os.path.splitext(s)[0]
        s = ''.join(c if c.isalnum() or c in '._-' else '_' for c in s)
        s = s.replace('.', '_').rstrip('_')
        return s if s else 'apk'
    
    def read_csv_apks(self, csv_file: str) -> List[str]:
        """Read the 7th column of the CSV file"""
        apks = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if self.mode == 'replay_new':
            start_row = 1
        else:
            start_row = 2
        
        for row in rows[start_row:]:
            if len(row) > 6 and row[6].strip():
                apks.append(row[6].strip())
        logger.info(f"Loaded {len(apks)} APKs from {csv_file}")
        return apks
    
    def start_emulators_batch(self, num_emulators: int) -> bool:
        """Start multiple emulators in batch"""
        try:
            logger.info(f"Starting {num_emulators} emulators...")
            result = subprocess.run(
                ['./start_emulator.sh', str(num_emulators)],
                check=True,
                capture_output=True,
                text=True,
                timeout=600  # 10-minute timeout
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
        """
        Enhanced kill_emulator - more reliable
        When force=True, it will forcibly clean up without hanging
        """
        port = serial.split('-')[1]
        logger.info(f"Killing emulator {serial}...")
        
        try:
            # Method 1: adb kill (set short timeout to avoid hanging)
            subprocess.run(
                ['adb', '-s', serial, 'emu', 'kill'],
                capture_output=True,
                timeout=3  # ← Wait only 3 seconds
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"adb kill timeout for {serial}, using force kill")
        except Exception as e:
            logger.debug(f"adb kill error: {e}")
        
        if force:
            # Method 2: pkill (won't hang)
            try:
                subprocess.run(
                    ['pkill', '-9', '-f', f'emulator.*-port {port}'],
                    capture_output=True,
                    timeout=2
                )
            except:
                pass
            
            # Method 3: Clean up PID file
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
    
    def generate_tasks_record(self, apks: List[str]) -> List[Dict]:
        """Generate record tasks"""
        tasks = []
        for apk_rel in apks:
            apk_path = apk_rel if apk_rel.startswith('/') else os.path.join(self.apk_base, apk_rel)
            if not os.path.exists(apk_path):
                logger.warning(f"APK not found: {apk_path}")
                continue
            
            suffix = self.sanitize_suffix(apk_path)
            for run_idx in range(1, self.run_count + 1):
                out_dir = f"record_output_{suffix}_run{run_idx}"
                if self.parent_dir:
                    out_dir = os.path.join(self.parent_dir, out_dir)
                if os.path.exists(out_dir):
                    logger.debug(f"Skip existing: {out_dir}")
                    continue
                
                tasks.append({
                    'apk_path': apk_path,
                    'suffix': suffix,
                    'run_idx': run_idx,
                    'out_dir': out_dir,
                    'policy': 'random_exploration'
                })
        return tasks
    
    def generate_tasks_replay_original(self, apks: List[str]) -> List[Dict]:
        """Generate replay_original tasks"""
        tasks = []
        for apk_rel in apks:
            apk_path = apk_rel if apk_rel.startswith('/') else os.path.join(self.apk_base, apk_rel)
            if not os.path.exists(apk_path):
                logger.warning(f"APK not found: {apk_path}")
                continue
            
            suffix = self.sanitize_suffix(apk_path)
            search_pattern = f"record_output_{suffix}_run*"
            if self.parent_dir:
                search_pattern = os.path.join(self.parent_dir, search_pattern)
            patterns = glob.glob(search_pattern)
            
            for pattern in patterns:
                match = re.search(r'_run(\d+)$', pattern)
                if match and os.path.isdir(pattern):
                    run_idx = int(match.group(1))
                    record_dir = pattern
                    replay_dir = f"replay_output_{suffix}_run{run_idx}"
                    if self.parent_dir:
                        replay_dir = os.path.join(self.parent_dir, replay_dir)
                    
                    if os.path.exists(replay_dir):
                        logger.debug(f"Skip existing: {replay_dir}")
                        continue
                    
                    tasks.append({
                        'apk_path': apk_path,
                        'suffix': suffix,
                        'run_idx': run_idx,
                        'out_dir': replay_dir,
                        'record_dir': record_dir,
                        'policy': 'replay'
                    })
        return tasks
    
    def generate_tasks_replay_new(self, apks: List[str]) -> List[Dict]:
        """Generate replay_new tasks"""
        tasks = []
        for base_index in range(len(apks) - 1, 0, -1):
            base_apk_rel = apks[base_index]
            base_apk = base_apk_rel if base_apk_rel.startswith('/') else os.path.join(self.apk_base, base_apk_rel)
            if not os.path.exists(base_apk):
                continue
            
            base_suffix = self.sanitize_suffix(base_apk)
            search_pattern = f"record_output_{base_suffix}_run*"
            if self.parent_dir:
                search_pattern = os.path.join(self.parent_dir, search_pattern)
            patterns = glob.glob(search_pattern)
            
            runs = []
            for pattern in patterns:
                match = re.search(r'_run(\d+)$', pattern)
                if match and os.path.isdir(pattern):
                    runs.append((int(match.group(1)), pattern))
            
            if not runs:
                continue
            
            target_apks = apks[:base_index]
            for target_apk_rel in target_apks:
                target_apk = target_apk_rel if target_apk_rel.startswith('/') else os.path.join(self.apk_base, target_apk_rel)
                if not os.path.exists(target_apk):
                    continue
                
                target_suffix = self.sanitize_suffix(target_apk)
                for run_idx, record_dir in runs:
                    replay_dir = f"replay_output_{target_suffix}_run{run_idx}_for_{base_suffix}"
                    if self.parent_dir:
                        replay_dir = os.path.join(self.parent_dir, replay_dir)
                    
                    if os.path.exists(replay_dir):
                        logger.debug(f"Skip existing: {replay_dir}")
                        continue
                    
                    tasks.append({
                        'apk_path': target_apk,
                        'suffix': target_suffix,
                        'run_idx': run_idx,
                        'out_dir': replay_dir,
                        'record_dir': record_dir,
                        'policy': 'replay',
                        'base_suffix': base_suffix
                    })
        return tasks
    
    def run_single_task(self, task: Dict, serial: str) -> bool:
        """
        Run a single task
        Key improvement: Added more protection to avoid hanging
        """
        apk_path = task['apk_path']
        out_dir = task['out_dir']
        run_idx = task['run_idx']
        policy = task['policy']
        record_dir = task.get('record_dir')
        
        port = serial.split('-')[1]
        log_file = os.path.join(self.log_dir, f"{os.path.basename(out_dir)}_{port}.log")
        
        logger.info(f"[{serial}] Running {task['suffix']} (run {run_idx})")
        
        # Build command
        cmd = [
            'python3', self.start_py,
            '-a', apk_path,
            '-o', out_dir,
            '-is_emulator',
            '-policy', policy,
            '-count', str(self.count),
            '-d', serial
        ]
        if policy == 'replay' and record_dir:
            cmd.extend(['-replay_output', record_dir])
        
        logger.debug(f"[{serial}] CMD: {' '.join(cmd)}")
        
        # Execute command
        success = False
        try:
            if self.test_mode:
                logger.info(f"[{serial}] TEST MODE → {out_dir}")
                time.sleep(1)
                success = True
            else:
                with open(log_file, 'w') as f:
                    # ✅ Key: Set timeout
                    result = subprocess.run(
                        cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        timeout=self.per_task_timeout
                    )
                    success = (result.returncode == 0)
        
        except subprocess.TimeoutExpired:
            logger.error(f"[{serial}] ✗ TIMEOUT after {self.per_task_timeout}s")
            # Clean up incomplete output
            shutil.rmtree(out_dir, ignore_errors=True)
            success = False
        
        except Exception as e:
            logger.error(f"[{serial}] ✗ Error: {e}")
            success = False
        
        finally:
            # ✅ Key: Always kill the emulator, using force=True
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
        """
        Run a batch of tasks
        Key improvement: Use queue for emulator allocation, thread-safe
        """
        batch_size = len(tasks)
        logger.info("=" * 60)
        logger.info(f"Batch: {batch_size} tasks")
        logger.info("=" * 60)
        
        # Start emulators
        if not self.start_emulators_batch(batch_size):
            logger.error("Failed to start emulators")
            return 0
        
        # ✅ Key improvement: Use thread-safe queue for emulator allocation
        serial_queue = Queue()
        for i in range(batch_size):
            port = self.base_port + i * self.port_step
            serial = f"emulator-{port}"
            serial_queue.put(serial)
        
        logger.info(f"Emulators ready: {batch_size}")
        
        # ✅ Improvement: Use fixed mapping to avoid race conditions
        def worker(task_and_serial):
            task, serial = task_and_serial
            return self.run_single_task(task, serial)
        
        # Pre-allocate task and emulator mapping
        task_assignments = []
        for task in tasks:
            serial = serial_queue.get()  # Get from queue
            task_assignments.append((task, serial))
        
        # Concurrent execution (now each task has a fixed serial, no conflicts)
        success_count = 0
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # ✅ Using map is clearer
            results = executor.map(worker, task_assignments)
            for success in results:
                if success:
                    success_count += 1
        
        logger.info(f"Batch completed: {success_count}/{batch_size} successful")
        
        # Finally ensure cleanup
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
        logger.info(f"{self.mode.upper()} Mode")
        logger.info(f"Max Parallel: {self.max_parallel}")
        if self.mode == 'record':
            logger.info(f"Runs per APK: {self.run_count}")
        logger.info("=" * 60)
        
        # Clean up environment
        self.cleanup_all_emulators()
        
        # Read APKs
        apks = self.read_csv_apks(self.csv_file)
        if not apks:
            logger.error("No APKs found")
            return
        
        # Generate tasks
        if self.mode == 'record':
            all_tasks = self.generate_tasks_record(apks)
        elif self.mode == 'replay_original':
            all_tasks = self.generate_tasks_replay_original(apks)
        elif self.mode == 'replay_new':
            all_tasks = self.generate_tasks_replay_new(apks)
        else:
            logger.error(f"Unknown mode: {self.mode}")
            return
        
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
            
            # Rest between batches
            if i + self.max_parallel < len(all_tasks):
                logger.info("Waiting 5s before next batch...")
                time.sleep(5)
        
        # Statistics
        logger.info(f"\n{'=' * 60}")
        logger.info(f"{self.mode.upper()} Completed")
        logger.info(f"Success: {total_success}/{len(all_tasks)}")
        logger.info(f"{'=' * 60}")


def signal_handler(signum, frame):
    """Ctrl+C handler"""
    logger.warning("Interrupted! Cleaning up...")
    subprocess.run(['pkill', '-9', 'emulator'], capture_output=True)
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='DroidBot Three Running Modes (Fixed Version)')
    
    parser.add_argument('mode', choices=['record', 'replay_original', 'replay_new'])
    parser.add_argument('--csv-file', '--csv', required=True)
    parser.add_argument('--apk-base', required=True)
    parser.add_argument('--android-home', default=os.path.expanduser("~/android-sdk"))
    parser.add_argument('--avd-name', '--avd', default='Android10.0')
    parser.add_argument('--start-py', default='start.py')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--max-parallel', type=int, default=8)
    parser.add_argument('--run-count', type=int, default=20)
    parser.add_argument('--per-task-timeout', type=int, default=1800)
    parser.add_argument('--log-dir', default=None)
    parser.add_argument('--test-mode', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--read-only', action='store_true')
    parser.add_argument('--boot-timeout', type=int, default=180)
    parser.add_argument('--max-retry', type=int, default=3)
    parser.add_argument('--parent-dir', default='', help='Parent directory to prefix all output directories')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        runner = DroidBotRunner(args)
        runner.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()