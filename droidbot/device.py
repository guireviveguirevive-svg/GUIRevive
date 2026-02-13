import logging
import os
import re
import subprocess
import sys
import time

import uiautomator2

from .adapter.adb import ADB
from .adapter.droidbot_app import DroidBotAppConn
from .adapter.logcat import Logcat
from .adapter.minicap import Minicap
from .adapter.process_monitor import ProcessMonitor
from .adapter.telnet import TelnetConsole
from .adapter.user_input_monitor import UserInputMonitor
from .adapter.droidbot_ime import DroidBotIme
from .app import App
from .intent import Intent
from .adapter.uiautomator2_helper import Uiautomator2_Helper
DEFAULT_NUM = '1234567890'
DEFAULT_CONTENT = 'Hello world!'


class Device(object):
    """
    this class describes a connected device
    """

    def __init__(self, device_serial=None, is_emulator=False, output_dir=None,
                 cv_mode=False, grant_perm=False, telnet_auth_token=None,
                 enable_accessibility_hard=False, humanoid=None, ignore_ad=False):
        """
        initialize a device connection
        :param device_serial: serial number of target device
        :param is_emulator: boolean, type of device, True for emulator, False for real device
        :return:
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        if device_serial is None:
            from .utils import get_available_devices
            all_devices = get_available_devices()
            if len(all_devices) == 0:
                self.logger.warning("ERROR: No device connected.")
                sys.exit(-1)
            device_serial = all_devices[0]
        if "emulator" in device_serial and not is_emulator:
            self.logger.warning("Seems like you are using an emulator. If so, please add is_emulator option.")
        self.serial = device_serial
        self.is_emulator = is_emulator
        self.cv_mode = cv_mode
        self.output_dir = output_dir
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
        self.grant_perm = grant_perm
        self.enable_accessibility_hard = enable_accessibility_hard
        self.humanoid = humanoid
        self.ignore_ad = ignore_ad

        # basic device information
        self.settings = {}
        self.display_info = None
        self.model_number = None
        self.sdk_version = None
        self.release_version = None
        self.ro_debuggable = None
        self.ro_secure = None
        self.connected = True
        self.last_know_state = None
        self.__used_ports = []
        self.pause_sending_event = False

        self.u2 = uiautomator2.connect(self.serial)
        # disable keyboard
        self.u2.set_fastinput_ime(True)
        # adapters
        self.adb = ADB(device=self)
        self.telnet = TelnetConsole(device=self, auth_token=telnet_auth_token)
        # self.droidbot_app = DroidBotAppConn(device=self)
        self.minicap = Minicap(device=self)
        self.logcat = Logcat(device=self)
        self.user_input_monitor = UserInputMonitor(device=self)
        self.process_monitor = ProcessMonitor(device=self)
        # self.droidbot_ime = DroidBotIme(device=self)
        self.uiautomator_helper = Uiautomator2_Helper(device=self, package_name="") # will be set later


        self.adapters = {
            self.adb: True,
            self.telnet: False,
            #self.droidbot_app: True,
            self.minicap: True,
            self.logcat: True,
            self.user_input_monitor: True,
            self.process_monitor: True,
            #self.droidbot_ime: True
        }

        # minicap currently not working on emulators
        if self.is_emulator:
            self.logger.info("disable minicap on emulator")
            self.adapters[self.minicap] = False

        # minicap is not supporting android 32 and above
        if self.get_sdk_version() >= 32:
            self.logger.info("disable minicap on sdk >= 32")
            self.adapters[self.minicap] = False

    
    def check_connectivity(self):
        """
        check if the device is available
        """
        for adapter in self.adapters:
            adapter_name = adapter.__class__.__name__
            adapter_enabled = self.adapters[adapter]
            if not adapter_enabled:
                print("[CONNECTION] %s is not enabled." % adapter_name)
            else:
                if adapter.check_connectivity():
                    print("[CONNECTION] %s is enabled and connected." % adapter_name)
                else:
                    print("[CONNECTION] %s is enabled but not connected." % adapter_name)

    def wait_for_device(self):
        """
        wait until the device is fully booted
        :return:
        """
        self.logger.info("waiting for device")
        try:
            subprocess.check_call(["adb", "-s", self.serial, "wait-for-device"])
            # while True:
            #     out = subprocess.check_output(
            #         ["adb", "-s", self.serial, "shell", "getprop", "init.svc.bootanim"]).split()[0]
            #     if not isinstance(out, str):
            #         out = out.decode()
            #     if out == "stopped":
            #         break
            #     time.sleep(1)
        except:
            self.logger.warning("error waiting for device")

    def set_up(self):
        """
        Set connections on this device
        """
        # wait for emulator to start
        self.wait_for_device()
        for adapter in self.adapters:
            adapter_enabled = self.adapters[adapter]
            if not adapter_enabled:
                continue
            adapter.set_up()

    def connect(self):
        """
        establish connections on this device
        :return:
        """
        for adapter in self.adapters:
            adapter_enabled = self.adapters[adapter]
            if not adapter_enabled:
                continue
            adapter.connect()

        self.get_sdk_version()
        self.get_release_version()
        self.get_ro_secure()
        self.get_ro_debuggable()
        self.get_display_info()

        self.unlock()
        self.check_connectivity()
        self.connected = True

    def disconnect(self):
        """
        disconnect current device
        :return:
        """
        self.connected = False
        for adapter in self.adapters:
            adapter_enabled = self.adapters[adapter]
            if not adapter_enabled:
                continue
            adapter.disconnect()

        if self.output_dir is not None:
            temp_dir = os.path.join(self.output_dir, "temp")
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)

    def tear_down(self):
        for adapter in self.adapters:
            adapter_enabled = self.adapters[adapter]
            if not adapter_enabled:
                continue
            adapter.tear_down()

    def is_foreground(self, app):
        """
        check if app is in foreground of device
        :param app: App
        :return: boolean
        """
        if isinstance(app, str):
            package_name = app
        elif isinstance(app, App):
            package_name = app.get_package_name()
        else:
            return False

        top_activity_name = self.get_top_activity_name()
        if top_activity_name is None:
            return False
        return top_activity_name.startswith(package_name)

    def get_model_number(self):
        """
        Get model number
        """
        if self.model_number is None:
            self.model_number = self.adb.get_model_number()
        return self.model_number

    def get_sdk_version(self):
        """
        Get version of current SDK
        """
        if self.sdk_version is None:
            self.sdk_version = self.adb.get_sdk_version()
        return self.sdk_version

    def get_release_version(self):
        """
        Get version of current SDK
        """
        if self.release_version is None:
            self.release_version = self.adb.get_release_version()
        return self.release_version

    def get_ro_secure(self):
        if self.ro_secure is None:
            self.ro_secure = self.adb.get_ro_secure()
        return self.ro_secure

    def get_ro_debuggable(self):
        if self.ro_debuggable is None:
            self.ro_debuggable = self.adb.get_ro_debuggable()
        return self.ro_debuggable

    def get_display_info(self, refresh=True):
        """
        get device display information, including width, height, and density
        :param refresh: if set to True, refresh the display info instead of using the old values
        :return: dict, display_info
        """
        if self.display_info is None or refresh:
            self.display_info = self.adb.get_display_info()
        return self.display_info

    def get_width(self, refresh=False):
        display_info = self.get_display_info(refresh=refresh)
        width = 0
        if "width" in display_info:
            width = display_info["width"]
        elif not refresh:
            width = self.get_width(refresh=True)
        else:
            self.logger.warning("get_width: width not in display_info")
        return width

    def get_height(self, refresh=False):
        display_info = self.get_display_info(refresh=refresh)
        height = 0
        if "height" in display_info:
            height = display_info["height"]
        elif not refresh:
            height = self.get_width(refresh=True)
        else:
            self.logger.warning("get_height: height not in display_info")
        return height

    def unlock(self):
        """
        unlock screen
        skip first-use tutorials
        etc
        :return:
        """
        self.adb.unlock()

    def shake(self):
        """
        shake the device
        """
        # TODO the telnet-simulated shake event is not usable
        telnet = self.telnet
        if telnet is None:
            self.logger.warning("Telnet not connected, so can't shake the device.")
        sensor_xyz = [(-float(v * 10) + 1, float(v) + 9.8, float(v * 2) + 0.5) for v in [1, -1, 1, -1, 1, -1, 0]]
        for (x, y, z) in sensor_xyz:
            telnet.run_cmd("sensor set acceleration %f:%f:%f" % (x, y, z))

    def add_env(self, env):
        """
        set env to the device
        :param env: instance of AppEnv
        """
        self.logger.info("deploying env: %s" % env)
        env.deploy(self)

    def add_contact(self, contact_data):
        """
        add a contact to device
        :param contact_data: dict of contact, should have keys like name, phone, email
        :return:
        """
        assert self.adb is not None
        contact_intent = Intent(prefix="start",
                                action="android.intent.action.INSERT",
                                mime_type="vnd.android.cursor.dir/contact",
                                extra_string=contact_data)
        self.send_intent(intent=contact_intent)
        time.sleep(2)
        self.adb.press("BACK")
        time.sleep(2)
        self.adb.press("BACK")
        return True

    def receive_call(self, phone=DEFAULT_NUM):
        """
        simulate a income phonecall
        :param phone: str, phonenum
        :return:
        """
        assert self.telnet is not None
        return self.telnet.run_cmd("gsm call %s" % phone)

    def cancel_call(self, phone=DEFAULT_NUM):
        """
        cancel phonecall
        :param phone: str, phonenum
        :return:
        """
        assert self.telnet is not None
        return self.telnet.run_cmd("gsm cancel %s" % phone)

    def accept_call(self, phone=DEFAULT_NUM):
        """
        accept phonecall
        :param phone: str, phonenum
        :return:
        """
        assert self.telnet is not None
        return self.telnet.run_cmd("gsm accept %s" % phone)

    def call(self, phone=DEFAULT_NUM):
        """
        simulate a outcome phonecall
        :param phone: str, phonenum
        :return:
        """
        call_intent = Intent(prefix='start',
                             action="android.intent.action.CALL",
                             data_uri="tel:%s" % phone)
        return self.send_intent(intent=call_intent)

    def send_sms(self, phone=DEFAULT_NUM, content=DEFAULT_CONTENT):
        """
        send a SMS
        :param phone: str, phone number of receiver
        :param content: str, content of sms
        :return:
        """
        send_sms_intent = Intent(prefix='start',
                                 action="android.intent.action.SENDTO",
                                 data_uri="sms:%s" % phone,
                                 extra_string={'sms_body': content},
                                 extra_boolean={'exit_on_sent': 'true'})
        self.send_intent(intent=send_sms_intent)
        time.sleep(2)
        self.adb.press('66')
        return True

    def receive_sms(self, phone=DEFAULT_NUM, content=DEFAULT_CONTENT):
        """
        receive a SMS
        :param phone: str, phone number of sender
        :param content: str, content of sms
        :return:
        """
        assert self.telnet is not None
        return self.telnet.run_cmd("sms send %s '%s'" % (phone, content))

    def set_gps(self, x, y):
        """
        set GPS positioning to x,y
        :param x: float
        :param y: float
        :return:
        """
        return self.telnet.run_cmd("geo fix %s %s" % (x, y))

    def set_continuous_gps(self, center_x, center_y, delta_x, delta_y):
        import threading
        gps_thread = threading.Thread(
            target=self.set_continuous_gps_blocked,
            args=(center_x, center_y, delta_x, delta_y))
        gps_thread.start()
        return True

    def set_continuous_gps_blocked(self, center_x, center_y, delta_x, delta_y):
        """
        simulate GPS on device via telnet
        this method is blocked
        @param center_x: x coordinate of GPS position
        @param center_y: y coordinate of GPS position
        @param delta_x: range of x coordinate
        @param delta_y: range of y coordinate
        """
        import random
        while self.connected:
            x = random.random() * delta_x * 2 + center_x - delta_x
            y = random.random() * delta_y * 2 + center_y - delta_y
            self.set_gps(x, y)
            time.sleep(3)

    def get_settings(self):
        """
        get device settings via adb
        """
        db_name = "/data/data/com.android.providers.settings/databases/settings.db"

        system_settings = {}
        out = self.adb.shell("sqlite3 %s \"select * from %s\"" % (db_name, "system"))
        out_lines = out.splitlines()
        for line in out_lines:
            segs = line.split('|')
            if len(segs) != 3:
                continue
            system_settings[segs[1]] = segs[2]

        secure_settings = {}
        out = self.adb.shell("sqlite3 %s \"select * from %s\"" % (db_name, "secure"))
        out_lines = out.splitlines()
        for line in out_lines:
            segs = line.split('|')
            if len(segs) != 3:
                continue
            secure_settings[segs[1]] = segs[2]

        self.settings['system'] = system_settings
        self.settings['secure'] = secure_settings
        return self.settings

    def change_settings(self, table_name, name, value):
        """
        dangerous method, by calling this, change settings.db in device
        be very careful for sql injection
        :param table_name: table name to work on, usually it is system or secure
        :param name: settings name to set
        :param value: settings value to set
        """
        db_name = "/data/data/com.android.providers.settings/databases/settings.db"

        self.adb.shell("sqlite3 %s \"update '%s' set value='%s' where name='%s'\""
                       % (db_name, table_name, value, name))
        return True

    def send_intent(self, intent):
        """
        send an intent to device via am (ActivityManager)
        :param intent: instance of Intent or str
        :return:
        """
        assert self.adb is not None
        assert intent is not None
        if isinstance(intent, Intent):
            cmd = intent.get_cmd()
        else:
            cmd = intent
        return self.adb.shell(cmd)

    def send_event(self, event, probe=False):
        """
        send one event to device
        :param event: the event to be sent
        :param probe: whether in probe mode (only effective for ScrollEvent), uses a smaller scroll distance when True
        :return:
        """
        from .input_event import ScrollEvent
        if isinstance(event, ScrollEvent):
            event.send(self, probe=probe)
        else:
            event.send(self)

    def start_app(self, app):
        """
        start an app on the device
        :param app: instance of App, or str of package name
        :return:
        """
        if isinstance(app, str):
            package_name = app
        elif isinstance(app, App):
            package_name = app.get_package_name()
            if app.get_main_activity():
                package_name += "/%s" % app.get_main_activity()
        else:
            self.logger.warning("unsupported param " + app + " with type: ", type(app))
            return
        intent = Intent(suffix=package_name)
        self.send_intent(intent)

    def get_top_activity_name(self):
        """
        Get current activity
        """
        r = self.adb.shell("dumpsys activity activities")
        activity_line_re = re.compile(r'\*\s*Hist\s*#\d+:\s*ActivityRecord\{[^ ]+\s*[^ ]+\s*([^ ]+)\s*t(\d+)}')
        m = activity_line_re.search(r)
        if m:
            return m.group(1)
        # data = self.adb.shell("dumpsys activity top").splitlines()
        # regex = re.compile("\s*ACTIVITY ([A-Za-z0-9_.]+)/([A-Za-z0-9_.]+)")
        # m = regex.search(data[1])
        # if m:
        #     return m.group(1) + "/" + m.group(2)
        self.logger.warning("Unable to get top activity name.")
        return None

    def get_current_activity_stack(self):
        """
        Get current activity stack
        :return: a list of str, each str is an activity name, the first is the top activity name
        """
        task_to_activities = self.get_task_activities()
        top_activity = self.get_top_activity_name()
        if top_activity:
            for task_id in task_to_activities:
                activities = task_to_activities[task_id]
                if len(activities) > 0 and activities[0] == top_activity:
                    return activities
            self.logger.warning("Unable to get current activity stack.")
            return [top_activity]
        else:
            return None

    def get_task_activities(self):
        """
        Get current tasks and corresponding activities.
        :return: a dict mapping each task id to a list of activities, from top to down.
        """
        task_to_activities = {}

        lines = self.adb.shell("dumpsys activity activities").splitlines()
        activity_line_re = re.compile(r'\*\s*Hist\s*#\d+:\s*ActivityRecord\{[^ ]+\s*[^ ]+\s*([^ ]+)\s*t(\d+)}')

        for line in lines:
            line = line.strip()
            activity_line_task_re = re.compile(r'^\s*Task\s*id\s*#(\d+)|^\s*Task\{\w+\s*#(\d+)')
            activity_line_task_m = activity_line_task_re.match(line)
            if activity_line_task_m:
                if activity_line_task_m.group(1):
                    task_id = activity_line_task_m.group(1)
                else:
                    task_id = activity_line_task_m.group(2)
                task_to_activities[task_id] = []
            elif re.match(r'\*\s*Hist\s*#', line):
                m = activity_line_re.match(line)
                if m:
                    activity = m.group(1)
                    task_id = m.group(2)
                    if task_id not in task_to_activities:
                        task_to_activities[task_id] = []
                    task_to_activities[task_id].append(activity)

        return task_to_activities

    def get_service_names(self):
        """
        get current running services
        :return: list of services
        """
        services = []
        dat = self.adb.shell('dumpsys activity services')
        lines = dat.splitlines()
        service_re = re.compile('^.+ServiceRecord{.+ ([A-Za-z0-9_.]+)/([A-Za-z0-9_.]+)')

        for line in lines:
            m = service_re.search(line)
            if m:
                package = m.group(1)
                service = m.group(2)
                services.append("%s/%s" % (package, service))
        return services

    def get_package_path(self, package_name):
        """
        get installation path of a package (app)
        :param package_name:
        :return: package path of app in device
        """
        dat = self.adb.shell('pm path %s' % package_name)
        package_path_re = re.compile('^package:(.+)$')
        m = package_path_re.match(dat)
        if m:
            path = m.group(1)
            return path.strip()
        return None

    def start_activity_via_monkey(self, package):
        """
        use monkey to start activity
        @param package: package name of target activity
        """
        cmd = 'monkey'
        if package:
            cmd += ' -p %s' % package
        out = self.adb.shell(cmd)
        if re.search(r"(Error)|(Cannot find 'App')", out, re.IGNORECASE | re.MULTILINE):
            raise RuntimeError(out)

    def install_app(self, app):
        """
        install an app to device
        @param app: instance of App
        @return:
        """
        assert isinstance(app, App)
        # subprocess.check_call(["adb", "-s", self.serial, "uninstall", app.get_package_name()],
        #                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        package_name = app.get_package_name()
        self.uiautomator_helper.package_name = package_name
        if package_name not in self.adb.get_installed_apps():
            install_cmd = ["adb", "-s", self.serial, "install", "-r", "-g"]
            if self.grant_perm and self.get_sdk_version() >= 23:
                install_cmd.append("-g")
            install_cmd.append(app.app_path)
            install_p = subprocess.Popen(install_cmd, stdout=subprocess.PIPE)
            while self.connected and package_name not in self.adb.get_installed_apps():
                print("Please wait while installing the app...")
                time.sleep(2)
            if not self.connected:
                install_p.terminate()
                return

        dumpsys_p = subprocess.Popen(["adb", "-s", self.serial, "shell",
                                      "dumpsys", "package", package_name], stdout=subprocess.PIPE)
        dumpsys_lines = []
        while True:
            line = dumpsys_p.stdout.readline()
            if not line:
                break
            if not isinstance(line, str):
                line = line.decode()
            dumpsys_lines.append(line)
        if self.output_dir is not None:
            package_info_file_name = "%s/dumpsys_package_%s.txt" % (self.output_dir, app.get_package_name())
            package_info_file = open(package_info_file_name, "w")
            package_info_file.writelines(dumpsys_lines)
            package_info_file.close()
        app.dumpsys_main_activity = self.__parse_main_activity_from_dumpsys_lines(dumpsys_lines)

        self.logger.info("App installed: %s" % package_name)
        self.logger.info("Main activity: %s" % app.get_main_activity())
    

    @staticmethod
    def __parse_main_activity_from_dumpsys_lines(lines):
        main_activity = None
        activity_line_re = re.compile("[^ ]+ ([^ ]+)/([^ ]+) filter [^ ]+")
        action_re = re.compile("Action: \"([^ ]+)\"")
        category_re = re.compile("Category: \"([^ ]+)\"")

        activities = {}

        cur_package = None
        cur_activity = None
        cur_actions = []
        cur_categories = []

        for line in lines:
            line = line.strip()
            m = activity_line_re.match(line)
            if m:
                activities[cur_activity] = {
                    "actions": cur_actions,
                    "categories": cur_categories
                }
                cur_package = m.group(1)
                cur_activity = m.group(2)
                if cur_activity.startswith("."):
                    cur_activity = cur_package + cur_activity
                cur_actions = []
                cur_categories = []
            else:
                m1 = action_re.match(line)
                if m1:
                    cur_actions.append(m1.group(1))
                else:
                    m2 = category_re.match(line)
                    if m2:
                        cur_categories.append(m2.group(1))

        if cur_activity is not None:
            activities[cur_activity] = {
                "actions": cur_actions,
                "categories": cur_categories
            }

        for activity in activities:
            if "android.intent.action.MAIN" in activities[activity]["actions"] \
                    and "android.intent.category.LAUNCHER" in activities[activity]["categories"]:
                main_activity = activity
        return main_activity

    def uninstall_app(self, app):
        """
        Uninstall an app from device.
        :param app: an instance of App or a package name
        """
        if isinstance(app, App):
            package_name = app.get_package_name()
        else:
            package_name = app
        if package_name in self.adb.get_installed_apps():
            uninstall_cmd = ["adb", "-s", self.serial, "uninstall", package_name]
            uninstall_p = subprocess.Popen(uninstall_cmd, stdout=subprocess.PIPE)
            while package_name in self.adb.get_installed_apps():
                print("Please wait while uninstalling the app...")
                time.sleep(2)
            uninstall_p.terminate()

    def get_app_pid(self, app):
        if isinstance(app, App):
            package = app.get_package_name()
        else:
            package = app

        name2pid = {}
        ps_out = self.adb.shell(["ps"])
        ps_out_lines = ps_out.splitlines()
        ps_out_head = ps_out_lines[0].split()
        if ps_out_head[1] != "PID" or ps_out_head[-1] != "NAME":
            self.logger.warning("ps command output format error: %s" % ps_out_head)
        for ps_out_line in ps_out_lines[1:]:
            segs = ps_out_line.split()
            if len(segs) < 4:
                continue
            pid = int(segs[1])
            name = segs[-1]
            name2pid[name] = pid

        if package in name2pid:
            return name2pid[package]

        possible_pids = []
        for name in name2pid:
            if name.startswith(package):
                possible_pids.append(name2pid[name])
        if len(possible_pids) > 0:
            return min(possible_pids)

        return None

    def push_file(self, local_file, remote_dir="/sdcard/"):
        """
        push file/directory to target_dir
        :param local_file: path to file/directory in host machine
        :param remote_dir: path to target directory in device
        :return:
        """
        if not os.path.exists(local_file):
            self.logger.warning("push_file file does not exist: %s" % local_file)
        self.adb.run_cmd(["push", local_file, remote_dir])

    def pull_file(self, remote_file, local_file):
        self.adb.run_cmd(["pull", remote_file, local_file])

    def take_screenshot(self):
        # image = None
        #
        # received = self.adb.shell("screencap -p").replace("\r\n", "\n")
        # import StringIO
        # stream = StringIO.StringIO(received)
        #
        # try:
        #     from PIL import Image
        #     image = Image.open(stream)
        # except IOError as e:
        #     self.logger.warning("exception in take_screenshot: %s" % e)
        # return image
        if self.output_dir is None:
            return None

        from datetime import datetime
        tag = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        local_image_dir = os.path.join(self.output_dir, "temp")
        if not os.path.exists(local_image_dir):
            os.makedirs(local_image_dir)

        if self.adapters[self.minicap] and self.minicap.last_screen:
            # minicap use jpg format
            local_image_path = os.path.join(local_image_dir, "screen_%s.jpg" % tag)
            with open(local_image_path, 'wb') as local_image_file:
                local_image_file.write(self.minicap.last_screen)
            return local_image_path
        else:
            # screencap use png format
            local_image_path = os.path.join(local_image_dir, "screen_%s.png" % tag)
            remote_image_path = "/sdcard/screen_%s.png" % tag
            self.adb.shell("screencap -p %s" % remote_image_path)
            self.pull_file(remote_image_path, local_image_path)
            self.adb.shell("rm %s" % remote_image_path)

        return local_image_path

    def get_current_state(self):
        self.logger.debug("getting current device state...")
        current_state = None
        try:
            views = self.get_views()
            foreground_activity = self.get_top_activity_name()
            activity_stack = self.get_current_activity_stack()
            background_services = self.get_service_names()
            screenshot_path = self.take_screenshot()
            self.logger.debug("finish getting current device state...")
            from .device_state import DeviceState
            current_state = DeviceState(self,
                                        views=views,
                                        foreground_activity=foreground_activity,
                                        activity_stack=activity_stack,
                                        background_services=background_services,
                                        screenshot_path=screenshot_path)
        except Exception as e:
            self.logger.warning("exception in get_current_state: %s" % e)
            import traceback
            traceback.print_exc()
        self.logger.debug("finish getting current device state...")
        self.last_know_state = current_state
        if not current_state:
            self.logger.warning("Failed to get current state!")
        return current_state

    def get_last_known_state(self):
        return self.last_know_state

    def view_touch(self, x, y):
        self.adb.touch(x, y)

    def view_long_touch(self, x, y, duration=2000):
        """
        Long touches at (x, y)
        @param duration: duration in ms
        This workaround was suggested by U{HaMi<http://stackoverflow.com/users/2571957/hami>}
        """
        self.adb.long_touch(x, y, duration)

    def view_drag(self, start_xy, end_xy, duration):
        """
        Sends drag event n PX (actually it's using C{input swipe} command.
        """
        self.adb.drag(start_xy, end_xy, duration)

    def view_append_text(self, text):
        if self.droidbot_ime.connected:
            self.droidbot_ime.input_text(text=text, mode=1)
        else:
            self.adb.type(text)

    def view_set_text(self, text):
        # if self.droidbot_ime.connected:
        #     self.droidbot_ime.input_text(text=text, mode=0)
        #  else:
        self.logger.warning("`adb shell input text` doesn't support setting text, appending instead.")
        self.adb.type(text)

    def key_press(self, key_code):
        self.adb.press(key_code)

    def shutdown(self):
        self.adb.shell("reboot -p")

    def get_views(self):
        if self.cv_mode and self.adapters[self.minicap]:
            # Get views using cv module
            views = self.minicap.get_views()
            if views:
                return views
            else:
                self.logger.warning("Failed to get views using OpenCV.")
        if self.uiautomator_helper:
            views = self.uiautomator_helper.get_views()
            if views:
                return views
            else:
                self.logger.warning("Failed to get views using UiAutomator.")
        # if self.droidbot_app and self.adapters[self.droidbot_app]:
        #     views = self.droidbot_app.get_views()
        #     if views:
        #         return views
        #     else:
        #         self.logger.warning("Failed to get views using Accessibility.")

        self.logger.warning("failed to get current views!")
        return None

    def get_random_port(self):
        """
        get a random port on host machine to establish connection
        :return: a port number
        """
        import socket
        temp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        temp_sock.bind(("", 0))
        port = temp_sock.getsockname()[1]
        temp_sock.close()
        if port in self.__used_ports:
            return self.get_random_port()
        self.__used_ports.append(port)
        return port

    def handle_rotation(self):
        if not self.adapters[self.minicap]:
            return
        self.pause_sending_event = True
        if self.minicap.check_connectivity():
            self.minicap.disconnect()
            self.minicap.connect()

        if self.minicap.check_connectivity():
            print("[CONNECTION] %s is reconnected." % self.minicap.__class__.__name__)
        self.pause_sending_event = False


    def install_files(self, app_package=None):
        """
        Install the files to the device for the app
        """
        if "com.atul.musicplayer" in app_package or "com.github.anrimian.musicplayer" in app_package or "remix.myplayer" in app_package or "org.gateshipone.odyssey" in app_package or "com.github.pakka_papad" in app_package or "code.name.monkey.retromusic" in app_package or "org.oxycblt.auxio" in app_package or "com.arn.scrobble" in app_package or "com.zionhuang.music" in app_package or "player.phonograph.plus" in app_package:
            self.push_file("droidbot/Document/Heartbeat.mp3", "/sdcard/Music/")
            self.push_file("droidbot/Document/intermission.mp3", "/sdcard/Music/")
            self.push_file("droidbot/Document/ThingsWeCantIgnore.mp3", "/sdcard/Music/")
            self.push_file("droidbot/Document/hip-hop.mp3", "/sdcard/Music/")
            self.push_file("droidbot/Document/puff-pass.mp3", "/sdcard/Music/")
            self.adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Music/Heartbeat.mp3")
            self.adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Music/intermission.mp3")
            self.adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Music/ThingsWeCantIgnore.mp3")
            self.adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Music/hip-hop.mp3")
            self.adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Music/puff-pass.mp3")

        elif "com.android.documentsui" in app_package or "exe.bbllw8.anemo" in app_package or "is.xyz.mpv" in app_package:
            # 1. mp3
            self.push_file("droidbot/Document/Heartbeat.mp3", "/sdcard/Music/")
            self.push_file("droidbot/Document/intermission.mp3", "/sdcard/Music/")

            # 2. image
            self.push_file("droidbot/Document/Android_logo.jpg", "/sdcard/Pictures/")
            self.push_file("droidbot/Document/Android_robot.png", "/sdcard/Pictures/")

            # 3. video
            self.push_file("droidbot/Document/sample_iPod.m4v", "/sdcard/Movies/")
            self.push_file("droidbot/Document/sample_mpeg4.mp4", "/sdcard/Movies/")

            # 4. file
            self.push_file("droidbot/Document/test.pdf", "/sdcard/Documents/")
            self.push_file("droidbot/Document/test.pdf", "/sdcard/Download/")
            self.push_file("droidbot/Document/hip-hop.mp3", "/sdcard/Download/")

            self.adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Music/Heartbeat.mp3")
            self.adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Music/intermission.mp3")
            self.adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Pictures/Android_logo.jpg")
            self.adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Pictures/Android_robot.png")
            self.adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Movies/sample_iPod.m4v")
            self.adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Movies/sample_mpeg4.mp4")
            self.adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Documents/test.pdf")
            self.adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Download/test.pdf")
            self.adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Download/hip-hop.mp3")


        elif app_package == "universe.constellation.orion.viewer":
            # place files in the download directory

            # 1. image
            self.push_file("droidbot/Document/Android_logo.jpg", "/sdcard/Download/")
            self.push_file("droidbot/Document/Android_robot.png", "/sdcard/Download/")

            # 2. pdf
            self.push_file("droidbot/Document/test.pdf", "/sdcard/Download/")
            self.adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Download/Android_logo.jpg")
            self.adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Download/Android_robot.png")
            self.adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Download/test.pdf")




    
    def skip_welcome(self, app_package=None):
        """
        Try to skip the app's welcome page
        :param app_package: the package name of the app, uses current app if None
        :return: whether the welcome page was successfully skipped

        Needs to be handled on a case-by-case basis
        """

        if app_package == "it.feio.android.omninotes" or app_package == "net.gsantner.markor":
            self.skip_omninotes_welcome()
        elif app_package == "com.ichi2.anki":
            self.skip_anki_welcome()
        elif app_package == "com.vrem.wifianalyzer":
            self.skip_wifianalyzer_welcome()
        elif app_package == "com.best.deskclock":
            self.skip_deskclock_welcome()
        elif app_package == "org.secuso.privacyfriendlytodolist":
            self.skip_todolist_welcome()
        elif app_package == "org.billthefarmer.diary":
            self.skip_diary_welcome()
        elif "com.quran.labs.androidquran" in app_package:
            self.skip_quran_welcome()
        elif app_package == "com.red.alert":
            self.skip_redalert_welcome()
        elif app_package == "io.github.muntashirakon.AppManager":
            self.skip_appmanager_welcome()
        elif app_package == "com.mxt.anitrend":
            self.skip_anitrend_welcome()
        elif app_package == "com.michaldrabik.showly2":
            self.skip_showly2_welcome()
        elif "com.atul.musicplayer" in app_package:
            self.skip_musicplayer_welcome()
        elif app_package == "org.zephyrsoft.trackworktime":
            self.skip_trackworktime_welcome()
        elif app_package == "com.android.documentsui" or app_package == "exe.bbllw8.anemo":
            self.skip_documentsui_welcome()
        elif app_package == "de.westnordost.streetcomplete":
            self.skip_streetcomplete_welcome()
        elif app_package == "org.isoron.uhabits":
            self.skip_uhabits_welcome()
        elif app_package == "com.mirfatif.permissionmanagerx":
            self.skip_permissionmanagerx_welcome()
        elif app_package == "org.secuso.privacyfriendlynotes":
            self.skip_privacyfriendlynotes_welcome()
        elif app_package == "eu.faircode.email":
            self.skip_email_welcome()
        elif app_package == "com.farmerbb.notepad":
            self.skip_notepad_welcome()
        elif app_package == "com.inspiredandroid.linuxcommandbibliotheca":
            self.skip_linuxcommandbibliotheca_welcome()
        elif app_package == "remix.myplayer" or app_package == "me.zhanghai.android.files":
            self.skip_myplayer_welcome()
        elif app_package == "org.gateshipone.odyssey":
            self.skip_odyssey_welcome()
        elif app_package == "net.bible.android.activity":
            self.skip_bible_welcome()
        elif app_package == "com.github.pakka_papad":
            self.skip_pakka_welcome()
        elif app_package == "com.cookiegames.smartcookie":
            self.skip_smartcookie_welcome()
        elif app_package == "code.name.monkey.retromusic":
            self.skip_retromusic_welcome()
        elif app_package == "org.koitharu.kotatsu":
            self.skip_kotatsu_welcome()
        elif "ua.syt0r.kanji" in app_package:
            self.skip_kanji_welcome()
        elif app_package == "org.schabi.newpipe":
            self.skip_newpipe_welcome()
        elif app_package == "com.onlyoffice.documents":
            self.skip_documents_welcome()
        elif app_package == "com.appmindlab.nano":
            self.skip_nano_welcome()
        elif app_package == "at.jclehner.rxdroid":
            self.skip_rxdroid_welcome()
        elif app_package == "player.phonograph.plus":
            self.skip_phonograph_welcome()
        elif app_package == "free.rm.skytube.oss":
            self.skip_skytube_welcome()
        else:
            return

    def skip_skytube_welcome(self):
        """
        Skip the SkyTube welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            skip_button = self.u2(text="OK")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            skip_button = self.u2(text="LATER")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            # Refresh x3
            for _ in range(3):
                try:
                    # pull-down to refresh -- based on standard screen ratio
                    self.u2.swipe(0.5, 0.25, 0.5, 0.85, 0.2)
                    time.sleep(0.8)  # give the page time to refresh
                except Exception as swipe_err:
                    print(f"Refresh swipe failed: {swipe_err}")
            
           
            return True
        except Exception as e:
            print(f"Error during Phonograph welcome screen skip: {e}")
            return False
           

    def skip_phonograph_welcome(self):
        """
        Skip the Phonograph welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            for i in range(6):
                skip_button = self.u2(description="NEXT")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)

                skip_button = self.u2(text="DONE")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)

                skip_button = self.u2(text="disable")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)

                skip_button = self.u2(text="GET STARTED")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)

                i += 1

            skip_button = self.u2(text="OK")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            return True
        except Exception as e:
            print(f"Error during Phonograph welcome screen skip: {e}")
            return False

    def skip_rxdroid_welcome(self):
        """
        Skip the Rxdroid welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            skip_button = self.u2(text="OK")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)
            return True
        except Exception as e:
            print(f"Error during Rxdroid welcome screen skip: {e}")
            return False

    def skip_nano_welcome(self):
        """
        Skip the Nano welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            skip_button = self.u2(text="SKIP")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            skip_button = self.u2(text="ACCEPT")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            skip_button = self.u2(text="USE DEFAULT")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            skip_button = self.u2(text="OK")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            skip_button = self.u2(text="Allow")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)
            return True
        except Exception as e:
            print(f"Error during Nano welcome screen skip: {e}")
            return False

    def skip_documents_welcome(self):

        """
        Skip the Documents welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click Allow
            for i in range(6):
                skip_button = self.u2(text="Next")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)

                skip_button = self.u2(text="NEXT")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)

                skip_button = self.u2(text="Get started")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)

                skip_button = self.u2(text="DONE")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)
                    
                skip_button = self.u2(text="Allow")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)


                i += 1


            return True
        except Exception as e:
            print(f"Error during Documents welcome screen skip: {e}")
            return False

    def skip_newpipe_welcome(self):
        """
        Skip the NewPipe welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            skip_button = self.u2(text="RETRY")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)
            return True
        except Exception as e:
            print(f"Error during NewPipe welcome screen skip: {e}")
            return False

    def skip_kanji_welcome(self):

        """
        Skip the Kanji welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            skip_button = self.u2(text="Close")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            return True
        except Exception as e:
            print(f"Error during Kanji welcome screen skip: {e}")
            return False

    def skip_kotatsu_welcome(self):
        """
        Skip the Kotatsu welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            
            if self.u2(text="Welcome").exists:
                flag1 = False
                skip_button = self.u2(text="Done")
                if skip_button.exists() and flag1 == False:
                    skip_button.click()
                    time.sleep(0.5)
                    flag1 = True

                skip_button = self.u2(text="DONE")
                if skip_button.exists() and flag1 == False:
                    skip_button.click()
                    time.sleep(0.5)
                    flag1 = True

                if flag1 != True: # special handling for welcome screen
                    self.u2.press("BACK")
                    time.sleep(0.5)

            
            flag2 = False # special handling for update page
            skip_button = self.u2(text="Close")
            if skip_button.exists() and flag2 == False:
                skip_button.click()
                time.sleep(0.5)
                flag2 = True

            skip_button = self.u2(text="Cancel")
            if skip_button.exists() and flag2 == False:
                skip_button.click()
                time.sleep(0.5)
                flag2 = True

            return True
        except Exception as e:
            print(f"Error during Kotatsu welcome screen skip: {e}")
            return False

    def skip_retromusic_welcome(self):

        """
        Skip the RetroMusic welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Back
            self.u2.press("BACK")
            time.sleep(0.5)

            
            return True
        except Exception as e:
            print(f"Error during RetroMusic welcome screen skip: {e}")
            return False

    def skip_smartcookie_welcome(self):
        """
        Skip the SmartCookie welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            for i in range(6):
                skip_button = self.u2(description="NEXT")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)

                skip_button = self.u2(description="DONE")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)

                skip_button = self.u2(text="Allow")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)

                skip_button = self.u2(description="Get started!")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)

                i += 1

            return True
        except Exception as e:
            print(f"Error during SmartCookie welcome screen skip: {e}")
            return False

    def skip_pakka_welcome(self):
        """
        Skip the Pakka welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            skip_button = self.u2(text="Grant access to read storage")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            skip_button = self.u2(text="Allow")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            skip_button = self.u2(text="Nexts")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            skip_button = self.u2(text="Scan storage")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            skip_button = self.u2(text="Finish")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            return True
        except Exception as e:
            print(f"Error during Pakka welcome screen skip: {e}")
            return False


            
    def skip_bible_welcome(self):
        """
        Skip the Odyssey welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click Allow
            skip_button = self.u2(text="DOWNLOAD DOCUMENTS")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)
            
            skip_button = self.u2(text="YES, DO NOT ASK AGAIN")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            time.sleep(10)
            
            return True
        except Exception as e:
            print(f"Error during Odyssey welcome screen skip: {e}")
            return False
    
    
    def skip_odyssey_welcome(self):
        """
        Skip the Odyssey welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click Allow
            skip_button = self.u2(text="Allow")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)
            
            skip_button = self.u2(text="Cancel")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            skip_button = self.u2(text="CANCEL")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)


            
            return True
        except Exception as e:
            print(f"Error during Odyssey welcome screen skip: {e}")
            return False

    def skip_myplayer_welcome(self):
        """
        Skip the Calendar welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click Allow
            skip_button = self.u2(text="Allow")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)
            return True
        except Exception as e:
            print(f"Error during Calendar welcome screen skip: {e}")
            return False


    def skip_linuxcommandbibliotheca_welcome(self):
        """
        Skip the LinuxCommandBibliotheca welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click Allow
            skip_button = self.u2(text="OK")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)
            return True
        except Exception as e:
            print(f"Error during LinuxCommandBibliotheca welcome screen skip: {e}")
            return False

    def skip_notepad_welcome(self):
        """
        Skip the Notepad welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click Allow
            skip_button = self.u2(text="CLOSE")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            return True
        except Exception as e:
            print(f"Error during Notepad welcome screen skip: {e}")
            return False

    def skip_email_welcome(self):
        """
        Skip the Email welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click Allow
            skip_button = self.u2(text="I agree")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)
            return True
        except Exception as e:
            print(f"Error during Email welcome screen skip: {e}")
            return False

    def skip_privacyfriendlynotes_welcome(self):
        """
        Skip the PrivacyFriendlyNotes welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click Allow
            skip_button = self.u2(text="OKAY")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            skip_button = self.u2(text="Skip")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            skip_button = self.u2(text="SKIP")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            

            return True
        except Exception as e:
            print(f"Error during PrivacyFriendlyNotes welcome screen skip: {e}")
            return False

    def skip_permissionmanagerx_welcome(self):
        """
        Skip the PermissionManagerX welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click Allow
            skip_button = self.u2(text="OK")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            return True
        except Exception as e:  
            print(f"Error during PermissionManagerX welcome screen skip: {e}")
            return False

    def skip_uhabits_welcome(self):
        """
        Skip the UHabits welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click Allow
            for i in range(5):
                # skip_button = self.u2(description="NEXT")
                # if skip_button.exists():
                #     skip_button.click()
                #     time.sleep(0.5)

                # org.isoron.uhabits:id/next
                skip_button = self.u2(resourceId="org.isoron.uhabits:id/next")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)

                # skip_button = self.u2(description="DONE")
                # if skip_button.exists():
                #     skip_button.click()
                #     time.sleep(0.5)
                #     break

                # org.isoron.uhabits:id/done
                skip_button = self.u2(resourceId="org.isoron.uhabits:id/done")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)
                    break

                i += 1

            return True
        except Exception as e:
            print(f"Error during UHabits welcome screen skip: {e}")
            return False

    def skip_streetcomplete_welcome(self):
        """
        Skip the StreetComplete welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click Next


            for i in range(8):
                skip_button = self.u2(text="Next")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)

                skip_button = self.u2(text="NEXT")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)

                skip_button = self.u2(text="Allow only while using the app")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)

                skip_button = self.u2(text="Let’s go!")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)


                skip_button = self.u2(text="LET’S GO!")
                if skip_button.exists():
                    skip_button.click()
                    time.sleep(0.5)
                
                i += 1

            time.sleep(10)

            return True
        except Exception as e:
            print(f"Error during StreetComplete welcome screen skip: {e}")
            return False

    def skip_documentsui_welcome(self):
        """
        Skip the DocumentsUI welcome/onboarding screens
        1. click "OK"
        """
        try:
            # Step 1: click Allow
            skip_button = self.u2(text="OK")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            return True
        except Exception as e:
            print(f"Error during DocumentsUI welcome screen skip: {e}")
            return False

    def skip_trackworktime_welcome(self):
        """
        Skip the TrackWorkTime welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            skip_button = self.u2(text="NOT NOW")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            # Step 1: click Allow
            skip_button = self.u2(text="OK")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            # Step 2: click Allow
            skip_button = self.u2(text='ALLOW ACCESS TO “DOWNLOADS”')
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            # Step 3: click Allow
            skip_button = self.u2(text="ALLOW")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            skip_button = self.u2(text="NO")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            return True
        except Exception as e:
            print(f"Error during TrackWorkTime welcome screen skip: {e}")
            return False

    def skip_musicplayer_welcome(self):
        """
        Skip the MusicPlayer welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click Allow
            skip_button = self.u2(text="Allow")
            if skip_button.exists():
                skip_button.click()
                time.sleep(1)

            return True
        except Exception as e:
            print(f"Error during MusicPlayer welcome screen skip: {e}")
            return False

    def skip_showly2_welcome(self):
        """
        Skip the Showly2 welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click Allow
            skip_button = self.u2(text="OK, I UNDERSTAND")
            if skip_button.exists():
                skip_button.click()
                time.sleep(1)

            return True
        except Exception as e:
            print(f"Error during Showly2 welcome screen skip: {e}")
            return False

            
    
    def skip_anitrend_welcome(self):
        """
        Skip the AniTrend welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click next

            for i in range(4):
                next_button = self.u2(resourceId="com.mxt.anitrend:id/ivNext")
                if next_button.exists():
                    next_button.click()
                    time.sleep(0.5)


            skip_button = self.u2(text="Get Started")
            if skip_button.exists():
                skip_button.click()
                time.sleep(1)

            # press system Back button
            self.u2.press("BACK")
            time.sleep(0.5)

            self.u2.press("BACK")
            time.sleep(0.5)

            return True
        except Exception as e:
            print(f"Error during AniTrend welcome screen skip: {e}")
            return False
    def skip_appmanager_welcome(self):
        """
        Skip the AppManager welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click Allow
            skip_button = self.u2(text="I AGREE")
            if skip_button.exists():
                skip_button.click()
                time.sleep(1)

            return True
        except Exception as e:
            print(f"Error during AppManager welcome screen skip: {e}")
            return False

    def skip_redalert_welcome(self):
        """
        Skip the RedAlert welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click Allow
            skip_button = self.u2(text="Not Now")
            if skip_button.exists():
                skip_button.click()
                time.sleep(2)

            skip_button = self.u2(text="OK")
            if skip_button.exists():
                skip_button.click()
                time.sleep(1)

            return True
        except Exception as e:
            print(f"Error during RedAlert welcome screen skip: {e}")
            return False

    def skip_quran_welcome(self):
        """
        Skip the Quran welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click Allow
            skip_button = self.u2(text="Yes")
            if skip_button.exists():
                skip_button.click()
                time.sleep(1)

            skip_button = self.u2(text="YES")
            if skip_button.exists():
                skip_button.click()
                time.sleep(1)

            time.sleep(120)
            return True
        except Exception as e:
            print(f"Error during Quran welcome screen skip: {e}")

    def skip_diary_welcome(self):
        """
        Skip the Diary welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click Allow
            skip_button = self.u2(text="Allow")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            skip_button = self.u2(text="Allow")
            if skip_button.exists():
                skip_button.click()
                time.sleep(0.5)

            
        except Exception as e:
            print(f"Error during Diary welcome screen skip: {e}")


    def skip_todolist_welcome(self):
        """
        Skip the TodoList welcome/onboarding screens
        1. click "Get Started"
        """
        try:
            # Step 1: click SKIP
            skip_button = self.u2(text="SKIP")
            if skip_button.exists():
                skip_button.click()

            return True
        except Exception as e:
            print(f"Error during TodoList welcome screen skip: {e}")
            return False

    def skip_deskclock_welcome(self):
        """
        Skip the DeskClock welcome/onboarding screens
        1. scroll
        2. click "Later"
        """
        try:
            # Step 1: Scroll
            self.u2.swipe(0.5, 0.8, 0.5, 0.2) 
            self.u2.swipe(0.5, 0.8, 0.5, 0.2)

            # Step 2: Click "Later"
            later_button = self.u2(text="Later")
            if later_button.exists():
                later_button.click()

            return True
        except Exception as e:
            print(f"Error during DeskClock welcome screen skip: {e}")
            return False

    def skip_anki_welcome(self):
        """
        Skip the Anki welcome/onboarding screens

        Steps:
        //*[@resource-id="com.ichi2.anki:id/get_started"]
        //*[@resource-id="com.ichi2.anki:id/switch_widget"]
        ALLOW
        //*[@resource-id="com.ichi2.anki:id/continue_button"]
        Back
        """
        try:
            start_time = time.time()
            # Maximum number of attempts to prevent infinite loop
            max_attempts = 6
            current_attempt = 0

            # if OK button exists, click it first
            ok_button = self.u2(text="OK")
            if ok_button.exists():
                ok_button.click()

            allow_button = self.u2(text="Allow")
            if allow_button.exists():
                allow_button.click()


            # Step 1: Click "Get Started"
            get_started_button = self.u2(text="Get Started")
            if get_started_button.exists():
                get_started_button.click()


            # Step 2: Click "Switch Widget"
            switch_widget_button = self.u2(resourceId="com.ichi2.anki:id/switch_widget")
            if switch_widget_button.exists():
                switch_widget_button.click()

            # Step 3: Handle permission dialogs
            permission_texts = ["OK", "ALLOW", "允许", "确定"]
            for text in permission_texts:
                permission_button = self.u2(text=text)
                if permission_button.exists():
                    permission_button.click()
                    break

            # Step 4: Click "Continue"
            continue_button = self.u2(text="Continue")
            if continue_button.exists():
                continue_button.click()


            end_time = time.time()
            print(f"skip_anki_welcome time: {end_time - start_time} seconds")

        except Exception as e:
            print(f"Error during Anki welcome screen skip: {e}")
            return False

    def skip_omninotes_welcome(self):
        """
        Skip the OmniNotes welcome/onboarding screens
        
        Steps:
        1. Repeatedly click the 'next' button
            # //*[@resource-id="it.feio.android.omninotes:id/next"]
        2. Finally click the 'done' button to complete onboarding
            # //*[@resource-id="it.feio.android.omninotes:id/done"]
        omninotes 4.7.2:
        continue
        OK
        Not now

        """
        try:
            max_attempts = 8
            current_attempt = 0

            continue_button = self.u2(text="continue")
            if continue_button.exists():
                continue_button.click()

            # if OK button exists, click it first
            ok_button = self.u2(text="OK")
            if ok_button.exists():
                ok_button.click()

            not_now_button = self.u2(text="Not now")
            if not_now_button.exists():
                not_now_button.click()

            while current_attempt < max_attempts:
                
                # fuzzy search for resource IDs containing 'next'
                next_buttons = self.u2(resourceIdMatches=".*next")
                # fuzzy search for resource IDs containing 'done'
                done_buttons = self.u2(resourceIdMatches=".*done")
                
                
                # prioritize handling the 'next' button
                if next_buttons.count > 0:
                    next_buttons[0].click()
                    time.sleep(0.5)  # increase wait time
                    current_attempt += 1
                    continue
                
                # if 'done' button is found
                if done_buttons.count > 0:
                    done_buttons[0].click()
                    time.sleep(0.5)
                    break
                
                # if no buttons are found, wait and retry
                time.sleep(0.5)
                current_attempt += 1
            
            
            # if OK button exists, click it
            ok_button = self.u2(text="OK")
            if ok_button.exists():
                ok_button.click()
            
            return True
        except Exception as e:
            print(f"Error during OmniNotes welcome screen skip: {e}")
            return False

    def skip_wifianalyzer_welcome(self):
        """
        Skip the WiFi Analyzer welcome/onboarding screens

        Steps:
        1. cliick "OK"
        2. click "Allow only while using the app"
        """
        try:

            # Step 1: Click "OK"
            ok_button = self.u2(text="OK")
            if ok_button.exists():
                ok_button.click()
                time.sleep(1)

            # Step 2: Click "Allow only while using the app"
            allow_button = self.u2(text="Allow only while using the app")
            if allow_button.exists():
                allow_button.click()
                time.sleep(1)

        except Exception as e:
            print(f"Error during WiFi Analyzer welcome screen skip: {e}")
            return False
