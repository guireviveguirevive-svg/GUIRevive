import json
import logging
import subprocess
import time

from .input_event import EventLog
from .input_policy import RANDOM_EXPLORATION, RandomExplorationPolicy, UtgBasedInputPolicy, UtgNaiveSearchPolicy, UtgGreedySearchPolicy, \
                         UtgReplayPolicy, \
                         POLICY_NAIVE_DFS, POLICY_GREEDY_DFS, \
                         POLICY_NAIVE_BFS, POLICY_GREEDY_BFS, \
                         POLICY_REPLAY, POLICY_MEMORY_GUIDED, POLICY_LLM_GUIDED, \
                         POLICY_MANUAL, POLICY_MONKEY, POLICY_NONE, GROUND_TRUTH, GroundTruthPolicy, \
                         MATCHING, MatchingPolicy, \
                         GUIDER, GuiderPolicy

DEFAULT_POLICY = RANDOM_EXPLORATION
DEFAULT_EVENT_INTERVAL = 1
DEFAULT_EVENT_COUNT = 100000000
DEFAULT_TIMEOUT = -1


class UnknownInputException(Exception):
    pass


class InputManager(object):
    """
    This class manages all events to send during app running
    """

    def __init__(self, device, app, policy_name, random_input,
                 event_count, event_interval,
                 script_path=None, profiling_method=None, master=None,
                 replay_output=None, ground_truth_path=None, failed_replay_output=None, output_dir=None,
                 without_taxonomy=False, without_rule=False, without_llm=False,
                 without_next_screen_summary=False, without_history_summary=False):
        """
        manage input event sent to the target device
        :param device: instance of Device
        :param app: instance of App
        :param policy_name: policy of generating events, string
        :return:
        """
        self.logger = logging.getLogger('InputEventManager')
        self.enabled = True

        self.device = device
        self.app = app
        self.policy_name = policy_name
        self.random_input = random_input
        self.events = []
        self.policy = None
        self.script = None
        self.event_count = event_count
        self.event_interval = event_interval
        self.replay_output = replay_output
        self.ground_truth_path = ground_truth_path
        self.failed_replay_output = failed_replay_output
        self.output_dir = output_dir
        self.monkey = None

        # Ablation experiment parameters
        self.without_taxonomy = without_taxonomy
        self.without_rule = without_rule
        self.without_llm = without_llm
        self.without_next_screen_summary = without_next_screen_summary
        self.without_history_summary = without_history_summary

        if script_path is not None:
            f = open(script_path, 'r')
            script_dict = json.load(f)
            from .input_script import DroidBotScript
            self.script = DroidBotScript(script_dict)

        self.policy = self.get_input_policy(device, app, master)
        self.profiling_method = profiling_method

    def get_input_policy(self, device, app, master):
        if self.policy_name == POLICY_NONE:
            input_policy = None
        elif self.policy_name == RANDOM_EXPLORATION:
            input_policy = RandomExplorationPolicy(device, app, self.random_input )
        elif self.policy_name == POLICY_MONKEY:
            input_policy = None
        elif self.policy_name in [POLICY_NAIVE_DFS, POLICY_NAIVE_BFS]:
            input_policy = UtgNaiveSearchPolicy(device, app, self.random_input, self.policy_name)
        elif self.policy_name in [POLICY_GREEDY_DFS, POLICY_GREEDY_BFS]:
            input_policy = UtgGreedySearchPolicy(device, app, self.random_input, self.policy_name)
        elif self.policy_name == POLICY_MEMORY_GUIDED:
            from .input_policy2 import MemoryGuidedPolicy
            input_policy = MemoryGuidedPolicy(device, app, self.random_input)
        elif self.policy_name == POLICY_LLM_GUIDED:
            from .input_policy3 import LLM_Guided_Policy
            input_policy = LLM_Guided_Policy(device, app, self.random_input)
        elif self.policy_name == POLICY_REPLAY:
            input_policy = UtgReplayPolicy(device, app, self.replay_output)
        elif self.policy_name == GROUND_TRUTH:
            input_policy = GroundTruthPolicy(device, app, self.replay_output, self.ground_truth_path)
        elif self.policy_name == MATCHING:
            input_policy = MatchingPolicy(device, app, self.replay_output, self.failed_replay_output, self.output_dir,
                                          without_taxonomy=self.without_taxonomy,
                                          without_rule=self.without_rule,
                                          without_llm=self.without_llm,
                                          without_next_screen_summary=self.without_next_screen_summary,
                                          without_history_summary=self.without_history_summary)

        elif self.policy_name == GUIDER:
            input_policy = GuiderPolicy(device, app, self.replay_output, self.failed_replay_output, self.output_dir)
        else:
            self.logger.warning("No valid input policy specified. Using policy \"none\".")
            input_policy = None
        if isinstance(input_policy, UtgBasedInputPolicy):
            input_policy.script = self.script
            input_policy.master = master
        return input_policy

    def add_event(self, event, action_count):
        """
        add one event to the event list
        :param event: the event to be added, should be subclass of AppEvent
        :param action_count: the current action count for logging
        :return:
        """
        if event is None:
            return
        self.events.append(event)

        event_log = EventLog(self.device, self.app, event, self.profiling_method)
        event_log.tag = str(action_count+1)
        event_log.start()
        while True:
            time.sleep(self.event_interval)
            if not self.device.pause_sending_event:
                break
        event_log.stop()

    def start(self):
        """
        start sending event
        """
        self.logger.info("start sending events, policy is %s" % self.policy_name)

        try:
            if self.policy is not None:
                self.policy.start(self)
            elif self.policy_name == POLICY_NONE:
                self.device.start_app(self.app)
                if self.event_count == 0:
                    return
                while self.enabled:
                    time.sleep(1)
            elif self.policy_name == POLICY_MONKEY:
                throttle = self.event_interval * 1000
                monkey_cmd = "adb -s %s shell monkey %s --ignore-crashes --ignore-security-exceptions" \
                             " --throttle %d -v %d" % \
                             (self.device.serial,
                              "" if self.app.get_package_name() is None else "-p " + self.app.get_package_name(),
                              throttle,
                              self.event_count)
                self.monkey = subprocess.Popen(monkey_cmd.split(),
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE)
                for monkey_out_line in iter(self.monkey.stdout.readline, ''):
                    if not isinstance(monkey_out_line, str):
                        monkey_out_line = monkey_out_line.decode()
                    self.logger.info(monkey_out_line)
                # may be disturbed from outside
                if self.monkey is not None:
                    self.monkey.wait()
            elif self.policy_name == POLICY_MANUAL:
                self.device.start_app(self.app)
                while self.enabled:
                    keyboard_input = input("press ENTER to save current state, type q to exit...")
                    if keyboard_input.startswith('q'):
                        break
                    state = self.device.get_current_state()
                    if state is not None:
                        state.save2dir()
        except KeyboardInterrupt:
            pass

        self.stop()
        self.logger.info("Finish sending events")

    def stop(self):
        """
        stop sending event
        """
        if self.monkey:
            if self.monkey.returncode is None:
                self.monkey.terminate()
            self.monkey = None
            pid = self.device.get_app_pid("com.android.commands.monkey")
            if pid is not None:
                self.device.adb.shell("kill -9 %d" % pid)
        self.enabled = False

