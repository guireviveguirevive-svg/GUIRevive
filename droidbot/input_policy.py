import sys
import json
import logging
import random
from abc import abstractmethod
import glob
import time
import os
from lxml import etree as ET
from droidbot.device_state import DeviceState

from .input_event import InputEvent, KeyEvent, IntentEvent, TouchEvent, ManualEvent, SetTextEvent, KillAppEvent
from .utg import UTG
from .utils import generate_html_report
from .UIMatch.Matcher import Matcher
from .UIMatch.Logger import get_logger
from .UIMatch.utils import read_image, compute_ssim, draw_original_element_on_image, draw_replay_element_on_image, get_encoded_image, openai_chat

# Max number of restarts
MAX_NUM_RESTARTS = 5
# Max number of steps outside the app
MAX_NUM_STEPS_OUTSIDE = 5
MAX_NUM_STEPS_OUTSIDE_KILL = 10
# Max number of replay tries
MAX_REPLY_TRIES = 5

# Some input event flags
EVENT_FLAG_STARTED = "+started"
EVENT_FLAG_START_APP = "+start_app"
EVENT_FLAG_STOP_APP = "+stop_app"
EVENT_FLAG_EXPLORE = "+explore"
EVENT_FLAG_NAVIGATE = "+navigate"
EVENT_FLAG_TOUCH = "+touch"

# Policy taxanomy
RANDOM_EXPLORATION = "random_exploration"
MATCHING = "matching"
GUIDER = "guider"
GROUND_TRUTH = "ground_truth"
POLICY_NAIVE_DFS = "dfs_naive"
POLICY_GREEDY_DFS = "dfs_greedy"
POLICY_NAIVE_BFS = "bfs_naive"
POLICY_GREEDY_BFS = "bfs_greedy"
POLICY_REPLAY = "replay"
POLICY_MANUAL = "manual"
POLICY_MONKEY = "monkey"
POLICY_NONE = "none"
POLICY_MEMORY_GUIDED = "memory_guided"  # implemented in input_policy2
POLICY_LLM_GUIDED = "llm_guided"  # implemented in input_policy3


class InputInterruptedException(Exception):
    pass


class InputPolicy(object):
    """
    This class is responsible for generating events to stimulate more app behaviour
    It should call AppEventManager.send_event method continuously
    """

    def __init__(self, device, app):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.app = app
        self.action_count = 0
        self.master = None
        self.input_manager=None

    def start(self, input_manager):
        """
        start producing events
        :param input_manager: instance of InputManager
        """
        self.action_count = 0
        self.input_manager = input_manager
        while input_manager.enabled and self.action_count < input_manager.event_count:
            try:
                # # make sure the first event is go to HOME screen
                # # the second event is to start the app
                # if self.action_count == 0 and self.master is None:
                #     event = KeyEvent(name="HOME")
                # elif self.action_count == 1 and self.master is None:
                #     event = IntentEvent(self.app.get_start_intent())
                if self.action_count == 0 and self.master is None:
                    event = KillAppEvent(app=self.app)
                    self.device.install_files(self.app.get_package_name())
                else:
                    # Attempt to skip welcome screen immediately after app launch
                    if self.action_count == 2:
                        self.logger.info("App started,: attempting to skip welcome screen...")
                        self.device.skip_welcome(self.app.get_package_name())
                    event = self.generate_event()
                input_manager.add_event(event, self.action_count)
            except KeyboardInterrupt:
                break
            except InputInterruptedException as e:
                self.logger.warning("stop sending events: %s" % e)
                break
            # except RuntimeError as e:
            #     self.logger.warning(e.message)
            #     break
            except Exception as e:
                self.logger.warning("exception during sending events: %s" % e)
                import traceback
                traceback.print_exc()
                continue
            self.action_count += 1

    @abstractmethod
    def generate_event(self):
        """
        generate an event
        @return:
        """
        pass


class NoneInputPolicy(InputPolicy):
    """
    do not send any event
    """

    def __init__(self, device, app):
        super(NoneInputPolicy, self).__init__(device, app)

    def generate_event(self):
        """
        generate an event
        @return:
        """
        return None


class UtgBasedInputPolicy(InputPolicy):
    """
    state-based input policy
    """

    def __init__(self, device, app, random_input):
        super(UtgBasedInputPolicy, self).__init__(device, app)
        self.random_input = random_input
        self.script = None
        self.master = None
        self.script_events = []
        self.last_event = None
        self.last_state = None
        self.current_state = None
        self.utg = UTG(device=device, app=app, random_input=random_input)
        self.script_event_idx = 0
        if self.device.humanoid is not None:
            self.humanoid_view_trees = []
            self.humanoid_events = []

    def generate_event(self):
        """
        generate an event
        @return:
        """

        # Get current device state
        self.current_state = self.device.get_current_state()
        if self.current_state is None:
            time.sleep(5)
            return KeyEvent(name="BACK")

        # self.__update_utg()
        self.current_state.tag = str(self.action_count) # Named by action_count for easy reference later
        self.current_state.save2dir()

        # update last view trees for humanoid
        if self.device.humanoid is not None:
            self.humanoid_view_trees = self.humanoid_view_trees + [self.current_state.view_tree]
            if len(self.humanoid_view_trees) > 4:
                self.humanoid_view_trees = self.humanoid_view_trees[1:]

        event = None

        # if the previous operation is not finished, continue
        if len(self.script_events) > self.script_event_idx:
            event = self.script_events[self.script_event_idx].get_transformed_event(self)
            self.script_event_idx += 1

        # First try matching a state defined in the script
        if event is None and self.script is not None:
            operation = self.script.get_operation_based_on_state(self.current_state)
            if operation is not None:
                self.script_events = operation.events
                # restart script
                event = self.script_events[0].get_transformed_event(self)
                self.script_event_idx = 1

        if event is None:
            event = self.generate_event_based_on_utg()

        # update last events for humanoid
        if self.device.humanoid is not None:
            self.humanoid_events = self.humanoid_events + [event]
            if len(self.humanoid_events) > 3:
                self.humanoid_events = self.humanoid_events[1:]

        self.last_state = self.current_state
        self.last_event = event
        return event

    def __update_utg(self):
        self.utg.add_transition(self.last_event, self.last_state, self.current_state)

    @abstractmethod
    def generate_event_based_on_utg(self):
        """
        generate an event based on UTG
        :return: InputEvent
        """
        pass


class UtgNaiveSearchPolicy(UtgBasedInputPolicy):
    """
    depth-first strategy to explore UFG (old)
    """

    def __init__(self, device, app, random_input, search_method):
        super(UtgNaiveSearchPolicy, self).__init__(device, app, random_input)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.explored_views = set()
        self.state_transitions = set()
        self.search_method = search_method

        self.last_event_flag = ""
        self.last_event_str = None
        self.last_state = None

        self.preferred_buttons = ["yes", "ok", "activate", "detail", "more", "access",
                                  "allow", "check", "agree", "try", "go", "next"]

    def generate_event_based_on_utg(self):
        """
        generate an event based on current device state
        note: ensure these fields are properly maintained in each transaction:
          last_event_flag, last_touched_view, last_state, exploited_views, state_transitions
        @return: InputEvent
        """
        self.save_state_transition(self.last_event_str, self.last_state, self.current_state)

        if self.device.is_foreground(self.app):
            # the app is in foreground, clear last_event_flag
            self.last_event_flag = EVENT_FLAG_STARTED
        else:
            number_of_starts = self.last_event_flag.count(EVENT_FLAG_START_APP)
            # If we have tried too many times but the app is still not started, stop DroidBot
            if number_of_starts > MAX_NUM_RESTARTS:
                raise InputInterruptedException("The app cannot be started.")

            # if app is not started, try start it
            if self.last_event_flag.endswith(EVENT_FLAG_START_APP):
                # It seems the app stuck at some state, and cannot be started
                # just pass to let viewclient deal with this case
                self.logger.info("The app had been restarted %d times.", number_of_starts)
                self.logger.info("Trying to restart app...")
                pass
            else:
                start_app_intent = self.app.get_start_intent()

                self.last_event_flag += EVENT_FLAG_START_APP
                self.last_event_str = EVENT_FLAG_START_APP
                return IntentEvent(start_app_intent)

        # select a view to click
        view_to_touch = self.select_a_view(self.current_state)

        # if no view can be selected, restart the app
        if view_to_touch is None:
            stop_app_intent = self.app.get_stop_intent()
            self.last_event_flag += EVENT_FLAG_STOP_APP
            self.last_event_str = EVENT_FLAG_STOP_APP
            return IntentEvent(stop_app_intent)

        view_to_touch_str = view_to_touch['view_str']
        if view_to_touch_str.startswith('BACK'):
            result = KeyEvent('BACK')
        else:
            result = TouchEvent(view=view_to_touch)

        self.last_event_flag += EVENT_FLAG_TOUCH
        self.last_event_str = view_to_touch_str
        self.save_explored_view(self.current_state, self.last_event_str)
        return result

    def select_a_view(self, state):
        """
        select a view in the view list of given state, let droidbot touch it
        @param state: DeviceState
        @return:
        """
        views = []
        for view in state.views:
            if view['enabled'] and len(view['children']) == 0:
                views.append(view)

        if self.random_input:
            random.shuffle(views)

        # add a "BACK" view, consider go back first/last according to search policy
        mock_view_back = {'view_str': 'BACK_%s' % state.foreground_activity,
                          'text': 'BACK_%s' % state.foreground_activity}
        if self.search_method == POLICY_NAIVE_DFS:
            views.append(mock_view_back)
        elif self.search_method == POLICY_NAIVE_BFS:
            views.insert(0, mock_view_back)

        # first try to find a preferable view
        for view in views:
            view_text = view['text'] if view['text'] is not None else ''
            view_text = view_text.lower().strip()
            if view_text in self.preferred_buttons \
                    and (state.foreground_activity, view['view_str']) not in self.explored_views:
                self.logger.info("selected an preferred view: %s" % view['view_str'])
                return view

        # try to find a un-clicked view
        for view in views:
            if (state.foreground_activity, view['view_str']) not in self.explored_views:
                self.logger.info("selected an un-clicked view: %s" % view['view_str'])
                return view

        # if all enabled views have been clicked, try jump to another activity by clicking one of state transitions
        if self.random_input:
            random.shuffle(views)
        transition_views = {transition[0] for transition in self.state_transitions}
        for view in views:
            if view['view_str'] in transition_views:
                self.logger.info("selected a transition view: %s" % view['view_str'])
                return view

        # no window transition found, just return a random view
        # view = views[0]
        # self.logger.info("selected a random view: %s" % view['view_str'])
        # return view

        # DroidBot stuck on current state, return None
        self.logger.info("no view could be selected in state: %s" % state.tag)
        return None

    def save_state_transition(self, event_str, old_state, new_state):
        """
        save the state transition
        @param event_str: str, representing the event cause the transition
        @param old_state: DeviceState
        @param new_state: DeviceState
        @return:
        """
        if event_str is None or old_state is None or new_state is None:
            return
        if new_state.is_different_from(old_state):
            self.state_transitions.add((event_str, old_state.tag, new_state.tag))

    def save_explored_view(self, state, view_str):
        """
        save the explored view
        @param state: DeviceState, where the view located
        @param view_str: str, representing a view
        @return:
        """
        if not state:
            return
        state_activity = state.foreground_activity
        self.explored_views.add((state_activity, view_str))

class RandomExplorationPolicy(UtgBasedInputPolicy):
    """
    Random exploration strategy
    """

    def __init__(self, device, app, random_input):
        super(RandomExplorationPolicy, self).__init__(device, app, random_input)
        self.logger = logging.getLogger(self.__class__.__name__)
        

        self.preferred_buttons = ["yes", "ok", "activate", "detail", "more", "access",
                                  "allow", "check", "agree", "try", "go", "next"]

        self.__nav_target = None
        self.__nav_num_steps = -1
        self.__num_restarts = 0
        self.__num_steps_outside = 0
        self.__event_trace = ""
        self.__missed_states = set()
        self.__random_explore = False

    def generate_event_based_on_utg(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        current_state = self.current_state
        self.logger.info("Current state: %s" % current_state.state_str)
        # if current_state.state_str in self.__missed_states:
        #     self.__missed_states.remove(current_state.state_str)
        if current_state.get_app_activity_depth(self.app) < 0:
            # If the app is not in the activity stack
            start_app_intent = self.app.get_start_intent()

            # It seems the app stucks at some state, has been
            # 1) force stopped (START, STOP)
            #    just start the app again by increasing self.__num_restarts
            # 2) started at least once and cannot be started (START)
            #    pass to let viewclient deal with this case
            # 3) nothing
            #    a normal start. clear self.__num_restarts.

            if self.__event_trace.endswith(EVENT_FLAG_START_APP + EVENT_FLAG_STOP_APP) \
                    or self.__event_trace.endswith(EVENT_FLAG_START_APP):
                self.__num_restarts += 1
                self.logger.info("The app had been restarted %d times.", self.__num_restarts)
            else:
                self.__num_restarts = 0

            # Check if we should try to start the app
            if not self.__event_trace.endswith(EVENT_FLAG_START_APP):
                if self.__num_restarts > MAX_NUM_RESTARTS:
                    # If the app had been restarted too many times, enter random mode
                    msg = "The app had been restarted too many times. Entering random mode."
                    self.logger.info(msg)
                    self.__random_explore = True
                else:
                    # Start the app
                    self.__event_trace += EVENT_FLAG_START_APP
                    self.logger.info("Trying to start the app...")
                    return IntentEvent(intent=start_app_intent)

        elif current_state.get_app_activity_depth(self.app) > 0:
            # If the app is in activity stack but is not in foreground
            self.__num_steps_outside += 1

            if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE:
                # If the app has not been in foreground for too long, try to go back
                if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE_KILL:
                    stop_app_intent = self.app.get_stop_intent()
                    go_back_event = IntentEvent(stop_app_intent)
                else:
                    go_back_event = KeyEvent(name="BACK")
                self.__event_trace += EVENT_FLAG_NAVIGATE
                self.logger.info("Going back to the app...")
                return go_back_event
        else:
            # If the app is in foreground
            self.__num_steps_outside = 0


        # Get all possible input events
        possible_events = current_state.get_possible_input_only_leaf_nodes(self.app.get_package_name())
        if len(possible_events) == 0:
            possible_events = current_state.get_possible_input()
        target_event = self._weighted_random_choice(possible_events)
        
        if target_event is None:
            self.logger.info("No possible events available. Trying to go back...")
            self.__event_trace += EVENT_FLAG_NAVIGATE
            return KeyEvent(name="BACK")
        
        if self.device is not None: # skip welcome may not have u2
            target_event.u2 = self.device.u2
        
        # Update event trace based on event type
        if hasattr(target_event, 'event_type'):
            if target_event.event_type in ['touch', 'long_touch', 'swipe', 'scroll', 'set_text', 'select']:
                self.__event_trace += EVENT_FLAG_TOUCH
            elif target_event.event_type == 'key':
                if target_event.name == 'BACK':
                    self.__event_trace += EVENT_FLAG_NAVIGATE
                else:
                    self.__event_trace += EVENT_FLAG_TOUCH
            elif target_event.event_type == 'intent':
                # Update event trace based on intent command type
                intent_cmd = getattr(target_event, 'intent', '')
                if 'start' in intent_cmd:
                    self.__event_trace += EVENT_FLAG_START_APP
                elif 'force-stop' in intent_cmd:
                    self.__event_trace += EVENT_FLAG_STOP_APP
                elif 'broadcast' in intent_cmd:
                    self.__event_trace += EVENT_FLAG_EXPLORE
                else:
                    # Default to explore for other intent types
                    self.__event_trace += EVENT_FLAG_EXPLORE
        
        return target_event

    def _weighted_random_choice(self, possible_events):
        """
        Weighted random selection function
        - touch: highest weight (50%)
        - scroll: lowest weight (5%)
        - other: medium weight (15% each)
        """
        if not possible_events:
            return None
        
        # Define weights for event types
        event_weights = {
            'touch': 50,          # Highest weight
            'long_touch': 15,     # Medium weight
            'swipe': 15,          # Medium weight
            'set_text': 20,       # Medium weight
            'select': 15,         # Medium weight
            'scroll': 20,         # Lowest weight
            'key': 15,            # Medium weight
            'intent': 20,         # Lower weight
        }
        
        # Assign weights to each event
        weighted_events = []
        for event in possible_events:
            event_type = getattr(event, 'event_type', 'unknown')
            weight = event_weights.get(event_type, 10)  # Default weight is 10
            weighted_events.extend([event] * weight)
        
        # If no events match any weights, fall back to original random selection
        if not weighted_events:
            return random.choice(possible_events)
        
        return random.choice(weighted_events)


class UtgGreedySearchPolicy(UtgBasedInputPolicy):
    """
    DFS/BFS (according to search_method) strategy to explore UFG (new)
    """

    def __init__(self, device, app, random_input, search_method):
        super(UtgGreedySearchPolicy, self).__init__(device, app, random_input)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.search_method = search_method

        self.preferred_buttons = ["yes", "ok", "activate", "detail", "more", "access",
                                  "allow", "check", "agree", "try", "go", "next"]

        self.__nav_target = None
        self.__nav_num_steps = -1
        self.__num_restarts = 0
        self.__num_steps_outside = 0
        self.__event_trace = ""
        self.__missed_states = set()
        self.__random_explore = False

    def generate_event_based_on_utg(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        current_state = self.current_state
        self.logger.info("Current state: %s" % current_state.state_str)
        if current_state.state_str in self.__missed_states:
            self.__missed_states.remove(current_state.state_str)

        if current_state.get_app_activity_depth(self.app) < 0:
            # If the app is not in the activity stack
            start_app_intent = self.app.get_start_intent()

            # It seems the app stucks at some state, has been
            # 1) force stopped (START, STOP)
            #    just start the app again by increasing self.__num_restarts
            # 2) started at least once and cannot be started (START)
            #    pass to let viewclient deal with this case
            # 3) nothing
            #    a normal start. clear self.__num_restarts.

            if self.__event_trace.endswith(EVENT_FLAG_START_APP + EVENT_FLAG_STOP_APP) \
                    or self.__event_trace.endswith(EVENT_FLAG_START_APP):
                self.__num_restarts += 1
                self.logger.info("The app had been restarted %d times.", self.__num_restarts)
            else:
                self.__num_restarts = 0

            # pass (START) through
            if not self.__event_trace.endswith(EVENT_FLAG_START_APP):
                if self.__num_restarts > MAX_NUM_RESTARTS:
                    # If the app had been restarted too many times, enter random mode
                    msg = "The app had been restarted too many times. Entering random mode."
                    self.logger.info(msg)
                    self.__random_explore = True
                else:
                    # Start the app
                    self.__event_trace += EVENT_FLAG_START_APP
                    self.logger.info("Trying to start the app...")
                    return IntentEvent(intent=start_app_intent)

        elif current_state.get_app_activity_depth(self.app) > 0:
            # If the app is in activity stack but is not in foreground
            self.__num_steps_outside += 1

            if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE:
                # If the app has not been in foreground for too long, try to go back
                if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE_KILL:
                    stop_app_intent = self.app.get_stop_intent()
                    go_back_event = IntentEvent(stop_app_intent)
                else:
                    go_back_event = KeyEvent(name="BACK")
                self.__event_trace += EVENT_FLAG_NAVIGATE
                self.logger.info("Going back to the app...")
                return go_back_event
        else:
            # If the app is in foreground
            self.__num_steps_outside = 0

        # Get all possible input events
        possible_events = current_state.get_possible_input()

        if self.random_input:
            random.shuffle(possible_events)

        if self.search_method == POLICY_GREEDY_DFS:
            possible_events.append(KeyEvent(name="BACK"))
        elif self.search_method == POLICY_GREEDY_BFS:
            possible_events.insert(0, KeyEvent(name="BACK"))

        # get humanoid result, use the result to sort possible events
        # including back events
        if self.device.humanoid is not None:
            possible_events = self.__sort_inputs_by_humanoid(possible_events)

        # If there is an unexplored event, try the event first
        for input_event in possible_events:
            if not self.utg.is_event_explored(event=input_event, state=current_state):
                self.logger.info("Trying an unexplored event.")
                self.__event_trace += EVENT_FLAG_EXPLORE
                return input_event

        target_state = self.__get_nav_target(current_state)
        if target_state:
            navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=target_state)
            if navigation_steps and len(navigation_steps) > 0:
                self.logger.info("Navigating to %s, %d steps left." % (target_state.state_str, len(navigation_steps)))
                self.__event_trace += EVENT_FLAG_NAVIGATE
                return navigation_steps[0][1]

        if self.__random_explore:
            self.logger.info("Trying random event.")
            random.shuffle(possible_events)
            return possible_events[0]

        # If couldn't find a exploration target, stop the app
        stop_app_intent = self.app.get_stop_intent()
        self.logger.info("Cannot find an exploration target. Trying to restart app...")
        self.__event_trace += EVENT_FLAG_STOP_APP
        return IntentEvent(intent=stop_app_intent)

    def __sort_inputs_by_humanoid(self, possible_events):
        if sys.version.startswith("3"):
            from xmlrpc.client import ServerProxy
        else:
            from xmlrpclib import ServerProxy
        proxy = ServerProxy("http://%s/" % self.device.humanoid)
        request_json = {
            "history_view_trees": self.humanoid_view_trees,
            "history_events": [x.__dict__ for x in self.humanoid_events],
            "possible_events": [x.__dict__ for x in possible_events],
            "screen_res": [self.device.display_info["width"],
                           self.device.display_info["height"]]
        }
        result = json.loads(proxy.predict(json.dumps(request_json)))
        new_idx = result["indices"]
        text = result["text"]
        new_events = []

        # get rid of infinite recursive by randomizing first event
        if not self.utg.is_state_reached(self.current_state):
            new_first = random.randint(0, len(new_idx) - 1)
            new_idx[0], new_idx[new_first] = new_idx[new_first], new_idx[0]

        for idx in new_idx:
            if isinstance(possible_events[idx], SetTextEvent):
                possible_events[idx].text = text
            new_events.append(possible_events[idx])
        return new_events

    def __get_nav_target(self, current_state):
        # If last event is a navigation event
        if self.__nav_target and self.__event_trace.endswith(EVENT_FLAG_NAVIGATE):
            navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=self.__nav_target)
            if navigation_steps and 0 < len(navigation_steps) <= self.__nav_num_steps:
                # If last navigation was successful, use current nav target
                self.__nav_num_steps = len(navigation_steps)
                return self.__nav_target
            else:
                # If last navigation was failed, add nav target to missing states
                self.__missed_states.add(self.__nav_target.state_str)

        reachable_states = self.utg.get_reachable_states(current_state)
        if self.random_input:
            random.shuffle(reachable_states)

        for state in reachable_states:
            # Only consider foreground states
            if state.get_app_activity_depth(self.app) != 0:
                continue
            # Do not consider missed states
            if state.state_str in self.__missed_states:
                continue
            # Do not consider explored states
            if self.utg.is_state_explored(state):
                continue
            self.__nav_target = state
            navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=self.__nav_target)
            if len(navigation_steps) > 0:
                self.__nav_num_steps = len(navigation_steps)
                return state

        self.__nav_target = None
        self.__nav_num_steps = -1
        return None

class UtgReplayPolicy(InputPolicy):
    """
    Replay DroidBot output generated by UTG policy
    """

    def __init__(self, device, app, replay_output):
        super(UtgReplayPolicy, self).__init__(device, app)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.replay_output = replay_output

        event_dir = os.path.join(replay_output, "events")
        files = [os.path.join(event_dir, x) for x in
                 next(os.walk(event_dir))[2]
                 if x.endswith(".json")]
        def _event_index(path):
            base = os.path.basename(path)
            name, _ = os.path.splitext(base)
            try:
                return int(name.split('_')[-1])
            except Exception:
                return float('inf')
        # Natural sort: sort by <num> in event_<num>.json in ascending order
        self.event_paths = sorted(files, key=_event_index)
        # skip HOME and start app intent
        self.device = device
        self.app = app
        self.event_idx = 1
        self.num_replay_tries = 0
        self.utg = UTG(device=device, app=app, random_input=None)
        self.last_event = None
        self.last_state = None
        self.current_state = None

    def generate_event(self):
        """
        generate an event based on replay_output
        @return: InputEvent
        """
        while self.event_idx < len(self.event_paths) and \
              self.num_replay_tries < MAX_REPLY_TRIES:
            self.num_replay_tries += 1
            current_state = self.device.get_current_state()
            if current_state is None:
                time.sleep(5)
                self.num_replay_tries = 0
                return KeyEvent(name="BACK")
            
            curr_event_idx = self.event_idx
            # self.__update_utg()
            self.current_state = current_state
            self.current_state.tag = str(curr_event_idx) # Named by events count for easy reference later
            self.current_state.save2dir()
            if curr_event_idx < len(self.event_paths):
                event_path = self.event_paths[curr_event_idx]
                with open(event_path, "r") as f:
                    curr_event_idx += 1

                    self.logger.info("debug curr_event_idx: " + str(curr_event_idx))

                    if curr_event_idx!= 2:
                        try:
                            event_dict = json.load(f)
                        except Exception as e:
                            self.logger.info("Loading %s failed" % event_path + "curr_event_idx: " + str(curr_event_idx))
                            continue

                    # if event_dict["start_state"] != current_state.state_str:
                    #     continue
                    # if not self.device.is_foreground(self.app):
                    #     # if current app is in background, bring it to foreground
                    #     # component = self.app.get_package_name()
                    #     # if self.app.get_main_activity():
                    #     #     component += "/%s" % self.app.get_main_activity()
                    #     return IntentEvent(self.app.get_start_intent())
                    
                    self.logger.info("Replaying %s" % event_path + "curr_event_idx: " + str(curr_event_idx))
                    self.event_idx = curr_event_idx
                    self.num_replay_tries = 0
                    
                    # Skip the 2nd event, directly return the app launch Intent
                    if curr_event_idx == 2: # Some second events are empty
                        return IntentEvent(self.app.get_start_intent())
                    
                    event = InputEvent.from_dict(event_dict["event"])
                    event.u2 = self.device.u2
                    if isinstance(event, IntentEvent):
                        return event
                    elif isinstance(event, KeyEvent):
                        return event


                    check_result = self.check_which_exists(event)
                    print("debug check_result", check_result)
                    if check_result[0] is None:
                        self.logger.warning(f"Widget not found for event: {event_path}")
                        self.logger.info("Stopping replay due to widget not found")
                        self.current_state.tag = str(curr_event_idx) # Named by events count for easy reference later
                        self.current_state.save2dir() # save the current state
                        self.input_manager.enabled = False
                        self.input_manager.stop()
                        break
                
                    
                    self.last_state = self.current_state
                    self.last_event = event
                    
                    
                    return event

            time.sleep(5)

        # raise InputInterruptedException("No more record can be replayed.")
    
    def check_if_same(self, current, record):
        if current is None or record is None:
            return False
        if current == record:
            return True
        return False

    def replace_view(self, event, current_view):
        event.view['resource_id'] = current_view['resource_id']
        event.view['text'] = current_view['text']
        event.view['content_description'] = current_view['content_description']
        event.view['class'] = current_view['class']
        event.view['instance'] = current_view['instance']
        event.view['bounds'] = current_view['bounds']
    
    def check_which_exists(self, event):
        resource_id = UtgReplayPolicy.__safe_dict_get(event.view, 'resource_id')
        text = UtgReplayPolicy.__safe_dict_get(event.view, 'text')
        content_description = UtgReplayPolicy.__safe_dict_get(event.view, 'content_description')
        class_name = UtgReplayPolicy.__safe_dict_get(event.view, 'class')
        instance = UtgReplayPolicy.__safe_dict_get(event.view, 'instance')

        u2 = self.device.u2
        

        if content_description is not None:
            if u2.exists(description=content_description, instance=instance):
                for current_view in self.current_state.views:
                    if self.check_if_same(current_view['content_description'], content_description) and self.check_if_same(current_view['instance'], instance):
                        self.replace_view(event, current_view)
                        break
                return 'content_description', content_description
        elif text is not None:
            if u2.exists(text=text, instance=instance):
                for current_view in self.current_state.views:
                    if self.check_if_same(current_view['text'], text) and self.check_if_same(current_view['instance'], instance):
                        self.replace_view(event, current_view)
                        break
                return 'text', text
        elif resource_id is not None:
            if u2.exists(resourceId=resource_id, instance=instance):
                for current_view in self.current_state.views:
                    if self.check_if_same(current_view['resource_id'], resource_id) and self.check_if_same(current_view['instance'], instance):
                        self.replace_view(event, current_view)
                        break
                return 'resource_id', resource_id
        elif class_name is not None:
            if u2.exists(className=class_name, instance=instance):
                for current_view in self.current_state.views:
                    if self.check_if_same(current_view['class'], class_name) and self.check_if_same(current_view['instance'], instance):
                        self.replace_view(event, current_view)
                        break
                return 'class_name', class_name
        elif class_name is not None and resource_id is not None and instance is not None:
            if u2.exists(className=class_name, resourceId=resource_id, instance=instance):
                for current_view in self.current_state.views:
                    if self.check_if_same(current_view['class'], class_name) and self.check_if_same(current_view['resource_id'], resource_id) and self.check_if_same(current_view['instance'], instance):
                        self.replace_view(event, current_view)
                        break
                return 'class_resource_instance', (class_name, resource_id, instance)
        
        return None, None
    

    @staticmethod
    def __safe_dict_get(view_dict, key, default=None):
        value = view_dict[key] if key in view_dict else None
        return value if value is not None else default

class GroundTruthPolicy(InputPolicy):
    """
    Replay DroidBot output generated by Ground Truth policy

    existing matched file: matched_element_<event_number>.json
    generating the following events
    """

    def __init__(self, device, app, replay_output, ground_truth_path):
        super(GroundTruthPolicy, self).__init__(device, app)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.replay_output = replay_output

        
        event_dir = os.path.join(replay_output, "events")
        files = [os.path.join(event_dir, x) for x in
                 next(os.walk(event_dir))[2]
                 if x.endswith(".json")]
        def _event_index(path):
            base = os.path.basename(path)
            name, _ = os.path.splitext(base)
            try:
                return int(name.split('_')[-1])
            except Exception:
                return float('inf')
        # Natural sort: sort by <num> in event_<num>.json in ascending order
        self.event_paths = sorted(files, key=_event_index)
        # skip HOME and start app intent
        self.device = device
        self.app = app
        self.event_idx = 1
        self.num_replay_tries = 0
        self.utg = UTG(device=device, app=app, random_input=None)
        self.last_event = None
        self.last_state = None
        self.current_state = None

        self.failed_event_number = 0
        self.matched_element = None
        self.load_ground_truth(ground_truth_path)


    def load_ground_truth(self, ground_truth_path):
        matched_files = glob.glob(os.path.join(ground_truth_path, "matched_element*.json"))
        if matched_files and len(matched_files) == 1:
            matched_file = matched_files[0]
            # Extract number from filename (e.g., extract 91 from matched_element_91.json)
            import re
            match = re.search(r'matched_element_(\d+)\.json', os.path.basename(matched_file))
            if match:
                self.failed_event_number = int(match.group(1))
            else:
                self.failed_event_number = 0
                
            with open(matched_file, "r") as f:
                self.matched_element = json.load(f)

            print("debug failed_event_number: ", self.failed_event_number)
            print("debug matched_element: ", self.matched_element)

    def generate_event(self):
        """
        generate an event based on replay_output
        @return: InputEvent
        """
        while self.event_idx < len(self.event_paths) and \
              self.num_replay_tries < MAX_REPLY_TRIES:
            self.num_replay_tries += 1
            current_state = self.device.get_current_state()
            if current_state is None:
                time.sleep(5)
                self.num_replay_tries = 0
                return KeyEvent(name="BACK")
            
            curr_event_idx = self.event_idx
            # self.__update_utg()
            self.current_state = current_state
            self.current_state.tag = str(curr_event_idx) # Named by events count for easy reference later
            self.current_state.save2dir()
            if curr_event_idx < len(self.event_paths):
                event_path = self.event_paths[curr_event_idx]
                with open(event_path, "r") as f:
                    curr_event_idx += 1

                    self.logger.info("debug curr_event_idx: " + str(curr_event_idx))

                    if curr_event_idx!= 2:
                        try:
                            event_dict = json.load(f)
                        except Exception as e:
                            self.logger.info("Loading %s failed" % event_path + "curr_event_idx: " + str(curr_event_idx))
                            continue

                    # if event_dict["start_state"] != current_state.state_str:
                    #     continue
                    # if not self.device.is_foreground(self.app):
                    #     # if current app is in background, bring it to foreground
                    #     # component = self.app.get_package_name()
                    #     # if self.app.get_main_activity():
                    #     #     component += "/%s" % self.app.get_main_activity()
                    #     return IntentEvent(self.app.get_start_intent())
                    
                    self.logger.info("Replaying %s" % event_path + "curr_event_idx: " + str(curr_event_idx))
                    self.event_idx = curr_event_idx
                    self.num_replay_tries = 0
                    
                    # Skip the 2nd event, directly return the app launch Intent
                    if curr_event_idx == 2: # Some second events are empty
                        return IntentEvent(self.app.get_start_intent())
                    
                    event = InputEvent.from_dict(event_dict["event"])
                    event.u2 = self.device.u2
                    if isinstance(event, IntentEvent):
                        return event
                    elif isinstance(event, KeyEvent):
                        return event


                    check_result = self.check_which_exists(event)
                    print("debug check_result", check_result)
                    if check_result[0] is None:
                        self.logger.warning(f"Widget not found for event: {event_path}")
                        self.logger.info("Stopping replay due to widget not found")
                        self.current_state.tag = str(curr_event_idx) # Named by events count for easy reference later
                        self.current_state.save2dir() # save the current state
                        self.input_manager.enabled = False
                        self.input_manager.stop()
                        break
                
                    
                    self.last_state = self.current_state
                    self.last_event = event
                    
                    
                    return event

            time.sleep(5)

        # raise InputInterruptedException("No more record can be replayed.")
    
    def check_if_same(self, current, record):
        if current is None or record is None:
            return False
        if current == record:
            return True
        return False

    def replace_view(self, event, current_view):
        event.view['resource_id'] = current_view['resource_id']
        event.view['text'] = current_view['text']
        event.view['content_description'] = current_view['content_description']
        event.view['class'] = current_view['class']
        event.view['instance'] = current_view['instance']
        event.view['bounds'] = current_view['bounds']
    
    
    def normalize(self, value):
        if value is None:
            return ""
        else:
            return value

    def compare_bounds(self, current_bounds, gt_bounds):
        str_cur_bounds = '['+str(current_bounds[0][0])+","+str(current_bounds[0][1])+"]["+str(current_bounds[1][0])+","+str(current_bounds[1][1])+']'
        return str_cur_bounds == str(gt_bounds)


    def check_which_exists(self, event):
        if self.failed_event_number == self.event_idx - 1:
            # using the matched element to replace the event
            for current_view in self.current_state.views:
                # normalize
                resource_id = self.normalize(current_view['resource_id'])
                text = self.normalize(current_view['text'])
                content_description = self.normalize(current_view['content_description'])
                class_name = self.normalize(current_view['class'])
                bounds = current_view['bounds']

                if self.check_if_same(resource_id, self.matched_element['resource-id']) and \
                   self.check_if_same(text, self.matched_element['text']) and \
                   self.check_if_same(content_description, self.matched_element['content-desc']) and \
                   self.check_if_same(class_name, self.matched_element['class']) and \
                   self.compare_bounds(bounds, self.matched_element['bounds']):
                    self.replace_view(event, current_view)
                    u2 = self.device.u2
                    return 'matched_element', self.matched_element
            
            return 'matched_element', None
        
        else:
        
            resource_id = GroundTruthPolicy.__safe_dict_get(event.view, 'resource_id')
            text = GroundTruthPolicy.__safe_dict_get(event.view, 'text')
            content_description = GroundTruthPolicy.__safe_dict_get(event.view, 'content_description')
            class_name = GroundTruthPolicy.__safe_dict_get(event.view, 'class')
            instance = GroundTruthPolicy.__safe_dict_get(event.view, 'instance')



            u2 = self.device.u2
            

            if content_description is not None:
                if u2.exists(description=content_description, instance=instance):
                    for current_view in self.current_state.views:
                        if self.check_if_same(current_view['content_description'], content_description) and self.check_if_same(current_view['instance'], instance):
                            self.replace_view(event, current_view)
                            break
                    return 'content_description', content_description
            elif text is not None:
                if u2.exists(text=text, instance=instance):
                    for current_view in self.current_state.views:
                        if self.check_if_same(current_view['text'], text) and self.check_if_same(current_view['instance'], instance):
                            self.replace_view(event, current_view)
                            break
                    return 'text', text
            elif resource_id is not None:
                if u2.exists(resourceId=resource_id, instance=instance):
                    for current_view in self.current_state.views:
                        if self.check_if_same(current_view['resource_id'], resource_id) and self.check_if_same(current_view['instance'], instance):
                            self.replace_view(event, current_view)
                            break
                    return 'resource_id', resource_id
            elif class_name is not None:
                if u2.exists(className=class_name, instance=instance):
                    for current_view in self.current_state.views:
                        if self.check_if_same(current_view['class'], class_name) and self.check_if_same(current_view['instance'], instance):
                            self.replace_view(event, current_view)
                            break
                    return 'class_name', class_name
            elif class_name is not None and resource_id is not None and instance is not None:
                if u2.exists(className=class_name, resourceId=resource_id, instance=instance):
                    for current_view in self.current_state.views:
                        if self.check_if_same(current_view['class'], class_name) and self.check_if_same(current_view['resource_id'], resource_id) and self.check_if_same(current_view['instance'], instance):
                            self.replace_view(event, current_view)
                            break
                    return 'class_resource_instance', (class_name, resource_id, instance)
            
            return None, None
    

    @staticmethod
    def __safe_dict_get(view_dict, key, default=None):
        value = view_dict[key] if key in view_dict else None
        return value if value is not None else default


class MatchingPolicy(InputPolicy):
    """
    Replay DroidBot output generated by Matching policy

    find the target element
    """

    def __init__(self, device, app, replay_output, failed_replay_output, output_dir,
                 without_taxonomy=False, without_rule=False, without_llm=False,
                 without_next_screen_summary=False, without_history_summary=False):
        super(MatchingPolicy, self).__init__(device, app)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.replay_output = replay_output # Normal record output
        self.failed_replay_output = failed_replay_output # Failed replay output
        self.output_dir = output_dir # Output directory

        event_dir = os.path.join(replay_output, "events")
        files = [os.path.join(event_dir, x) for x in
                 next(os.walk(event_dir))[2]
                 if x.endswith(".json")]
        def _event_index(path):
            base = os.path.basename(path)
            name, _ = os.path.splitext(base)
            try:
                return int(name.split('_')[-1])
            except Exception:
                return float('inf')
        # Natural sort: sort by <num> in event_<num>.json in ascending order
        self.event_paths = sorted(files, key=_event_index)
        # skip HOME and start app intent
        self.device = device
        self.app = app
        self.event_idx = 1
        self.num_replay_tries = 0
        self.utg = UTG(device=device, app=app, random_input=None)
        self.last_event = None
        self.last_state = None
        self.current_state = None

        # Failed event related
        self.failed_event_number = 0
        self.failed_event_path = None
        self.failed_event_json = None
        self.failed_event_xml_tree = None
        self.failed_event_png_path = None
        self.failed_event_png = None
        self.failed_event_png_next = None
        self.failed_event_png_next_path = None
        self.load_failed_event() # Load failed_event_number, failed_event_json, failed_event_xml_tree, failed_event_png

        # Mode management, first replay mode, then repair mode
        self.mode = "replay"  # Two modes: "replay" (replay mode) and "repair" (repair mode)

        # Repair process tracking
        self.repair_trace = []  # Complete trace of repair process, each step includes: {step_number, matched_element, screenshot}
        self.try_count = 0  # Number of attempts to find target element
        self.exploration_step = 0  # Exploration step counter, globally managed
        # Exploration state management (for random exploration fallback)
        self.visited_states = set()  # Visited state hashes, to avoid repeated exploration

        # Feedback mechanism: record incorrect match results, exclude them during retry
        self.excluded_views = []  # Excluded views (previously matched successfully but failed later)
        self.last_repaired_view = None  # View matched in the last repair attempt
        self.exploration_retry_count = 0  # Exploration retry count
        self.max_exploration_retries = 3  # Maximum retry count

        # Activity path tracking (for determining back feasibility)
        self.activity_trace = []  # Record activity changes after each event execution
        self.exploration_activity_trace = []  # Record activity changes during exploration

        self.original_next_screen_summary = None # Summary of the original next screen

        # Temporary file storage directory
        self.exploration_tmp_dir = os.path.join(self.output_dir, "exploration_tmp/")
        os.makedirs(self.exploration_tmp_dir, exist_ok=True)

        # Prevent DFS infinite loops by recording visited navigation elements
        self.visited_navigation_elements = set()  # Record previously selected navigation elements (activity, resource_id, class, content_desc, click_type)

        # Configure logger output to file
        self._setup_file_logger()

        # LLM configuration
        self.llm_api_key = os.getenv("API_KEY")  # OpenAI API Key

        # Ablation experiments
        self.without_taxonomy = without_taxonomy
        self.without_rule = without_rule
        self.without_llm = without_llm
        self.without_next_screen_summary = without_next_screen_summary
        self.without_history_summary = without_history_summary
        self.logger.info(f"Ablation experiments - without_taxonomy: {self.without_taxonomy}, without_rule: {self.without_rule}, without_llm: {self.without_llm}, without_next_screen_summary: {self.without_next_screen_summary}, without_history_summary: {self.without_history_summary}")

    def _setup_file_logger(self):
        """Configure logger output to a file in the exploration_tmp directory"""
        log_file = os.path.join(self.exploration_tmp_dir, "repair.log")

        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Set format
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        # Add to logger
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)

        self.logger.info(f"Logger initialized, log file: {log_file}")

    def load_failed_event(self):

        # 1. Load failed_event_number
        all_event_files = glob.glob(os.path.join(self.failed_replay_output, "events/*.json"))
        event_numbers = [int(os.path.basename(file).split('_')[-1].split('.')[0]) for file in all_event_files]
        self.failed_event_number = max(event_numbers) + 1

        # 2. Load failed_event_json
        failed_event_json_path = os.path.join(self.replay_output, f"events/event_{self.failed_event_number}.json")
        self.failed_event_path = failed_event_json_path
        with open(failed_event_json_path, 'r') as f:
            self.failed_event_json = json.load(f)

        # 3. Load failed_event_xml_tree
        failed_event_xml_tree_path = os.path.join(self.replay_output, f"xmls/xml_{self.failed_event_number-1}.xml")
        with open(failed_event_xml_tree_path, 'r') as f:
            self.failed_event_xml_tree = ET.parse(f)

        # 4. Load failed_event_png
        failed_event_png_path = os.path.join(self.replay_output, f"states/screen_{self.failed_event_number-1}.png")
        self.failed_event_png_path = failed_event_png_path
        self.failed_event_png = read_image(failed_event_png_path) # PIL.Image.Image

        # 5. Load failed_event_png_next
        failed_event_png_next_path = os.path.join(self.replay_output, f"states/screen_{self.failed_event_number}.png")
        self.failed_event_png_next_path = failed_event_png_next_path
        self.failed_event_png_next = read_image(failed_event_png_next_path) # PIL.Image.Image
    
   
    def normalize(self, value):
        if value is None:
            return ""
        else:
            return value

    def compare_bounds(self, current_bounds, gt_bounds):
        str_cur_bounds = '['+str(current_bounds[0][0])+","+str(current_bounds[0][1])+"]["+str(current_bounds[1][0])+","+str(current_bounds[1][1])+']'
        return str_cur_bounds == str(gt_bounds)


    def generate_event(self):
        """
        generate an event based on replay_output
        @return: InputEvent
        """
        while self.event_idx < len(self.event_paths) and \
              self.num_replay_tries < MAX_REPLY_TRIES:
            self.num_replay_tries += 1
            current_state = self.device.get_current_state()
            if current_state is None:
                time.sleep(5)
                self.num_replay_tries = 0
                return KeyEvent(name="BACK")

            curr_event_idx = self.event_idx
            # self.__update_utg()
            self.current_state = current_state
            self.current_state.tag = str(curr_event_idx) # Named by events count for easy reference later
            self.current_state.save2dir()

            # Update the activity_after of the previous event (replay phase)
            if len(self.activity_trace) > 0 and self.activity_trace[-1]['activity_after'] is None:
                self.activity_trace[-1]['activity_after'] = current_state.foreground_activity

            if curr_event_idx < len(self.event_paths):
                event_path = self.event_paths[curr_event_idx]
                with open(event_path, "r") as f:
                    curr_event_idx += 1

                    time.sleep(1) # Wait 1 second for app to load

                    self.logger.info("debug curr_event_idx: " + str(curr_event_idx))

                    if curr_event_idx!= 2:
                        try:
                            event_dict = json.load(f)
                        except Exception as e:
                            self.logger.info("Loading %s failed" % event_path + "curr_event_idx: " + str(curr_event_idx))
                            continue

                    self.logger.info("Replaying %s" % event_path + "curr_event_idx: " + str(curr_event_idx))
                    self.event_idx = curr_event_idx
                    self.num_replay_tries = 0
                    
                    # Skip the 2nd event, directly return the app launch Intent
                    if curr_event_idx == 2: # Some second events are empty
                        return IntentEvent(self.app.get_start_intent())

                    if self.app.get_package_name() == "com.appmindlab.nano" and curr_event_idx == 3:
                        # This app has an issue, restart it once more after restarting
                        self.device.adb.shell("am force-stop %s" % self.app.get_package_name())
                        self.device.start_app(self.app)
                        time.sleep(2)

                    
                    
                    event = InputEvent.from_dict(event_dict["event"])
                    event.u2 = self.device.u2
                    if isinstance(event, IntentEvent):
                        return event
                    elif isinstance(event, KeyEvent):
                        return event

                    # If reaching the failed event, switch to exploration mode
                    if curr_event_idx == self.failed_event_number:
                        self.logger.info("Reached failed event, switching to exploration mode")
                        self.mode = "explore"
                        self.target_event = event
                        time.sleep(1)
                        return self.start_exploration()
                    check_result = self.check_which_exists(event)
                    print("debug check_result", check_result)
                    if check_result[0] is None:
                        self.logger.warning(f"Widget not found for event: {event_path}")

                        # Check if the repaired event failed (indicating previous match was wrong, need to retry)
                        if curr_event_idx == self.failed_event_number + 1:
                            self.logger.info("Repaired event failed! Adding to excluded list and retrying...")
                            # Add incorrect view to exclusion list
                            if self.last_repaired_view:
                                self.excluded_views.append(self.last_repaired_view)
                            self.last_repaired_view = None
                            self.exploration_retry_count += 1
                            self.logger.info(f"Retry {self.exploration_retry_count}/{self.max_exploration_retries}, excluded {len(self.excluded_views)} views")

                            # Check if there are retry chances left
                            # if self.exploration_retry_count < self.max_exploration_retries:
                            #     # Restart app and replay to before failed_event
                            #     self.logger.info("Restarting app and replaying to retry exploration...")

                            #     # Reset event_idx to failed_event
                            #     self.event_idx = self.failed_event_number

                            #     # Clear activity trace and start over
                            #     self.activity_trace = []
                            #     self.exploration_activity_trace = []

                            #     # Restart exploration (will first replay to failed_event)
                            #     return self._restart_and_replay_to_failed_event()
                            # else:
                            #     self.logger.warning(f"Max retries ({self.max_exploration_retries}) reached, giving up")

                        self.logger.info("Stopping replay due to widget not found")
                        self.current_state.tag = str(curr_event_idx) # Named by events count for easy reference later
                        self.current_state.save2dir() # save the current state
                        self.input_manager.enabled = False
                        self.input_manager.stop()
                        break

                    if curr_event_idx == self.failed_event_number + 2:
                        # Fixed successfully, stop directly, no need to continue replay
                        self.logger.info("Repaired event succeeded! Stopping replay...")
                        self.input_manager.enabled = False
                        self.input_manager.stop()
                        break



                    self.last_state = self.current_state
                    self.last_event = event

                    # Record activity changes (for later back decision)
                    activity_before = self.current_state.foreground_activity
                    self.activity_trace.append({
                        'event_idx': curr_event_idx,
                        'activity_before': activity_before,
                        'activity_after': None  # Updated after execution
                    })

                    return event

            time.sleep(5)

    def start_exploration(self, max_steps=15):
        """
        Exploration mode: First look for target element at each step, if not found, let LLM recommend navigation

        Args:
            max_steps: Maximum exploration steps

        Process:
        Each step:
        1. Get clickable elements on current page
        2. Search for target element
        3. Found → Verify (cross page needs LLM judge) → Return success
        4. Not found → LLM recommends navigation element → Click → Enter next step
        """

        # Record activity at start of exploration, for later return navigation and retry
        start_activity = self.current_state.foreground_activity
        self.start_activity_before_repair = start_activity  # Save as instance variable for retry
        self.logger.info(f"Exploration starting from activity: {start_activity}")

        # Clear activity trace for exploration phase
        self.exploration_activity_trace = []

        # Reset exploration step counter
        self.exploration_step = 0

        while self.exploration_step < max_steps:
            cross_page = (self.exploration_step > 0)  # step 0 is same page, step 1+ is cross page
            self.logger.info(f"=== Exploration step {self.exploration_step}/{max_steps} ({'cross page' if cross_page else 'same page'}) ===")

            # 1. Get all clickable elements on current page
            possible_events = self.current_state.get_possible_input(package_name=self.app.get_package_name())
            time.sleep(1)
            if len(possible_events) == 0:
                self.logger.warning("No clickable elements found, pressing BACK to return...")
                self.device.send_event(KeyEvent(name="BACK"))
                time.sleep(0.5)
                self.current_state = self.device.get_current_state()
                self.exploration_step += 1
                continue

            # Add scrollable events
            scrollable_events = []
            for event in possible_events:
                if event.event_type == "scroll":
                    scrollable_events.append(event)

            # Filter according to UIMatch rules
            self.logger.info(f"Filtering events: original count = {len(possible_events)}")
            possible_events = self._filter_events_by_rules(possible_events)
            if self.app.get_package_name() == "com.amaze.filemanager" or self.app.get_package_name() == "com.appmindlab.nano":
                possible_events = self.current_state.get_possible_input_only_leaf_nodes(self.app.get_package_name())
            self.logger.info(f"After filtering: {len(possible_events)} events")


            # 2. Find target element on current page
            self.logger.info(f"Step {self.exploration_step}: Searching for target element...")
            matched_view, matching_method = self.find_target_element_in_page(self.current_state, self.exploration_step, cross_page, is_has_next_screen_summary=not self.without_next_screen_summary)
            self.logger.info(f"Element match result: {matched_view is not None}, method: {matching_method}")

            # 3. Same page handling (includes scroll voting mechanism)
            if not cross_page:
                current_activity = self.current_state.foreground_activity

                # Filter out usable scroll events
                scroll_events = self.filter_scroll_events(scrollable_events)

                # If there are scroll events, use voting mechanism (traverse all scrollable views)
                if len(scroll_events) > 0 and current_activity == start_activity:
                    self.logger.info(f"Same page with {len(scroll_events)} scrollable views, using voting mechanism...")

                    # Record executed scrolls for later reverse recovery
                    executed_scrolls = []  # [(scroll_event, direction), ...]

                    # Collect results from multiple searches for voting
                    candidates = {}  # {view_str: {'view': view, 'count': count, 'last_state': state}}

                    # First search (current page before scroll)
                    if matched_view:
                        view_str = matched_view.get('view_str', '')
                        candidates[view_str] = {
                            'view': matched_view,
                            'count': 1,
                            'last_state': self.current_state,
                            'matching_method': matching_method,
                            'scroll_index': 0  # Found before scroll, index is 0
                        }
                        self.logger.info(f"Initial match found (before scroll): {view_str[:20]}...")

                    # Traverse all scrollable views to scroll
                    for i, scroll_event in enumerate(scroll_events):
                        state_before = self.current_state.state_str
                        # Record screenshot path before scroll (for similarity comparison)
                        screenshot_before = self.current_state.screenshot_path

                        self.logger.info(f"Scroll {i + 1}/{len(scroll_events)} for voting...")
                        self.device.send_event(scroll_event, probe=True) # Smaller scroll distance
                        executed_scrolls.append(scroll_event)  # Record executed scroll
                        time.sleep(0.5)
                        self.exploration_step += 1

                        new_state = self.device.get_current_state()
                        self.current_state = new_state

                        # Check screenshot similarity before and after scroll, if > 0.95 means scroll is ineffective, skip
                        screenshot_after = new_state.screenshot_path if new_state else None
                        if screenshot_before and screenshot_after and os.path.exists(screenshot_before) and os.path.exists(screenshot_after):
                            try:
                                img_before = read_image(screenshot_before)
                                img_after = read_image(screenshot_after)
                                similarity = compute_ssim(img_before, img_after)
                                if similarity > 0.95:
                                    self.logger.info(f"Scroll had no effect (similarity={similarity:.4f} > 0.95), skipping...")
                                    continue
                            except Exception as e:
                                self.logger.warning(f"Failed to compare screenshots: {e}")

                        # Find target in new page (this will save screen_same_page_{step}.png and xml_same_page_{step}.xml)
                        new_matched, new_matching_method = self.find_target_element_in_page(self.current_state, self.exploration_step, cross_page, is_has_next_screen_summary=not self.without_next_screen_summary)

                        # Record scroll to trace (using screenshot path saved by find_target_element_in_page)

                        scroll_trace = {
                            'step': self.exploration_step,
                            'action': 'scroll',
                            'direction': scroll_event.direction,
                            'event': {
                                'type': 'scroll',
                                'view': scroll_event.view if hasattr(scroll_event, 'view') else None
                            },
                            'state_before': state_before,
                            'state_after': new_state.state_str if new_state else None,
                            'screenshot': os.path.join(self.exploration_tmp_dir, f"states/screen_same_page_{self.exploration_step}.png"),
                            'xml': os.path.join(self.exploration_tmp_dir, f"xmls/xml_same_page_{self.exploration_step}.xml"),
                            'found_target': new_matched is not None,
                            'matching_method': new_matching_method
                        }
                        self.repair_trace.append(scroll_trace)

                        if new_matched:
                            view_str = new_matched.get('view_str', '')
                            current_scroll_index = i + 1  # After the i-th scroll, index is i+1
                            if view_str in candidates:
                                candidates[view_str]['count'] += 1
                                candidates[view_str]['view'] = new_matched  # Update to the latest view
                                candidates[view_str]['last_state'] = self.current_state
                                candidates[view_str]['matching_method'] = new_matching_method
                                # Keep the earliest scroll_index, do not update
                                self.logger.info(f"View {view_str[:20]}... found again, count={candidates[view_str]['count']}, first_scroll_index={candidates[view_str]['scroll_index']}")
                            else:
                                candidates[view_str] = {
                                    'view': new_matched,
                                    'count': 1,
                                    'last_state': self.current_state,
                                    'matching_method': new_matching_method,
                                    'scroll_index': current_scroll_index  # Record the scroll index when first appeared
                                }
                                self.logger.info(f"New candidate found: {view_str[:20]}..., scroll_index={current_scroll_index}")

                    # Voting: select the best candidate
                    # If only one non-LLM match, directly select as best; otherwise vote by count
                    should_return = False
                    repaired_event = None

                    if candidates:
                        # Output all scroll candidates to json file
                        scroll_candidates_output = []
                        for view_str, candidate_info in candidates.items():
                            scroll_candidates_output.append({
                                'view_str': view_str,
                                'count': candidate_info['count'],
                                'scroll_index': candidate_info.get('scroll_index', -1),
                                'matching_method': candidate_info['matching_method'],
                                'view': candidate_info['view']
                            })
                        scroll_candidates_path = os.path.join(self.exploration_tmp_dir, "repair_logs", f"scroll_candidates_event_{self.failed_event_number}.json")
                        # Ensure directory exists
                        repair_logs_dir = os.path.join(self.exploration_tmp_dir, "repair_logs")
                        if not os.path.exists(repair_logs_dir):
                            os.makedirs(repair_logs_dir)
                        with open(scroll_candidates_path, 'w') as f:
                            json.dump({
                                'failed_event_number': self.failed_event_number,
                                'total_candidates': len(candidates),
                                'candidates': scroll_candidates_output
                            }, f, indent=2, ensure_ascii=False)
                        self.logger.info(f"Saved scroll candidates to: {scroll_candidates_path}")

                        # Find non-LLM matched candidates (high accuracy, but still need LLM judge verification)
                        non_llm_candidates = {k: v for k, v in candidates.items()
                                              if v['matching_method'] and v['matching_method'] != 'llm'}

                        selection_method = None  # Used to record selection method
                        best_candidate = None
                        matched_view = None

                        if len(non_llm_candidates) == 1:
                            # Only one non-LLM match, select as best candidate
                            best_view_str = list(non_llm_candidates.keys())[0]
                            best_candidate = non_llm_candidates[best_view_str]
                            matched_view = best_candidate['view']
                            self.current_state = best_candidate['last_state']
                            selection_method = 'direct_non_llm'
                            self.logger.info(f"Direct match: exactly one non-LLM candidate (method={best_candidate['matching_method']})")
                        elif len(candidates) == 1:
                            # Only one candidate, select as best candidate
                            best_view_str = list(candidates.keys())[0]
                            best_candidate = candidates[best_view_str]
                            matched_view = best_candidate['view']
                            self.current_state = best_candidate['last_state']
                            selection_method = 'single_candidate'
                            self.logger.info(f"Single candidate: only one candidate available (method={best_candidate['matching_method']})")
                        else:
                            # Multiple candidates -> use LLM to select
                            self.logger.info(f"Using LLM to select best candidate from {len(candidates)} candidates (non-LLM count={len(non_llm_candidates)})...")
                            result = self.llm_select_best_candidate(candidates)

                            if result:
                                best_view_str, best_candidate = result
                                matched_view = best_candidate['view']
                                self.current_state = best_candidate['last_state']
                                selection_method = 'llm_select'
                                self.logger.info(f"LLM selected candidate: {best_view_str[:30]}..., method={best_candidate['matching_method']}")
                            else:
                                self.logger.info("LLM selected NONE - no matching candidate")

                        # After selecting candidate, call llm_judge_exploration_success for final verification
                        if best_candidate:
                            self.logger.info(f"Calling llm_judge_exploration_success to verify the selected candidate...")
                            # Do not pass matched view here; only judge whether on the correct page
                            judge_result = self.llm_judge_exploration_success(matched_view=matched_view, is_has_next_screen_summary=not self.without_next_screen_summary, is_has_taxonomy=not self.without_taxonomy)
                            self.logger.info(f"LLM judge result: {judge_result}")
                        else:
                            judge_result = False

                        if judge_result:
                            self.logger.info("✓ Found target element (same page with scroll voting)")
                            self.exploration_step += 1  # Increment step for match_found record

                            self.repair_trace.append({
                                'step': self.exploration_step,
                                'action': 'match_found',
                                'event': None,
                                'state_after': self.current_state.state_str,
                                'screenshot': os.path.join(self.exploration_tmp_dir, f"states/screen_same_page_{self.exploration_step}.png"),
                                'xml': os.path.join(self.exploration_tmp_dir, f"xmls/xml_same_page_{self.exploration_step}.xml"),
                                'found_target': True,
                                'matched_view': matched_view,
                                'judge_result': judge_result,
                                'selection_method': selection_method,
                                'voting_count': best_candidate['count'],
                                'total_candidates': len(candidates),
                                'matching_method': best_candidate['matching_method']
                            })

                            repaired_event = self._create_repaired_event(matched_view)
                            self.last_repaired_view = matched_view

                            repaired_event_info = {
                                'tag': f"repaired_event_step_{self.exploration_step}",
                                'event': {
                                    'event_type': repaired_event.event_type if hasattr(repaired_event, 'event_type') else 'touch',
                                    'log_lines': None,
                                    'x': None,
                                    'y': None,
                                    'view': matched_view
                                },
                                'start_state': self.current_state.state_str if self.current_state else None,
                                'stop_state': None,
                                'event_str': repaired_event.get_event_str(self.current_state) if hasattr(repaired_event, 'get_event_str') else str(repaired_event)
                            }
                            event_dir = os.path.join(self.exploration_tmp_dir, "events")
                            if not os.path.exists(event_dir):
                                os.makedirs(event_dir)
                            repaired_event_path = os.path.join(event_dir, f"event_repaired_step_{self.exploration_step}.json")
                            with open(repaired_event_path, 'w') as f:
                                json.dump(repaired_event_info, f, indent=2, ensure_ascii=False)
                            self.logger.info(f"Saved repaired event to: {repaired_event_path}")
                            should_return = True
                        else:
                            self.logger.info("Voting best candidate failed LLM judge, continuing to navigation...")
                            matched_view = None  # Clear and continue to navigation logic
                    else:
                        self.logger.info("No candidates found after scroll voting")
                        matched_view = None

                    # Target found, need to determine whether to reverse scroll back to where the target element is visible
                    if should_return:
                        target_scroll_index = best_candidate.get('scroll_index', 0)
                        current_scroll_count = len(executed_scrolls)
                        reverse_count = current_scroll_count - target_scroll_index

                        if reverse_count > 0:
                            self.logger.info(f"Target found at scroll_index={target_scroll_index}, current at {current_scroll_count}, reversing {reverse_count} scrolls...")
                            reverse_dir = {'down': 'up', 'up': 'down', 'right': 'left', 'left': 'right'}
                            # Reverse from the last executed scroll
                            for j in range(reverse_count):
                                scroll_event = executed_scrolls[current_scroll_count - 1 - j]
                                state_before = self.current_state.state_str if self.current_state else None
                                scroll_event.direction = reverse_dir[scroll_event.direction]
                                self.device.send_event(scroll_event)
                                time.sleep(0.3)
                                self.exploration_step += 1
                                self.current_state = self.device.get_current_state()

                                # Save screenshot after reverse scroll for tracking
                                if self.current_state:
                                    self.current_state.tag = f"same_page_{self.exploration_step}"
                                    state_dir = os.path.join(self.exploration_tmp_dir, "states")
                                    self.current_state.save2dir(state_dir)

                                # Record reverse scroll to trace
                                self.repair_trace.append({
                                    'step': self.exploration_step,
                                    'action': 'reverse_scroll',
                                    'direction': scroll_event.direction,
                                    'event': {
                                        'type': 'scroll',
                                        'view': scroll_event.view if hasattr(scroll_event, 'view') else None
                                    },
                                    'state_before': state_before,
                                    'state_after': self.current_state.state_str if self.current_state else None,
                                    'screenshot': os.path.join(self.exploration_tmp_dir, f"states/screen_same_page_{self.exploration_step}.png"),
                                    'xml': os.path.join(self.exploration_tmp_dir, f"xmls/xml_same_page_{self.exploration_step}.xml")
                                })

                        self.save_repair_trace()
                        return repaired_event

                    # On failure, reverse scroll to restore page state for subsequent cross-page logic
                    if executed_scrolls:
                        self.logger.info(f"Reversing {len(executed_scrolls)} scroll operations to restore page state...")
                        reverse_dir = {'down': 'up', 'up': 'down', 'right': 'left', 'left': 'right'}
                        for scroll_event in reversed(executed_scrolls):
                            state_before = self.current_state.state_str if self.current_state else None
                            scroll_event.direction = reverse_dir[scroll_event.direction]
                            self.device.send_event(scroll_event)
                            time.sleep(0.3)
                            self.exploration_step += 1
                            self.current_state = self.device.get_current_state()

                            # Save screenshot after reverse scroll for tracking
                            if self.current_state:
                                self.current_state.tag = f"same_page_{self.exploration_step}"
                                state_dir = os.path.join(self.exploration_tmp_dir, "states")
                                self.current_state.save2dir(state_dir)

                            # Record reverse scroll to trace
                            self.repair_trace.append({
                                'step': self.exploration_step,
                                'action': 'reverse_scroll',
                                'direction': scroll_event.direction,
                                'event': {
                                    'type': 'scroll',
                                    'view': scroll_event.view if hasattr(scroll_event, 'view') else None
                                },
                                'state_before': state_before,
                                'state_after': self.current_state.state_str if self.current_state else None,
                                'screenshot': os.path.join(self.exploration_tmp_dir, f"states/screen_same_page_{self.exploration_step}.png"),
                                'xml': os.path.join(self.exploration_tmp_dir, f"xmls/xml_same_page_{self.exploration_step}.xml")
                            })

                else:
                    # Same page has no scroll events, judge directly
                    if matched_view:
                        self.logger.info(f"Step {self.exploration_step}: Found target (same page, no scroll), using LLM to judge...")
                        # Do not pass matched view here; only judge whether on the correct page
                        judge_result = self.llm_judge_exploration_success(matched_view=matched_view, is_has_next_screen_summary=not self.without_next_screen_summary, is_has_taxonomy=not self.without_taxonomy)
                        self.logger.info(f"LLM judge result: {judge_result}")

                        if judge_result:
                            self.logger.info("✓ Found target element (same page, no scroll)")

                            self.repair_trace.append({
                                'step': self.exploration_step,
                                'action': 'match_found',
                                'event': None,
                                'state_after': self.current_state.state_str,
                                'screenshot': os.path.join(self.exploration_tmp_dir, f"states/screen_same_page_{self.exploration_step}.png"),
                                'xml': os.path.join(self.exploration_tmp_dir, f"xmls/xml_same_page_{self.exploration_step}.xml"),
                                'found_target': True,
                                'matched_view': matched_view,
                                'judge_result': judge_result,
                                'matching_method': matching_method
                            })

                            repaired_event = self._create_repaired_event(matched_view)
                            self.last_repaired_view = matched_view

                            repaired_event_info = {
                                'tag': f"repaired_event_step_{self.exploration_step}",
                                'event': {
                                    'event_type': repaired_event.event_type if hasattr(repaired_event, 'event_type') else 'touch',
                                    'log_lines': None,
                                    'x': None,
                                    'y': None,
                                    'view': matched_view
                                },
                                'start_state': self.current_state.state_str if self.current_state else None,
                                'stop_state': None,
                                'event_str': repaired_event.get_event_str(self.current_state) if hasattr(repaired_event, 'get_event_str') else str(repaired_event)
                            }
                            event_dir = os.path.join(self.exploration_tmp_dir, "events")
                            if not os.path.exists(event_dir):
                                os.makedirs(event_dir)
                            repaired_event_path = os.path.join(event_dir, f"event_repaired_step_{self.exploration_step}.json")
                            with open(repaired_event_path, 'w') as f:
                                json.dump(repaired_event_info, f, indent=2, ensure_ascii=False)
                            self.logger.info(f"Saved repaired event to: {repaired_event_path}")

                            self.save_repair_trace()
                            return repaired_event
                        else:
                            self.logger.info("Same page judge failed, continuing to navigation...")
                            matched_view = None

            # 4. Cross page handling
            if cross_page and matched_view:
                self.logger.info(f"Step {self.exploration_step}: Found potential target (cross page), using LLM to judge...")
                # Do not pass matched view here; only judge whether on the correct page
                judge_result = self.llm_judge_exploration_success(matched_view=matched_view, is_has_next_screen_summary=not self.without_next_screen_summary, is_has_taxonomy=not self.without_taxonomy)
                self.logger.info(f"LLM judge result: {judge_result}")

                if judge_result:
                    self.logger.info("✓ Found target element (cross page, judge succeeded)")

                    step_info = {
                        'step': self.exploration_step,
                        'action': 'match_found',
                        'event': None,
                        'state_after': self.current_state.state_str,
                        'screenshot': os.path.join(self.exploration_tmp_dir, f"states/screen_same_page_{self.exploration_step}.png"),
                        'xml': os.path.join(self.exploration_tmp_dir, f"xmls/xml_same_page_{self.exploration_step}.xml"),
                        'found_target': True,
                        'matched_view': matched_view,
                        'judge_result': judge_result,
                        'matching_method': matching_method
                    }
                    self.repair_trace.append(step_info)

                    repaired_event = self._create_repaired_event(matched_view)
                    self.last_repaired_view = matched_view

                    repaired_event_info = {
                        'tag': f"repaired_event_step_{self.exploration_step}",
                        'event': {
                            'event_type': repaired_event.event_type if hasattr(repaired_event, 'event_type') else 'touch',
                            'log_lines': None,
                            'x': None,
                            'y': None,
                            'view': matched_view
                        },
                        'start_state': self.current_state.state_str if self.current_state else None,
                        'stop_state': None,
                        'event_str': repaired_event.get_event_str(self.current_state) if hasattr(repaired_event, 'get_event_str') else str(repaired_event)
                    }
                    event_dir = os.path.join(self.exploration_tmp_dir, "events")
                    if not os.path.exists(event_dir):
                        os.makedirs(event_dir)
                    repaired_event_path = os.path.join(event_dir, f"event_repaired_step_{self.exploration_step}.json")
                    with open(repaired_event_path, 'w') as f:
                        json.dump(repaired_event_info, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"Saved repaired event to: {repaired_event_path}")

                    self.save_repair_trace()
                    return repaired_event
                else:
                    self.logger.info("Cross page judge failed, continuing exploration...")
                    matched_view = None

            # 4. Target element not found (or cross page judge failed), use LLM to recommend navigation
            self.logger.info(f"Step {self.exploration_step}: Target not found, asking LLM for navigation recommendation...")
            # Re-fetch possible_events (views may have changed after scroll)
            possible_events = self.current_state.get_possible_input(package_name=self.app.get_package_name())
            # Filter according to UIMatch rules
            self.logger.info(f"Filtering events: original count = {len(possible_events)}")
            possible_events = self._filter_events_by_rules(possible_events)
            if self.app.get_package_name() == "com.amaze.filemanager" or self.app.get_package_name() == "com.appmindlab.nano":
                possible_events = self.current_state.get_possible_input_only_leaf_nodes(self.app.get_package_name())
            self.logger.info(f"After filtering: {len(possible_events)} events")

            # 4.1 Get unique views and the event types for each view
            # Deduplicate by bounds + class + text + resource_id + content_description
            unique_views = []  # list of views (deduplicated)
            view_event_types = {}  # view_index -> [event_types]
            view_to_events = {}  # view_index -> {event_type: event}

            for event in possible_events:
                if hasattr(event, 'view') and event.view:
                    view = event.view
                    bounds = view.get('bounds', [])
                    view_class = view.get('class', '')
                    text = view.get('text', '')
                    resource_id = view.get('resource_id', '')
                    content_desc = view.get('content_description', '')

                    # Check if a view with the same attributes already exists
                    existing_idx = None
                    for idx, existing_view in enumerate(unique_views):
                        existing_bounds = existing_view.get('bounds', [])
                        existing_class = existing_view.get('class', '')
                        existing_text = existing_view.get('text', '')
                        existing_resource_id = existing_view.get('resource_id', '')
                        existing_content_desc = existing_view.get('content_description', '')

                        # Only consider it the same widget if all attributes match
                        if (bounds == existing_bounds and
                            view_class == existing_class and
                            text == existing_text and
                            resource_id == existing_resource_id and
                            content_desc == existing_content_desc):
                            existing_idx = idx
                            break

                    # Get event type
                    event_type = getattr(event, 'event_type', 'touch')

                    if existing_idx is not None:
                        # View already exists, add event type
                        if event_type not in view_event_types[existing_idx]:
                            view_event_types[existing_idx].append(event_type)
                        view_to_events[existing_idx][event_type] = event
                    else:
                        # New view
                        new_idx = len(unique_views)
                        unique_views.append(view)
                        view_event_types[new_idx] = [event_type]
                        view_to_events[new_idx] = {event_type: event}

            self.logger.info(f"Extracted {len(unique_views)} unique views from {len(possible_events)} events")

            # Filter out already visited view + event_type combinations
            current_activity = self.current_state.foreground_activity
            filtered_indices = []
            for idx, view in enumerate(unique_views):
                event_types_for_view = view_event_types.get(idx, [])
                # Check if all event_types for this view have been visited
                has_unvisited = False
                for et in event_types_for_view:
                    view_id = self._get_view_navigation_id(view, current_activity, et)
                    if view_id not in self.visited_navigation_elements:
                        has_unvisited = True
                        break
                if has_unvisited:
                    filtered_indices.append(idx)
                else:
                    self.logger.info(f"Filtering out visited element: {view.get('resource_id', '')} {view.get('class', '')}")

            # Rebuild the filtered list
            if len(filtered_indices) < len(unique_views):
                self.logger.info(f"Filtered {len(unique_views) - len(filtered_indices)} visited elements, {len(filtered_indices)} remaining")
                new_unique_views = []
                new_view_event_types = {}
                new_view_to_events = {}
                for new_idx, old_idx in enumerate(filtered_indices):
                    new_unique_views.append(unique_views[old_idx])
                    new_view_event_types[new_idx] = view_event_types[old_idx]
                    new_view_to_events[new_idx] = view_to_events[old_idx]
                unique_views = new_unique_views
                view_event_types = new_view_event_types
                view_to_events = new_view_to_events

            recommended_idx, recommended_event_type = self.llm_recommend_exploration(unique_views, view_event_types, self.exploration_step, is_has_next_screen_summary=not self.without_next_screen_summary, is_has_history_summary=not self.without_history_summary)

            # Handle BACK action
            if recommended_idx == -1 and recommended_event_type == 'back':
                self.logger.info("LLM recommended BACK action, executing...")
                back_event = KeyEvent(name="BACK")
                self.device.send_event(back_event)
                time.sleep(1)

                # Record BACK action
                new_state = self.device.get_current_state()
                step_info = {
                    'step': self.exploration_step,
                    'action': 'back',
                    'state_after': new_state.state_str if new_state else None,
                    'screenshot': None,  # No screenshot needed for back action
                }
                self.repair_trace.append(step_info)

                self.current_state = new_state
                self.exploration_step += 1  # Increment step count
                continue  # Continue to next exploration round

            # Handle SCROLL_DOWN action
            if recommended_idx == -2 and recommended_event_type == 'scroll_down':
                self.logger.info("LLM recommended SCROLL_DOWN action, executing...")

                # Use u2 to scroll down (swipe up from lower-middle of screen)
                try:
                    self.device.u2.swipe(0.5, 0.7, 0.5, 0.3, duration=0.3)
                    time.sleep(1)
                except Exception as e:
                    self.logger.warning(f"SCROLL_DOWN failed: {e}")

                # Record SCROLL_DOWN action
                new_state = self.device.get_current_state()
                step_info = {
                    'step': self.exploration_step,
                    'action': 'scroll_down',
                    'state_after': new_state.state_str if new_state else None,
                    'screenshot': None,  # No screenshot needed for scroll_down action
                }
                self.repair_trace.append(step_info)

                self.current_state = new_state
                self.exploration_step += 1  # Increment step count
                continue  # Continue to next exploration round

            if recommended_idx is None or recommended_idx < 0 or recommended_idx >= len(unique_views):
                self.logger.warning(f"Invalid LLM recommendation: {recommended_idx}, stopping exploration")
                continue

            # 5. Get the corresponding event from view_to_events
            events_for_view = view_to_events.get(recommended_idx, {})
            if recommended_event_type and recommended_event_type in events_for_view:
                event = events_for_view[recommended_event_type]
            elif events_for_view:
                # Use the first available event type
                first_type = list(events_for_view.keys())[0]
                event = events_for_view[first_type]
                self.logger.info(f"Event type '{recommended_event_type}' not available, using '{first_type}' instead")
            else:
                self.logger.warning(f"No events found for view index {recommended_idx}")
                continue

            self.logger.info(f"Clicking navigation element [view {recommended_idx}, type {recommended_event_type}]: {event}")

            # Mark this element + event_type as visited
            # if recommended_idx != -1 and recommended_event_type != 'back':
            #     clicked_view = unique_views[recommended_idx]
            #     clicked_event_type = event.event_type if hasattr(event, 'event_type') else 'touch'
            #     view_id = self._get_view_navigation_id(clicked_view, self.current_state.foreground_activity, clicked_event_type)
            #     self.visited_navigation_elements.add(view_id)
            #     self.logger.info(f"Marked as visited: {view_id}")

            # Record the activity before clicking
            activity_before = self.current_state.foreground_activity

            self.device.send_event(event)
            time.sleep(1)

            # 6. Get the new state after clicking
            new_state = self.device.get_current_state()

            # Record activity changes during exploration (for back navigation decisions)
            activity_after = new_state.foreground_activity if new_state else None
            self.exploration_activity_trace.append({
                'step': self.exploration_step,
                'action': 'navigation',
                'activity_before': activity_before,
                'activity_after': activity_after
            })
            self.logger.info(f"Activity trace: {activity_before} → {activity_after}")

            # 7. Record the navigation step trace
            step_info = {
                'step': self.exploration_step,
                'action': 'navigation',
                'recommended_idx': recommended_idx,
                'event': {
                    'type': event.event_type if hasattr(event, 'event_type') else 'unknown',
                    'view': event.view if hasattr(event, 'view') else None
                },
                'state_after': new_state.state_str if new_state else None,
                'screenshot': os.path.join(self.exploration_tmp_dir, f"states/screen_same_page_{self.exploration_step}.png"),
                'marked_screenshot': os.path.join(self.exploration_tmp_dir, f"images/marked_recommended_step_{self.exploration_step}.png"),
                'xml': os.path.join(self.exploration_tmp_dir, f"xmls/xml_same_page_{self.exploration_step}.xml"),
                'found_target': False,
                'matched_view': None
            }
            self.repair_trace.append(step_info)

            # 8. Save navigation event to file
            event_info = {
                'tag': f"navigation_step_{self.exploration_step}",
                'event': {
                    'event_type': event.event_type if hasattr(event, 'event_type') else 'touch',
                    'log_lines': None,
                    'x': None,
                    'y': None,
                    'view': event.view if hasattr(event, 'view') else None
                },
                'start_state': self.current_state.state_str if self.current_state else None,
                'stop_state': new_state.state_str if new_state else None,
                'event_str': event.get_event_str(self.current_state) if hasattr(event, 'get_event_str') else str(event)
            }
            event_dir = os.path.join(self.exploration_tmp_dir, "events")
            if not os.path.exists(event_dir):
                os.makedirs(event_dir)
            event_path = os.path.join(event_dir, f"event_navigation_step_{self.exploration_step}.json")
            with open(event_path, 'w') as f:
                json.dump(event_info, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved navigation event to: {event_path}")

            # 9. Update current state, increment step count, continue to next loop
            self.current_state = new_state
            self.exploration_step += 1

        # Exploration complete, target not found
        self.logger.warning("Exploration finished, target element not found")
        self.save_repair_trace()
        return None


   


    def _can_back_to_activity(self, target_activity):
        """
        Determine whether the target activity can be reached by pressing the Back key

        Based on activity_trace and exploration_activity_trace:
        - If target_activity is upstream (in back stack), back is possible
        - If target_activity is downstream or on a different branch, back is not possible

        Returns:
            (can_back, estimated_back_count): whether back is possible, estimated presses needed
        """
        current_state = self.device.get_current_state()
        current_activity = current_state.foreground_activity

        # Already at target activity
        if current_activity == target_activity:
            return True, 0

        # Merge activity traces from replay phase and exploration phase
        all_trace = self.activity_trace + self.exploration_activity_trace

        if not all_trace:
            self.logger.warning("No activity trace available, cannot determine back feasibility")
            return False, 0

        # Find the last occurrence of target activity in the trace
        target_idx = -1
        for i in range(len(all_trace) - 1, -1, -1):
            if all_trace[i].get('activity_after') == target_activity:
                target_idx = i
                break

        if target_idx == -1:
            self.logger.info(f"Target activity {target_activity} never appeared in trace, cannot back")
            return False, 0

        # Count activity changes after target_idx (this is the number of back presses needed)
        back_count = 0
        for i in range(target_idx + 1, len(all_trace)):
            before = all_trace[i].get('activity_before')
            after = all_trace[i].get('activity_after')
            if before and after and before != after:
                back_count += 1

        self.logger.info(f"Target activity at trace[{target_idx}], estimated {back_count} backs needed")
        return True, back_count

    def _navigate_back_to_activity(self, target_activity, max_back_presses=10):
        """
        Press Back key until returning to the target activity

        First check if back is possible (based on activity graph), give up if not

        Args:
            target_activity: name of the target activity
            max_back_presses: maximum number of Back presses

        Returns:
            (back_events, success): list of events during back navigation and whether it succeeded
        """
        back_events = []

        # First check if back is possible
        can_back, estimated_count = self._can_back_to_activity(target_activity)
        if not can_back:
            self.logger.warning(f"Cannot back to {target_activity} (not in back stack), giving up")
            return back_events, False

        # Use estimated back count, but do not exceed max_back_presses
        actual_max = min(estimated_count + 2, max_back_presses)  # +2 as error tolerance
        self.logger.info(f"Trying to back to {target_activity}, estimated {estimated_count} backs, max {actual_max}")

        for i in range(actual_max):
            current_state = self.device.get_current_state()
            current_activity = current_state.foreground_activity

            # Already at target activity
            if current_activity == target_activity:
                self.logger.info(f"✓ Back to {target_activity} after {i} back presses")
                return back_events, True

            # Press Back key
            back_event = KeyEvent("BACK")
            self.device.send_event(back_event)
            time.sleep(0.5)

            new_state = self.device.get_current_state()
            back_events.append({
                'action': 'back',
                'step': i,
                'from_activity': current_activity,
                'to_activity': new_state.foreground_activity if new_state else None
            })

            self.logger.info(f"BACK ({i+1}/{actual_max}): {current_activity} → {new_state.foreground_activity if new_state else '?'}")

            # Check if the app was exited
            if new_state and new_state.foreground_activity and self.app.package_name not in new_state.foreground_activity:
                self.logger.warning(f"Exited app during back navigation, stopping")
                return back_events, False

        # Final check
        current_state = self.device.get_current_state()
        current_activity = current_state.foreground_activity
        if current_activity == target_activity:
            self.logger.info(f"✓ Back to {target_activity}")
            return back_events, True

        self.logger.warning(f"Failed to back to {target_activity}, current: {current_activity}")
        return back_events, False

    def llm_recommend_exploration(self, unique_views, view_event_types, step_num, is_has_history = False, is_has_history_summary = True, is_has_next_screen_summary = True):
        """
        Use LLM to recommend the component most likely to contain the target element

        Args:
            unique_views: deduplicated view list
            view_event_types: view_index -> [event_types] dictionary
            step_num: current step number
            is_has_history: whether there is operation history, passes history images to let LLM know which paths have been tried
            is_has_history_summary: whether there is operation history summary, passes a summary to let LLM know which paths have been tried
        Returns:
            (recommended_idx, recommended_event_type) tuple, returns (None, None) on failure
        """
        try:
            

            # 1. Read the marked image of the original element (red box)
            marked_original_path = self.failed_event_png_path.replace(".png", "_marked_original_element.png")
            if not os.path.exists(marked_original_path):
                # If not exists, need to generate
                original_element = self._find_original_element(self.failed_event_path, self.failed_event_xml_tree)
                if original_element is None:
                    self.logger.warning("Cannot find original element")
                    return None, None
                original_bounds = self._parse_bounds(original_element.attrib.get("bounds", ""))
                original_img = read_image(self.failed_event_png_path)
                marked_original_img = draw_original_element_on_image(original_img, original_bounds)
                marked_original_img.save(marked_original_path)
            else:
                marked_original_img = read_image(marked_original_path)

            # 2. Mark all unique views on the current page (green box)
            current_screenshot = self.current_state.screenshot_path
            current_img = read_image(current_screenshot)
            current_img_backup = current_img.copy()

            for i, view in enumerate(unique_views):
                bounds = view.get('bounds')
                if bounds:
                    bounds_str = f"[{bounds[0][0]},{bounds[0][1]}][{bounds[1][0]},{bounds[1][1]}]"
                    current_img = draw_replay_element_on_image(current_img, bounds_str, id=i)

            screen_path = os.path.join(self.exploration_tmp_dir, "images")
            if not os.path.exists(screen_path):
                os.makedirs(screen_path)

            # Save the marked image
            marked_candidates_path = os.path.join(screen_path, f"marked_exploration_candidates_step_{step_num}.png")
            current_img.save(marked_candidates_path)

            # 3. Crop the original element image
            original_element = self._find_original_element(self.failed_event_path, self.failed_event_xml_tree)
            original_bounds = self._parse_bounds(original_element.attrib.get("bounds", ""))
            # Convert to PIL crop format (x1, y1, x2, y2)
            crop_bounds = (original_bounds[0][0], original_bounds[0][1], original_bounds[1][0], original_bounds[1][1])
            original_full_img = read_image(self.failed_event_png_path)
            original_element_img = original_full_img.crop(crop_bounds)

            # 4. Encode images
            marked_original_base64 = get_encoded_image(marked_original_img)
            original_element_base64 = get_encoded_image(original_element_img)
            marked_candidates_base64 = get_encoded_image(current_img)

            # 5. Collect previous operation history (to let LLM know which paths have been tried)
            exploration_history = []
            if is_has_history:
                for trace_step in self.repair_trace:
                    action = trace_step.get('action', '')
                    if action == 'navigation':
                        # Use the already generated marked_recommended_image (showing the clicked element)
                        marked_screenshot_path = trace_step.get('marked_screenshot')
                        if marked_screenshot_path and os.path.exists(marked_screenshot_path):
                            # Get the actual event_type (touch, long_touch, set_text, etc.)
                            event_info = trace_step.get('event', {})
                            event_type = event_info.get('type', 'touch') if isinstance(event_info, dict) else 'touch'
                            self.logger.info(f"Adding history step {trace_step.get('step')}: {marked_screenshot_path}, event_type: {event_type}")
                            step_img = read_image(marked_screenshot_path)
                            exploration_history.append({
                                'step': trace_step.get('step'),
                                'event_type': event_type,
                                'base64': get_encoded_image(step_img)
                            })
                    elif action == 'back':
                        self.logger.info(f"Adding history step {trace_step.get('step')}: BACK action")
                        exploration_history.append({
                            'step': trace_step.get('step'),
                            'event_type': 'back'
                        })
                    elif action == 'scroll_down':
                        self.logger.info(f"Adding history step {trace_step.get('step')}: SCROLL_DOWN action")
                        exploration_history.append({
                            'step': trace_step.get('step'),
                            'event_type': 'scroll_down'
                        })

                self.logger.info(f"Collected {len(exploration_history)} exploration history steps")

            if is_has_history_summary:
                for i, trace_step in enumerate(self.repair_trace):
                    action = trace_step.get('action', '')

                    # Handle back action: add directly, no LLM summary needed
                    if action == 'back':
                        exploration_history.append({
                            'step': trace_step.get('step'),
                            'summary': 'Pressed BACK button → Returned to previous screen'
                        })
                        self.logger.info(f"Added BACK action for step {trace_step.get('step')}")
                        continue

                    # Handle scroll_down action
                    if action == 'scroll_down':
                        exploration_history.append({
                            'step': trace_step.get('step'),
                            'summary': 'Scrolled DOWN → Revealed more content below'
                        })
                        self.logger.info(f"Added SCROLL_DOWN action for step {trace_step.get('step')}")
                        continue

                    if action == 'navigation':  # navigation requires LLM-generated summary
                        # If summary already exists, use it directly
                        if trace_step.get('summary') is not None:
                            summary = trace_step.get('summary')
                            exploration_history.append({
                                'step': trace_step.get('step'),
                                'summary': summary
                            })
                            self.logger.info(f"Using existing summary for step {trace_step.get('step')}: {summary[:50]}...")
                            continue

                        # Get the marked_screenshot for current step (with clicked element marked)
                        marked_screenshot_path = trace_step.get('marked_screenshot')
                        screenshot_path = trace_step.get('screenshot')

                        if not marked_screenshot_path or not os.path.exists(marked_screenshot_path):
                            self.logger.warning(f"Step {trace_step.get('step')}: marked_screenshot not found")
                            continue

                        # Get next step screenshot as next_img (result after clicking)
                        next_screenshot_path = None
                        if i + 1 < len(self.repair_trace):
                            next_step = self.repair_trace[i + 1]
                            next_screenshot_path = next_step.get('screenshot')

                        if not next_screenshot_path or not os.path.exists(next_screenshot_path):
                            self.logger.warning(f"Step {trace_step.get('step')}: next screenshot not found")
                            continue

                        # Get element bounds for cropping
                        event_info = trace_step.get('event', {})
                        view_info = event_info.get('view', {}) if isinstance(event_info, dict) else {}
                        bounds = view_info.get('bounds') if isinstance(view_info, dict) else None

                        # Read images
                        current_marked_img = read_image(marked_screenshot_path)
                        next_img = read_image(next_screenshot_path)

                        # Crop element image
                        if bounds and isinstance(bounds, list) and len(bounds) == 2:
                            try:
                                x1, y1 = bounds[0]
                                x2, y2 = bounds[1]
                                current_screenshot_img = read_image(screenshot_path)
                                current_element_img = current_screenshot_img.crop((x1, y1, x2, y2))
                            except Exception as e:
                                self.logger.warning(f"Failed to crop element: {e}, using full screenshot")
                                current_element_img = current_marked_img
                        else:
                            self.logger.warning(f"Failed to get bounds for step {trace_step.get('step')}")
                            continue

                        # Convert to base64
                        current_marked_img_base64 = get_encoded_image(current_marked_img)
                        current_element_img_base64 = get_encoded_image(current_element_img)
                        next_img_base64 = get_encoded_image(next_img)

                        # Call LLM to generate summary
                        self.logger.info(f"Generating summary for step {trace_step.get('step')}...")
                        summary = self.get_current_navigatiton_summary(
                            current_marked_img_base64,
                            current_element_img_base64,
                            next_img_base64
                        )

                        if summary:
                            trace_step['summary'] = summary
                            exploration_history.append({
                                'step': trace_step.get('step'),
                                'summary': summary
                            })
                            self.logger.info(f"Generated summary for step {trace_step.get('step')}: {summary[:50]}...")
                        else:
                            self.logger.warning(f"Failed to generate summary for step {trace_step.get('step')}")


            # 6. Construct prompt (passing view_event_types and operation history)
            system_prompt, user_prompt = self._construct_exploration_llm_prompt(
                marked_original_base64, original_element_base64, marked_candidates_base64,
                view_event_types, exploration_history
            )

            # 7. Call the LLM
            response, token_usage = openai_chat(system_prompt, user_prompt, self.llm_api_key, "gpt-5.1", "gpt")
            self.logger.info(f"LLM exploration response: {response}")
            self.logger.info(f"Token usage: {token_usage}")

            # 8. Parse the returned component ID and event type
            import re

            # First check if [BACK] was returned
            if re.search(r'\[BACK\]', response, re.IGNORECASE):
                self.logger.info("LLM recommended BACK action")
                return -1, 'back'  # Return -1 to indicate BACK action

            if re.search(r'\[SCROLL_DOWN\]', response, re.IGNORECASE):
                self.logger.info("LLM recommended SCROLL_DOWN action")
                return -2, 'scroll_down'  # Return -2 to indicate SCROLL_DOWN action

            # Try to match format: [index:event_type] or [index]
            pattern_with_type = r'\[(\d+):(\w+)\]'
            pattern_simple = r'\[(\d+)\]'

            match_with_type = re.search(pattern_with_type, response)
            match_simple = re.search(pattern_simple, response)

            recommended_idx = None
            recommended_event_type = 'touch'  # Default to touch

            if match_with_type:
                recommended_idx = int(match_with_type.group(1))
                recommended_event_type = match_with_type.group(2)
            elif match_simple:
                recommended_idx = int(match_simple.group(1))

            if recommended_idx is not None:
                self.logger.info(f"LLM recommended view index: {recommended_idx}, event_type: {recommended_event_type}")

                # Check if view index is valid
                if recommended_idx >= len(unique_views) or recommended_idx < 0:
                    self.logger.warning("LLM recommended view index out of range, skipping")
                    return None, None

                # Generate marked_recommended_image
                marked_recommended_image_path = os.path.join(screen_path, f"marked_recommended_step_{step_num}.png")
                view = unique_views[recommended_idx]
                bounds = view.get('bounds')
                if bounds:
                    bounds_str = f"[{bounds[0][0]},{bounds[0][1]}][{bounds[1][0]},{bounds[1][1]}]"
                    current_img_backup = draw_replay_element_on_image(current_img_backup, bounds_str, id=recommended_idx)
                    current_img_backup.save(marked_recommended_image_path)

                return recommended_idx, recommended_event_type
            else:
                self.logger.warning("LLM did not return a valid component index")
                return None, None

        except Exception as e:
            self.logger.error(f"Error in llm_recommend_exploration: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def get_current_navigatiton_summary(self, current_marked_img_base64, current_element_img_base64, next_img_base64):
        """
        Get a summary of the current navigation action, to be passed to LLM later to avoid repeatedly clicking the same element.

        Input:
        - current_marked_img_base64: Screenshot of the current action (red box marks the clicked element)
        - current_element_img_base64: Cropped image of the clicked element
        - next_img_base64: Screenshot after clicking

        Returns:
        - Action summary string (describes what was clicked and what effect it produced)
        """
        try:
            system_prompt = """You are an Android UI navigation analyst. Your task is to summarize a navigation action for future reference.

Given:
1. A screenshot with the clicked element marked in a GREEN box
2. A zoomed-in image of that clicked element
3. The screenshot AFTER clicking the element

Please provide a concise summary (1-2 sentences) following this structure:
"Clicked [element description] → [page transition status]. [what changed]"

Structure:
1. What element was clicked (describe it briefly)
2. Whether the page navigated or stayed the same (e.g., "Stayed on same page", "Navigated to new page", "Returned to previous page")
3. What specifically changed compared to before (e.g., dialog appeared, menu opened, checkbox toggled, content updated)

Examples:
- "Clicked 'Settings' menu icon → Navigated to new page. Shows Settings screen with list of preference options."
- "Clicked 'More options' button (three dots) → Stayed on same page. A dropdown menu appeared with Edit/Delete/Share options."
- "Clicked 'Force Dark Theme' checkbox → Stayed on same page. A 'Privileges' dialog appeared requesting root/ADB access."
- "Clicked 'Save' button → Returned to previous page. Changes saved successfully."
- "Clicked hamburger menu icon → Stayed on same page. Navigation drawer slid in from left."

Be factual and concise. This summary will help avoid clicking the same element repeatedly."""

            user_prompt = {
                'ori_analyze': [
                    {"type": "text", "text": "## Current Screen\nBelow is the screenshot with the clicked element marked in a GREEN box:"},
                    current_marked_img_base64,
                    {"type": "text", "text": "## Clicked Element\nBelow is the zoomed-in image of the clicked element:"},
                    current_element_img_base64
                ],
                'update_analyze': [
                    {"type": "text", "text": "## Screen After Click\nBelow is the screenshot AFTER clicking the element:"},
                    next_img_base64,
                    {"type": "text", "text": "Please describe: 1) What was clicked 2) Did the page navigate or stay the same? 3) What changed?"}
                ]
            }

            response, token_usage = openai_chat(
                system_prompt, user_prompt,
                self.llm_api_key, "gpt-5.1", "gpt"
            )

            summary = response.strip()
            self.logger.info(f"Navigation summary: {summary}")
            self.logger.info(f"Token usage: {token_usage}")

            return summary

        except Exception as e:
            self.logger.error(f"Error in get_current_navigatiton_summary: {e}")
            import traceback
            traceback.print_exc()
            return None

    
    def _construct_exploration_llm_prompt(self, marked_original_img_base64, original_element_img_base64, marked_replay_img_base64, view_event_types=None, exploration_history=None, original_next_screen_summary=None):
        """
        Construct the LLM prompt for exploration recommendation.

        Args:
            view_event_types: Optional, {view_index: [event_types]} dictionary
            exploration_history: Optional, list of previous action history
            original_next_screen_summary: Optional, summary of the next screen
        """

        system_prompt = """
You are an Android developer skilled at analyzing GUI layouts and understanding how UI widgets relate and evolve across different app versions.

In software version iterations, the original target widget may no longer be visible in the updated screen. It may be relocated into a menu, settings page, dialog, drawer, collapsible item, or other UI entry point.

## TASK:
1. Read the original UI information and understand the purpose and meaning of the original widget (marked with red boxes).
2. Read the updated version's screenshot (marked with green boxes showing all clickable UI components).
3. Infer which green-boxed UI widget is the MOST LIKELY ENTRY POINT that the user should click next to reveal or access the target widget.
4. If the potential widget supports multiple interaction types (for example, touch, long_touch), choose the most appropriate one based on the functionality of the original widget.


## GUIDELINES:
1. You are NOT performing similarity matching. You are performing **functional and structural inference** based on UI design conventions.
2. If the current screen is a temporary dialog or blocking screen that prevents accessing the target functionality, you should recommend a BACK action.
3. Analyze the exploration history step by step:
   - Prioritize paths that are semantically closest to the target widget.
   - Revisit previously opened containers to explore unexplored sub-options, while avoiding repeated navigation sequences.
   - Detect dead-end paths and perform BACK to restore alternative exploration branches.
   - If the search becomes overly deep without progress, backtrack to higher-level entry points and redirect exploration.


## OUTPUT FORMAT:
Return the most likely UI widget's Number and the recommended interaction type, or explicitly recommend BACK.

EXAMPLE OUTPUT:

```result.md
### Analyze_Process
Explain why the target widget is likely located inside a specific menu or entry point, and describe the reasoning used to choose the best candidate.

### Recommended_UI_No
[18:touch]
OR
[BACK]
```

Note: The format is [index:event_type], where event_type can be touch, long_touch, etc.
If only one action type is available, just use that type.

"""

        analyze_ori_scenarios_prompt = f"""
I will provide you with the original application version's screenshot (marked with red boxes indicating an original UI widget).


* Original Screenshot
```
Please see the above Figure.
```
"""

        analyze_ori_element_prompt = f"""
I will provide you with the original UI widget Figure.

* Original UI widget Figure
```
Please see the above Figure.
```
"""
        # Build view event types description
        view_info_str = ""
        if view_event_types:
            view_info_str = "\n\n**Available UI Components and their interaction types:**\n"
            for idx in sorted(view_event_types.keys()):
                event_types = view_event_types[idx]
                event_types_str = ", ".join(event_types) if event_types else "touch"
                view_info_str += f"- [{idx}]: available actions: {event_types_str}\n"

        analyze_update_effects_prompt = f"""
I will provide you with the updated application version's screenshot. Different UI components are marked with green boxes and assigned a numerical sequence number.


* Updated Screenshot
```
Please see the above Figure.
```
{view_info_str}
"""



        user_prompt = {}

        user_prompt['ori_analyze'] = [marked_original_img_base64] + [{"type": "text", "text": analyze_ori_scenarios_prompt}] + [original_element_img_base64] + [{"type": "text", "text": analyze_ori_element_prompt}]

        user_prompt['update_analyze'] = [marked_replay_img_base64] + [
            {"type": "text", "text": analyze_update_effects_prompt}
        ]

        # Add historical action information
        if exploration_history and len(exploration_history) > 0:
            # Check whether to use the new summary format (text summary) or the old image format
            has_summary = any(hist.get('summary') for hist in exploration_history)

            if has_summary:
                # New format: use text summary
                history_prompt = f"""
### Previous Exploration History

The following {len(exploration_history)} navigation steps were performed.
"""
                for i, hist in enumerate(exploration_history):
                    step = hist.get('step', i)
                    summary = hist.get('summary', '')
                    if summary:
                        history_prompt += f"- Step {step}: {summary}\n"

                history_prompt += """
Based on the above history, the previously tried paths did not lead to the target widget.
Please recommend a UI component or action to explore.
"""
                user_prompt['update_analyze'] = user_prompt['update_analyze'] + [
                    {"type": "text", "text": history_prompt}
                ]

            else:
                # Old format: use images
                history_prompt = f"""
### Previous Exploration History

The following {len(exploration_history)} navigation steps were performed but did not reveal the target element.
Each step shows the UI component that was interacted with and the interaction type.
"""
                history_images = []
                for i, hist in enumerate(exploration_history):
                    step = hist.get('step', i)
                    event_type = hist.get('event_type', 'touch')
                    if event_type == 'back':
                        history_prompt += f"- Step {step}: Pressed BACK button\n"
                    else:
                        # Show the actual action type (touch, long_touch, set_text, etc.)
                        action_desc = {
                            'touch': 'Clicked (touch)',
                            'long_touch': 'Long-pressed',
                            'set_text': 'Set text on'
                        }.get(event_type, f'Interacted ({event_type}) with')
                        history_prompt += f"- Step {step}: {action_desc} a UI component (see History Figure {i+1})\n"
                        if hist.get('base64'):
                            history_images.append(hist['base64'])

                history_prompt += "\nThe green-boxed element in each History Figure shows the element that was interacted with in that step.\n"

                # Concatenate history images and text to user_prompt['update_analyze']
                user_prompt['update_analyze'] = user_prompt['update_analyze'] + history_images + [
                    {"type": "text", "text": history_prompt}
                ]

        return system_prompt, user_prompt

   
    def llm_judge_exploration_success(self, matched_view = None, only_last_screen = True, is_has_taxonomy = True, is_has_next_screen_summary = True):
        """
        Use LLM to judge whether the exploration successfully completed the original functionality.

        Input:
        - Original element marked image (red box marks the target element)
        - marked_clicked images for each step during exploration (green box marks click position)
        - Final state screenshot
        - matched_view: Optional, if provided, marks that element on the current screenshot (for verification after scroll voting)

        Output:
        - True: LLM judges exploration as successfully completing the original functionality
        - False: LLM judges exploration as failing to complete the original functionality
        """
        try:

            # 1. Prepare the original element marked image
            marked_original_path = self.failed_event_png_path.replace(".png", "_marked_original_element.png")
            if not os.path.exists(marked_original_path):
                original_element = self._find_original_element(self.failed_event_path, self.failed_event_xml_tree)
                if original_element is None:
                    self.logger.warning("Cannot find original element for judge")
                    return False
                original_bounds = self._parse_bounds(original_element.attrib.get("bounds", ""))
                original_img = read_image(self.failed_event_png_path)
                marked_original_img = draw_original_element_on_image(original_img, original_bounds)
                marked_original_img.save(marked_original_path)
            else:
                marked_original_img = read_image(marked_original_path)

            marked_original_base64 = get_encoded_image(marked_original_img)

            # 2. Crop the original element image
            original_element = self._find_original_element(self.failed_event_path, self.failed_event_xml_tree)
            if original_element is None:
                self.logger.warning("Cannot find original element for cropping")
                return False
            original_bounds = self._parse_bounds(original_element.attrib.get("bounds", ""))
            # Convert to PIL crop format (x1, y1, x2, y2)
            crop_bounds = (original_bounds[0][0], original_bounds[0][1], original_bounds[1][0], original_bounds[1][1])
            original_full_img = read_image(self.failed_event_png_path)
            original_element_img = original_full_img.crop(crop_bounds)
            original_element_img_base64 = get_encoded_image(original_element_img)

            # 3. Collect the marked_recommended_image for each step during exploration
            exploration_steps_images = []

            for trace_step in self.repair_trace:
                if trace_step.get('action') == 'navigation':
                    # Use the already generated marked_recommended_image
                    marked_recommended_image_path = trace_step.get('marked_screenshot')
                    if marked_recommended_image_path and os.path.exists(marked_recommended_image_path):
                        print(f"Adding marked_recommended_image for step {trace_step.get('step')}: {marked_recommended_image_path}")
                        step_img = read_image(marked_recommended_image_path)
                        exploration_steps_images.append({
                            'step': trace_step.get('step'),
                            'base64': get_encoded_image(step_img)
                        })
                        self.logger.info(f"Added marked_recommended_image for step {trace_step.get('step')}: {marked_recommended_image_path}")

            if only_last_screen and len(exploration_steps_images) > 0:
                exploration_steps_images = exploration_steps_images[-3:]
                self.logger.info(f"Only using last 3 screens for judge")
            
            self.logger.info(f"Collected {len(exploration_steps_images)} exploration steps images for judge")

            # 4. Get current page screenshot
            current_screenshot_path = self.current_state.screenshot_path
            if current_screenshot_path and os.path.exists(current_screenshot_path):
                current_img = read_image(current_screenshot_path)
            else:
                self.logger.warning("Current screenshot not found")
                return False

            # 5. If matched_view exists, mark it as the current step (regardless of whether there were previous exploration steps)
            if matched_view:
                matched_bounds = matched_view.get('bounds', None)
                if matched_bounds:
                    self.logger.info(f"Marking matched_view on current screenshot: bounds={matched_bounds}")
                    # Convert bounds format: [[x1, y1], [x2, y2]] -> "[x1,y1][x2,y2]"
                    if isinstance(matched_bounds, list) and len(matched_bounds) == 2:
                        bounds_str = f"[{matched_bounds[0][0]},{matched_bounds[0][1]}][{matched_bounds[1][0]},{matched_bounds[1][1]}]"
                    else:
                        bounds_str = str(matched_bounds)
                    marked_current_img = draw_replay_element_on_image(current_img.copy(), bounds_str, id="1")
                    # save for debug
                    if not os.path.exists(self.exploration_tmp_dir):
                        os.makedirs(self.exploration_tmp_dir)
                    if not os.path.exists(os.path.join(self.exploration_tmp_dir, "images")):
                        os.makedirs(os.path.join(self.exploration_tmp_dir, "images"))
                    marked_current_img_path = os.path.join(self.exploration_tmp_dir, "images", f"marked_matched_view_step_{self.exploration_step + 1}.png")
                    marked_current_img.save(marked_current_img_path)
                    exploration_steps_images.append({
                        'step': self.exploration_step + 1,
                        'base64': get_encoded_image(marked_current_img)
                    })
                else:
                    self.logger.warning("matched_view has no bounds, using unmarked screenshot")
                    exploration_steps_images.append({
                        'step': self.exploration_step + 1,
                        'base64': get_encoded_image(current_img)
                    })
            elif not exploration_steps_images:
                # No matched_view and no previous exploration steps, use the original image
                exploration_steps_images.append({
                    'step': self.exploration_step + 1,
                    'base64': get_encoded_image(current_img)
                })


            # v2: Have LLM summarize the changes from current screen to next screen, then pass the description to LLM to judge success
            if is_has_next_screen_summary == False:
                original_next_screen_summary = None
            else:
                if self.failed_event_png_next and self.original_next_screen_summary is None:
                    next_screen_base64 = get_encoded_image(self.failed_event_png_next)
                    original_next_screen_summary = self.get_next_current_description(marked_original_base64, original_element_img_base64, next_screen_base64)
                    self.original_next_screen_summary = original_next_screen_summary
                elif self.original_next_screen_summary is not None:
                    original_next_screen_summary = self.original_next_screen_summary
                else:
                    original_next_screen_summary = None

            
            # 4. Construct LLM prompt
            if is_has_taxonomy:
                system_prompt, user_prompt = self._construct_judge_exploration_prompt(
                    marked_original_base64,
                    original_element_img_base64,
                    exploration_steps_images,
                    # original_next_screen_base64
                    original_next_screen_summary
                )
            else:
                # Ablation experiments, replace with a plain prompt
                system_prompt, user_prompt = self._construct_judge_exploration_prompt_plain(
                    marked_original_base64,
                    original_element_img_base64,
                    exploration_steps_images,
                    # original_next_screen_base64
                    original_next_screen_summary
                )



            # 5. Call LLM
            response, token_usage = openai_chat(
                system_prompt, user_prompt,
                self.llm_api_key, "gpt-5.1", "gpt"
            )
            self.logger.info(f"LLM judge response: {response}")
            self.logger.info(f"Token usage: {token_usage}")

            # 6. Parse LLM response - only check the last line to avoid false matching phrases like "Yes or No" in the text
            last_line = response.strip().split('\n')[-1].upper().strip()
            if last_line == "YES":
                self.logger.info("✓ LLM judged exploration as SUCCESS")
                return True
            elif last_line == "NO":
                self.logger.info("✗ LLM judged exploration as FAILED")
                return False
            else:
                # If the last line is not purely YES/NO, try checking the last word
                last_word = last_line.split()[-1] if last_line.split() else ""
                if last_word == "YES":
                    self.logger.info("✓ LLM judged exploration as SUCCESS")
                    return True
                elif last_word == "NO":
                    self.logger.info("✗ LLM judged exploration as FAILED")
                    return False
                else:
                    self.logger.warning(f"LLM returned ambiguous result: {response}")
                    return False

        except Exception as e:
            self.logger.error(f"Error in llm_judge_exploration_success: {e}")
            import traceback
            traceback.print_exc()
            return False

    
    def get_next_current_description(self, marked_original_base64, original_element_img_base64, next_screen_base64):
        """
        Have LLM summarize the effect of operating on the current marked widget (1-3 sentences).

        Input:
        - marked_original_base64: Original screenshot (red box marks the target widget)
        - original_element_img_base64: Cropped image of the target widget
        - next_screen_base64: Screenshot after clicking

        Returns:
        - Effect description string
        """
        try:

            system_prompt = """You are an Android UI analysis expert. Your task is to describe the effect of clicking/tapping on a UI element.

Given:
1. A screenshot with the target element marked in a RED box
2. A zoomed-in image of that element
3. The screenshot of the screen AFTER clicking the element

Please describe in 1-3 sentences following this structure:
1. First, state whether the page navigated or stayed the same (e.g., "Stayed on same page", "Navigated to a new page", "Returned to previous page")
2. Then, describe the difference between the current screen and the previous screen
3. Specifically mention what parts changed (e.g., dialog appeared, menu opened, content updated, element state toggled)

Examples:
- "Stayed on same page. A 'Privileges' dialog appeared in the center, prompting the user to grant root access. The dialog contains 'DON'T REMIND', 'GET HELP', and 'OK' buttons."
- "Navigated to a new page. Now showing the Settings screen with a list of preference options including Theme, Language, and About."
- "Stayed on same page. The checkbox state toggled from unchecked to checked. No other visible changes."

Be concise and factual. Do not speculate about functionality not visible in the screenshots."""

            # Construct user_prompt, format compatible with openai_chat
            user_prompt = {
                'ori_analyze': [
                    {"type": "text", "text": "## Original Screen\nBelow is the screenshot with the target element marked in a RED box:"},
                    marked_original_base64,
                    {"type": "text", "text": "## Target Element\nBelow is the zoomed-in image of the target element:"},
                    original_element_img_base64
                ],
                'update_analyze': [
                    {"type": "text", "text": "## Screen After Click\nBelow is the screenshot AFTER clicking the target element:"},
                    next_screen_base64,
                    {"type": "text", "text": "Please describe: 1) Did the page navigate or stay the same? 2) What changed compared to before?"}
                ]
            }

            response, token_usage = openai_chat(
                system_prompt, user_prompt,
                self.llm_api_key, "gpt-5.1", "gpt"
            )

            description = response.strip()
            self.logger.info(f"Next screen description: {description}")
            self.logger.info(f"Token usage: {token_usage}")

            return description

        except Exception as e:
            self.logger.error(f"Error in _get_next_current_description: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        
    
    def llm_select_best_candidate(self, candidates: dict):
        """
        Have LLM select the best match from all candidates.

        Input:
        - Original element marked image (red box)
        - Original element cropped image
        - Screenshot of each candidate (marked position, numbered 1, 2, 3...)

        Output:
        - (view_str, candidate_info) or None (no match)
        """
        try:
            from .UIMatch.utils import (
                read_image, draw_original_element_on_image,
                draw_replay_element_on_image, get_encoded_image, openai_chat
            )

            if not candidates:
                self.logger.warning("No candidates to select from")
                return None

            # 1. Prepare the original element marked image
            marked_original_path = self.failed_event_png_path.replace(".png", "_marked_original_element.png")
            if not os.path.exists(marked_original_path):
                original_element = self._find_original_element(self.failed_event_path, self.failed_event_xml_tree)
                if original_element is None:
                    self.logger.warning("Cannot find original element for selection")
                    return None
                original_bounds = self._parse_bounds(original_element.attrib.get("bounds", ""))
                original_img = read_image(self.failed_event_png_path)
                marked_original_img = draw_original_element_on_image(original_img, original_bounds)
                marked_original_img.save(marked_original_path)
            else:
                marked_original_img = read_image(marked_original_path)

            marked_original_base64 = get_encoded_image(marked_original_img)

            # 2. Crop the original element image
            original_element = self._find_original_element(self.failed_event_path, self.failed_event_xml_tree)
            if original_element is None:
                self.logger.warning("Cannot find original element for cropping")
                return None
            original_bounds = self._parse_bounds(original_element.attrib.get("bounds", ""))
            crop_bounds = (original_bounds[0][0], original_bounds[0][1], original_bounds[1][0], original_bounds[1][1])
            original_full_img = read_image(self.failed_event_png_path)
            original_element_img = original_full_img.crop(crop_bounds)
            original_element_img_base64 = get_encoded_image(original_element_img)

            # 3. Generate marked screenshot for each candidate
            candidates_list = list(candidates.items())
            candidates_images = []

            for idx, (view_str, candidate_info) in enumerate(candidates_list, start=1):
                view = candidate_info['view']
                last_state = candidate_info.get('last_state')

                if last_state and last_state.screenshot_path and os.path.exists(last_state.screenshot_path):
                    screenshot_img = read_image(last_state.screenshot_path)
                else:
                    self.logger.warning(f"Candidate {idx} has no valid screenshot, skipping")
                    continue

                # Get candidate's bounds
                bounds = view.get('bounds', [[0, 0], [100, 100]])
                if isinstance(bounds, list) and len(bounds) == 2:
                    bounds_tuple = (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1])
                else:
                    bounds_tuple = bounds

                # Mark candidate position on the screenshot
                marked_img = draw_replay_element_on_image(screenshot_img, bounds_tuple, idx)
                marked_img_base64 = get_encoded_image(marked_img)

                # Save the marked image for debugging
                marked_img_path = os.path.join(
                    self.exploration_tmp_dir, "states",
                    f"candidate_{self.failed_event_number}_{idx}.png"
                )
                marked_img.save(marked_img_path)

                candidates_images.append({
                    'index': idx,
                    'view_str': view_str,
                    'image_base64': marked_img_base64,
                    'matching_method': candidate_info.get('matching_method', 'unknown'),
                    'count': candidate_info.get('count', 1),
                    'text': view.get('text') or view.get('content_description') or '',
                    'bounds': bounds,
                    'class': view.get('class', ''),
                    'resource_id': view.get('resource_id', '')
                })

            if not candidates_images:
                self.logger.warning("No valid candidate images generated")
                return None

            # 4. Construct prompt
            system_prompt, user_prompt = self._construct_select_candidate_prompt(
                marked_original_base64,
                original_element_img_base64,
                candidates_images
            )

            # 5. Call LLM
            response, token_usage = openai_chat(
                system_prompt, user_prompt,
                self.llm_api_key, "gpt-5.1", "gpt"
            )
            self.logger.info(f"LLM select candidate response: {response}")
            self.logger.info(f"Token usage: {token_usage}")

            # 6. Parse LLM response
            response_upper = response.upper()

            # Check if NONE was returned
            if "NONE" in response_upper:
                self.logger.info("LLM selected NONE - no matching candidate")
                return None

            # Try to extract CANDIDATE_N
            import re
            match = re.search(r'CANDIDATE[_\s]*(\d+)', response_upper)
            if match:
                selected_idx = int(match.group(1))
                if 1 <= selected_idx <= len(candidates_images):
                    selected_view_str = candidates_images[selected_idx - 1]['view_str']
                    selected_candidate = candidates[selected_view_str]
                    self.logger.info(f"✓ LLM selected CANDIDATE_{selected_idx}: {selected_view_str[:30]}...")
                    return (selected_view_str, selected_candidate)
                else:
                    self.logger.warning(f"LLM returned invalid index: {selected_idx}, max is {len(candidates_images)}")
                    return None
            else:
                self.logger.warning(f"LLM returned unparseable result: {response}")
                return None

        except Exception as e:
            self.logger.error(f"Error in llm_select_best_candidate: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _construct_select_candidate_prompt(self, marked_original_base64, original_element_img_base64, candidates_images):
        """
        Construct the prompt for LLM to select the best candidate.
        Format must comply with openai_chat function requirements: {'ori_analyze': [...], 'update_analyze': [...]}
        """
        system_prompt = """
You are an Android UI analysis expert with experience in UI evolution across app versions.
Your task is to select the best matching candidate on the updated app version corresponds to the original UI widget from the old version.

## Key Rules
1. Focus on FUNCTIONALITY, not just appearance
2. The selected element should be able to perform the exact same action as the original
3. Consider: text content, content description, element type, and position
4. If NO candidate matches the original element's function, answer NONE

## Output Format
Provide a brief reasoning, then output ONLY one of:
CANDIDATE_1
CANDIDATE_2
...
CANDIDATE_N
or
NONE
"""

        # ori_analyze: original element information
        ori_analyze_text = """
I will provide you with the original application version's screenshot (marked with red box indicating the target UI element) and the cropped original UI element.

* Original Screenshot (with target element marked in red box)
```
please see Figure 1.
```

* Original UI Element (cropped)
```
please see Figure 2.
```
"""
        ori_analyze = [
            marked_original_base64,
            original_element_img_base64,
            {"type": "text", "text": ori_analyze_text}
        ]

        # update_analyze: candidate element information
        candidates_text = f"""
I will provide you with {len(candidates_images)} candidate elements found during scroll exploration. Each candidate is marked with a green box and labeled with an index number.

Your task is to determine which candidate element (if any) can perform the SAME FUNCTION as the original target element.

## Candidate Elements:
"""
        for candidate in candidates_images:
            idx = candidate['index']
            candidates_text += f"""
### CANDIDATE_{idx}:
(See corresponding image below)
"""

        candidates_text += """
## Question:
Which candidate element best matches the original element's FUNCTION? Select the one that can perform the same action.
If none of the candidates match, answer NONE.

Answer format: CANDIDATE_1 / CANDIDATE_2 / ... / NONE
"""

        # Construct update_analyze: all candidate images + text description
        update_analyze = []
        for candidate in candidates_images:
            update_analyze.append(candidate['image_base64'])
        update_analyze.append({"type": "text", "text": candidates_text})

        user_prompt = {
            'ori_analyze': ori_analyze,
            'update_analyze': update_analyze
        }

        return system_prompt, user_prompt

    def _construct_judge_exploration_prompt_plain(self, marked_original_base64, original_element_img_base64, exploration_steps_images, original_next_screen_summary = None):
        """
        original_next_screen_base64: summary of the next screen
        """

        system_prompt = """
You are an Android UI testing expert. You are working on UI evolution analysis across app versions.
Your task is to determine whether the current exploration trace has already reached a screen where the user is positioned to perform the SAME FUNCTION as the original UI element in the OLD version.

Judge this only based on visible UI semantics and screen context.

Use the following simple rule:

SUCCESS if the selected UI element on the FINAL SCREEN appears to serve the same user purpose or intent as the original UI element.

If the equivalence is uncertain or ambiguous, judge it as NO.

Do NOT assume any future navigation or hidden interactions.


# ===============================
# Output Format (STRICT)
# ===============================

Provide a brief explanation (1–3 sentences), then output ONLY:

YES
or
NO
    """


        original_block = """
# ===============================
# Input Description
# ===============================

## Original Target Element
Below is the screenshot of the original UI element (red box), followed by a zoomed-in figure of that element in the OLD version.

Identify the semantic meaning of this element in the OLD version and understand what kind of user action it enables, regardless of visual form.
    """

        user_prompt = {}
        user_prompt["ori_analyze"] = [
            {"type": "text", "text": original_block},
            marked_original_base64,
            original_element_img_base64
        ]

        if original_next_screen_summary:
            original_next_screen_block = f"""
## Original Next Screen
Below is the summary of the screen reached in the OLD version immediately AFTER interacting with the original UI element.

Use this screen to infer the semantic meaning of the ORIGINAL UI element, especially when the original icon or label is abstract, by leveraging the resulting screen’s title and content.
{original_next_screen_summary}
            """
            
            user_prompt["ori_analyze"].append({"type": "text", "text": original_next_screen_block})

        steps_block = f"""
## Exploration Steps
These images show the navigation path in the NEW version. Each step highlights the UI element that was operated at that step (green box).

Total steps: {len(exploration_steps_images)}
Steps are in order from step 0 to step {len(exploration_steps_images) - 1}.
Your final judgment MUST be based ONLY on what is explicitly visible on the FINAL SCREEN and the exploration steps. The green-boxed element on the FINAL SCREEN is the candidate element to be judged.
    """

        user_prompt["update_analyze"] = [
            {"type": "text", "text": steps_block}
        ] + [step["base64"] for step in exploration_steps_images]

        question_text = """
## Final Question
Based on your analysis, has the exploration already reached a screen where the user is positioned to perform the SAME FUNCTION as the original UI element in the OLD version, without assuming any additional screens or interactions?

Answer strictly with:

YES
or
NO
    """
    # Append question to user_prompt["update_analyze"]
        user_prompt["update_analyze"] = user_prompt["update_analyze"] + [{"type": "text", "text": question_text}]

        return system_prompt, user_prompt


    def _construct_judge_exploration_prompt(self, marked_original_base64, original_element_img_base64, exploration_steps_images, original_next_screen_summary = None):

        """
        original_next_screen_base64: summary of the next screen
        """

        system_prompt = """
You are an Android UI testing expert. You are working on UI evolution analysis across app versions.
Your task is to determine whether the current exploration trace has already reached a screen where the user is positioned to perform the SAME FUNCTION as the original UI element in the OLD version.

# ===============================
# 1. Widget Type Classification
# ===============================

You MUST classify BOTH:
- the original UI element (red-boxed) in the OLD version, AND
- the selected UI element (green-boxed) on the FINAL SCREEN in the NEW version
into exactly ONE of the following categories, based on their semantic role.

### (A) ENTRY-TYPE WIDGET (Navigation Control)
Examples: “More Options” (⋮), menu items, page entries, list items, buttons whose purpose is to navigate into another screen.
Function characteristics:
- Its purpose is to OPEN another page / dialog / menu.
- It does NOT directly change a value.

### (B) TERMINAL-READONLY WIDGET
Examples: labels displaying current state, informational text, static indicators without user interaction.
Function characteristics:
- Displays information but cannot change it.

### (C) TERMINAL-EDITABLE WIDGET (Actionable Setting)
Examples: switch, checkbox, radio option, editable text field, dialog with selectable options.
Function characteristics:
- Allows directly changing a value or selecting an option.

You may use the provided "Original Next Screen" (if available) as supporting evidence to determine the original UI element's type.

# ===============================
# 2. Success Judgment Rules
# ===============================

Judge success according to the widget type of the original UI element and the selected UI element on the FINAL SCREEN and the following rules:

First, their widget types must be the SAME.

Then, judge success as follows:

- ENTRY-TYPE:
  SUCCESS if the selected UI widget allows the user to start the same functional flow as the original widget.
  The user does NOT need to have already entered the next screen; being able to navigate to the target functionality is sufficient.
  Differences in visual form (icon vs menu item), placement, or presentation style do NOT affect equivalence.

- TERMINAL-READONLY:
  SUCCESS if the same information (or its clear equivalent) is visible on the final screen.

- TERMINAL-EDITABLE:
  SUCCESS if the corresponding editable control (e.g., switch, checkbox, radio options, or option dialog) is visible and directly reachable on the final screen.
  Do NOT assume that a configurable option is editable unless an explicit control (e.g., switch, checkbox, radio buttons, or a visible option dialog) is present on the screen.
  A text row or list item that requires another tap to open a sub-screen or dialog is NOT considered editable.


# ===============================
# Output Format (STRICT)
# ===============================

Provide a brief explanation (1–3 sentences), then output ONLY:

YES
or
NO
    """


        original_block = """
# ===============================
# Input Description
# ===============================

## Original Target Element
Below is the screenshot of the original UI element (red box), followed by a zoomed-in figure of that element in the OLD version.

Identify the semantic meaning of this element in the OLD version and understand what kind of user action it enables, regardless of visual form.
    """

        user_prompt = {}
        user_prompt["ori_analyze"] = [
            {"type": "text", "text": original_block},
            marked_original_base64,
            original_element_img_base64
        ]

        if original_next_screen_summary:
            original_next_screen_block = f"""
## Original Next Screen
Below is the summary of the screen reached in the OLD version immediately AFTER interacting with the original UI element.

Use this screen to infer the semantic meaning of the ORIGINAL UI element, especially when the original icon or label is abstract, by leveraging the resulting screen’s title and content.
{original_next_screen_summary}
            """
            
            user_prompt["ori_analyze"].append({"type": "text", "text": original_next_screen_block})

        steps_block = f"""
## Exploration Steps
These images show the navigation path in the NEW version. Each step highlights the UI element that was operated at that step (green box).

Total steps: {len(exploration_steps_images)}
Steps are in order from step 0 to step {len(exploration_steps_images) - 1}.
Your final judgment MUST be based ONLY on what is explicitly visible on the FINAL SCREEN and the exploration steps. The green-boxed element on the FINAL SCREEN is the candidate element to be judged.
    """

        user_prompt["update_analyze"] = [
            {"type": "text", "text": steps_block}
        ] + [step["base64"] for step in exploration_steps_images]

        question_text = """
## Final Question
Based on your analysis, has the exploration already reached a screen where the user is positioned to perform the SAME FUNCTION as the original UI element in the OLD version, without assuming any additional screens or interactions?

Answer strictly with:

YES
or
NO
    """
    # Append question to user_prompt["update_analyze"]
        user_prompt["update_analyze"] = user_prompt["update_analyze"] + [{"type": "text", "text": question_text}]

        return system_prompt, user_prompt


    def _create_repaired_event(self, matched_view):
        """
        Create the repaired event.

        Args:
            matched_view: The matched view dictionary

        Returns:
            TouchEvent: The repaired touch event
        """
        # Record this matched view for subsequent feedback mechanism
        self.last_repaired_view = matched_view

        repaired_event = TouchEvent(view=matched_view)
        # repaired_event.u2 = self.device.u2
        self.mode = "replay"  # Switch back to replay mode
        return repaired_event

    def filter_scroll_events(self, scrollable_events):
        """
        Filter usable scroll_events from all scrollable_events.

        Processing logic:
        1. Collect unique scrollable views (deduplicate by view_str)
        2. Determine scroll direction based on view's aspect ratio
        3. For overlapping scrollable views:
          - If directions are the same and the intersection area exceeds 80% of the smaller view, the smaller one can be removed
          - If directions are different, keep both

        Args:
            scrollable_events: Original list of scrollable events

        Returns:
            Filtered list of scroll_events
        """
        def bounds_overlap(bounds1, bounds2, overlap_threshold=0.8):
            """Detect whether two rectangles significantly overlap (80% threshold)"""
            x1_min, y1_min = bounds1[0]
            x1_max, y1_max = bounds1[1]
            x2_min, y2_min = bounds2[0]
            x2_max, y2_max = bounds2[1]

            # First check if there is any overlap
            if x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min:
                return False

            # Calculate overlap area
            overlap_x_min = max(x1_min, x2_min)
            overlap_y_min = max(y1_min, y2_min)
            overlap_x_max = min(x1_max, x2_max)
            overlap_y_max = min(y1_max, y2_max)
            overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)

            # Calculate the area of both rectangles
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            smaller_area = min(area1, area2)

            # Only count as overlap if the overlap area exceeds 80% of the smaller rectangle
            if smaller_area > 0 and overlap_area / smaller_area >= overlap_threshold:
                return True
            return False

        def get_scroll_direction(view, events):
            """
            Determine scroll direction based on view's class.

            - HorizontalScrollView → only try 'right'
            - Others → prefer 'down', then try 'right'

            Returns:
                (direction, event): Scroll direction and corresponding event, returns (None, None) if no matching direction event
            """
            view_class = view.get('class', '') if view else ''

            if 'HorizontalScrollView' in view_class:
                directions = ['right']
            else:
                directions = ['down', 'right']

            for direction in directions:
                if direction in events:
                    self.logger.info(f"View class={view_class}, using direction='{direction}'")
                    return direction, events[direction]

            return None, None

        # 1. Collect all unique scrollable view objects (deduplicate by view_str)
        scrollable_views = {}  # {view_str: {'view': view, 'events': {direction: event}}}
        for event in scrollable_events:
            view = event.view
            if view:
                view_str = view.get('view_str', str(view.get('bounds', '')))
                if view_str not in scrollable_views:
                    scrollable_views[view_str] = {'view': view, 'events': {}}
                scrollable_views[view_str]['events'][event.direction] = event

        self.logger.info(f"Found {len(scrollable_views)} unique scrollable views")

        # 2. Test actual scroll direction for each view, collect valid scroll events
        # [(view_str, view, direction, event, bounds, area)]
        valid_scroll_views = []
        for view_str, view_data in scrollable_views.items():
            view = view_data['view']
            events = view_data['events']

            # Determine scroll direction based on aspect ratio
            direction, scroll_event = get_scroll_direction(view, events)

            if direction is None:
                self.logger.info(f"Skipping view {view_str}: no valid scroll direction")
                continue

            if 'bounds' in view:
                bounds = view['bounds']
                view_width = bounds[1][0] - bounds[0][0]
                view_height = bounds[1][1] - bounds[0][1]
                view_area = view_width * view_height
                valid_scroll_views.append((view_str, view, direction, scroll_event, bounds, view_area))
                self.logger.info(f"Valid scroll view: direction={direction}, size={view_width}x{view_height}")
            else:
                valid_scroll_views.append((view_str, view, direction, scroll_event, None, 0))

        # 3. Remove overlapping views (same direction: keep larger area; different direction: keep both)
        unique_views = []
        for item in valid_scroll_views:
            view_str, view, direction, scroll_event, bounds, area = item
            if bounds is None:
                unique_views.append(item)
                continue

            should_add = True
            for i, existing_item in enumerate(unique_views):
                existing_bounds = existing_item[4]
                existing_direction = existing_item[2]

                if existing_bounds and bounds_overlap(bounds, existing_bounds):
                    # Different directions, keep both
                    if direction != existing_direction:
                        self.logger.info(f"Keeping both overlapping views: different scroll direction ({direction} vs {existing_direction})")
                        continue

                    # Same direction, keep the one with larger area
                    existing_area = existing_item[5]
                    if area > existing_area:
                        self.logger.info(f"Replacing overlapping view: new area {area} > existing {existing_area}")
                        unique_views[i] = item
                    else:
                        self.logger.info(f"Skipping overlapping view: area {area} <= existing {existing_area}")
                    should_add = False
                    break

            if should_add:
                unique_views.append(item)

        # 4. Return filtered scroll events
        scroll_events = []
        for view_str, view, direction, scroll_event, bounds, area in unique_views:
            scroll_events.append(scroll_event)
            self.logger.info(f"Added scroll {direction} event for view")

        return scroll_events

    def find_target_element_in_page(self, current_state, step, cross_page=False, is_has_next_screen_summary=True):
        """
        Find the target element in the current page using the UIMatch algorithm.

        Args:
            current_state: Current device state

        Returns:
            (matched_view, matching_method): Successfully matched view dictionary and matching method, or (None, None)
        """


        try:
            # Use the already loaded failed event data
            if self.failed_event_json is None or self.failed_event_xml_tree is None:
                self.logger.error("Failed event data not loaded")
                return None, None

            # Find the target element in the failed event's XML
            original_element = self._find_original_element(self.failed_event_path, self.failed_event_xml_tree)

            if original_element is None:
                self.logger.warning("Target element not found in original XML")
                return None, None

            # Save current state's XML and screenshot to the exploration_tmp directory
            current_state.tag = f"same_page_{step}"
            state_dir = os.path.join(self.exploration_tmp_dir, "states")
            current_state.save2dir(state_dir)
            current_xml_path = os.path.join(self.exploration_tmp_dir, f"xmls/xml_same_page_{step}.xml")
            current_png_path = os.path.join(self.exploration_tmp_dir, f"states/screen_same_page_{step}.png")

            # Parse current XML
            with open(current_xml_path, 'r', encoding='utf-8') as f:
                current_xml_tree = ET.parse(f)

            matcher = Matcher(
                original_png=self.failed_event_png_path,
                original_tree=self.failed_event_xml_tree,
                original_element=original_element,
                replay_png=current_png_path,
                replay_tree=current_xml_tree,
                logger=self.logger,
                cross_page=cross_page
            )

            # 1. Prepare the original element marked image
            marked_original_path = self.failed_event_png_path.replace(".png", "_marked_original_element.png")
            if not os.path.exists(marked_original_path):
                original_element = self._find_original_element(self.failed_event_path, self.failed_event_xml_tree)
                if original_element is None:
                    self.logger.warning("Cannot find original element for judge")
                    return False
                original_bounds = self._parse_bounds(original_element.attrib.get("bounds", ""))
                original_img = read_image(self.failed_event_png_path)
                marked_original_img = draw_original_element_on_image(original_img, original_bounds)
                marked_original_img.save(marked_original_path)
            else:
                marked_original_img = read_image(marked_original_path)

            marked_original_base64 = get_encoded_image(marked_original_img)

            # 2. Crop the original element image
            original_element = self._find_original_element(self.failed_event_path, self.failed_event_xml_tree)
            if original_element is None:
                self.logger.warning("Cannot find original element for cropping")
                return False
            original_bounds = self._parse_bounds(original_element.attrib.get("bounds", ""))
            # Convert to PIL crop format (x1, y1, x2, y2)
            crop_bounds = (original_bounds[0][0], original_bounds[0][1], original_bounds[1][0], original_bounds[1][1])
            original_full_img = read_image(self.failed_event_png_path)
            original_element_img = original_full_img.crop(crop_bounds)
            original_element_img_base64 = get_encoded_image(original_element_img)



            # v2: Have LLM summarize the changes from current screen to next screen, then pass the description to LLM to recommend the next element
            if is_has_next_screen_summary == False:
                original_next_screen_summary = None
            else:
                if self.failed_event_png_next and self.original_next_screen_summary is None:
                    next_screen_base64 = get_encoded_image(self.failed_event_png_next)
                    original_next_screen_summary = self.get_next_current_description(marked_original_base64, original_element_img_base64, next_screen_base64)
                    self.original_next_screen_summary = original_next_screen_summary
                elif self.original_next_screen_summary is not None:
                    original_next_screen_summary = self.original_next_screen_summary
                else:
                    original_next_screen_summary = None

            

            # Execute matching
            matching_result = matcher.matching(app_name=self.app.get_package_name(), without_llm=self.without_llm, without_rule=self.without_rule, original_next_screen_summary=original_next_screen_summary)

            if matching_result.get("success"):
                matched_element = matching_result.get("matched_element")
                matching_method = matching_result.get('matching_method', 'unknown')
                self.logger.info(f"✓ Found target element using {matching_method} matching")

                # using the matched element to replace the event
                for current_view in current_state.views:
                    # normalize
                    resource_id = self.normalize(current_view['resource_id'])
                    text = self.normalize(current_view['text'])  # May change
                    content_description = self.normalize(current_view['content_description'])
                    class_name = self.normalize(current_view['class'])
                    bounds = current_view['bounds']

                    if self.check_if_same(resource_id, matched_element.get('resource-id')) and \
                    self.check_if_same(content_description, matched_element.get('content-desc')) and \
                    self.check_if_same(class_name, matched_element.get('class')) and \
                    self.compare_bounds(bounds, matched_element.get('bounds')):

                        # Check if in exclusion list
                        if self._is_view_excluded(current_view):
                            self.logger.info(f"Skipping excluded view: {current_view.get('view_str', 'unknown')}")
                            continue

                        return current_view, matching_method

                # If no matching view found, return None
                self.logger.warning("Matched element not found in current_state.views")
                return None, None
            else:
                self.logger.info("Target element not found in current page")
                return None, None

        except Exception as e:
            self.logger.error(f"Error in find_target_element_in_page: {e}")
            import traceback
            traceback.print_exc()
            return None, None


    def _find_original_element(self, event_path, xml_tree) -> ET.Element:
        """
        Find the corresponding element in xml_tree based on information in the event.

        Args:
            event_path: Event file path
            xml_tree: XML tree object

        Returns:
            The found element object or None
        """
        import json
        
        # 1. Extract bounds and class information from the event file
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
        
        # 2. Find the node with the same attributes in the XML tree
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

    

    def _is_view_excluded(self, view):
        """
        Check if a view is in the exclusion list.

        Args:
            view: The view dictionary to check

        Returns:
            True if excluded, False otherwise
        """
        if not self.excluded_views:
            return False

        view_str = view.get('view_str', '')
        view_bounds = view.get('bounds', [])

        for excluded in self.excluded_views:
            # Compare by view_str (unique identifier)
            if excluded.get('view_str') == view_str:
                return True
            # Or compare by bounds
            if excluded.get('bounds') == view_bounds:
                return True

        return False

    def _get_view_navigation_id(self, view, activity=None, click_type=None):
        """
        Generate a unique navigation identifier for the view, used to determine if it has been visited.

        Args:
            view: view dictionary
            activity: Current activity name

        Returns:
            (activity, resource_id, class, content_desc) tuple
        """
        resource_id = view.get('resource_id', '') or ''
        class_name = view.get('class', '') or ''
        content_desc = view.get('content_description', '') or ''
        activity = activity or ''
        click_type = click_type or ''
        return (activity, resource_id, class_name, content_desc, click_type)


    def _filter_events_by_rules(self, events):
        """
        Filter overlapping events. When multiple events' center points are covered by the same clickable element,
        only keep the topmost one (highest drawing order).

        Algorithm logic (reference: _HindenWidgetFilter):
        1. Traverse events in ascending drawing order
        2. Use rtree index to record each event's center point
        3. When encountering a clickable event, check if its bounds contain previous events' center points
        4. If so, mark previous events as covered (occluded)
        5. Finally return only the non-occluded events

        Args:
            events: Original event list

        Returns:
            filtered_events: Filtered event list
        """
        import rtree

        self.logger.info(f"Filtering events: original count = {len(events)}")

        # If the event list is empty, return directly
        if not events:
            return events

        try:
            # Sort events by drawing order
            def get_drawing_order(event):
                if hasattr(event, 'view') and event.view:
                    return event.view.get('drawing-order', 0) or 0
                return 0

            sorted_events = sorted(events, key=get_drawing_order)

            # Use rtree index
            idx = rtree.index.Index()
            event_nodes = []  # Store events and their covered status
            covered_set = set()  # Indices of occluded events

            for event in sorted_events:
                # Non-view events are added directly
                if not hasattr(event, 'view') or not event.view:
                    event_nodes.append(event)
                    continue

                bounds = event.view.get('bounds')
                if not bounds:
                    event_nodes.append(event)
                    continue

                # Parse bounds: [[x1, y1], [x2, y2]]
                x1, y1 = bounds[0]
                x2, y2 = bounds[1]

                # Check if the current event is clickable
                clickable = event.view.get('clickable', False)

                if clickable:
                    # Find previous events covered by the current event (center points within current bounds)
                    covered_ids = list(idx.intersection((x1, y1, x2, y2)))
                    for covered_id in covered_ids:
                        # Check if the covered event comes from the same view (same bounds)
                        # If it's different event types from the same view, they should not be filtered
                        covered_event = event_nodes[covered_id]
                        if hasattr(covered_event, 'view') and covered_event.view:
                            cb = covered_event.view.get('bounds')
                            if cb:
                                # If bounds are the same, it's the same view, skip
                                if cb == bounds:
                                    continue
                                # Different view, mark as covered
                                covered_set.add(covered_id)
                                # Remove the covered event from the index
                                cx = (cb[0][0] + cb[1][0]) / 2
                                cy = (cb[0][1] + cb[1][1]) / 2
                                idx.delete(covered_id, (cx, cy, cx, cy))

                # Calculate the current event's center point and insert into index
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                cur_id = len(event_nodes)
                idx.insert(cur_id, (center_x, center_y, center_x, center_y))
                event_nodes.append(event)

            # Filter out occluded events
            filtered_events = []
            for i, event in enumerate(event_nodes):
                if i not in covered_set:
                    filtered_events.append(event)

            self.logger.info(f"Filter stats: covered={len(covered_set)}, kept={len(filtered_events)}")
            self.logger.info(f"Events after filtering: {len(filtered_events)}")
            return filtered_events

        except Exception as e:
            self.logger.error(f"Error in _filter_events_by_uimatch_rules: {e}")
            import traceback
            traceback.print_exc()
            # Return original event list on error
            return events

    def _find_element_by_bounds(self, xml_tree, bounds_str):
        """
        Find the element with specified bounds in the XML tree.

        Args:
            xml_tree: XML tree
            bounds_str: Bounds string in format [x1,y1][x2,y2]

        Returns:
            The found element or None
        """
        for element in xml_tree.iter():
            if element.get("bounds") == bounds_str:
                return element
        return None

    
    def _parse_bounds(self, bounds_str):
        """
        Parse bounds string into coordinate list.

        Args:
            bounds_str: String in "[x1,y1][x2,y2]" format

        Returns:
            List in [[x1, y1], [x2, y2]] format
        """
        import re
        match = re.findall(r'\[(\d+),(\d+)\]', bounds_str)
        if len(match) == 2:
            return [[int(match[0][0]), int(match[0][1])],
                    [int(match[1][0]), int(match[1][1])]]
        return [[0, 0], [0, 0]]

    def save_repair_trace(self):
        """
        Save the complete repair trace to file.
        """
        try:
            repair_log_dir = os.path.join(self.exploration_tmp_dir, "repair_logs")
            os.makedirs(repair_log_dir, exist_ok=True)

            repair_log_path = os.path.join(
                repair_log_dir,
                f"repair_trace_event_{self.failed_event_number}.json"
            )

            # Determine if repair succeeded (last step's found_target is True)
            repair_success = False
            if self.repair_trace and self.repair_trace[-1].get('found_target'):
                repair_success = True

            # Complete repair trace
            repair_log = {
                'failed_event_number': self.failed_event_number,
                'repair_success': repair_success,
                'total_steps': len(self.repair_trace),
                'timestamp': str(__import__('datetime').datetime.now()),
                'trace': self.repair_trace  # Save the complete trace list
            }

            # Save to JSON file
            with open(repair_log_path, 'w', encoding='utf-8') as f:
                json.dump(repair_log, f, indent=2, ensure_ascii=False)

            self.logger.info(f"✓ Repair trace saved to: {repair_log_path}")

        except Exception as e:
            self.logger.error(f"Error saving repair trace: {e}")
            import traceback
            traceback.print_exc()

    def check_if_same(self, current, record):
        if current is None or record is None:
            return False
        if current == record:
            return True
        return False

    def replace_view(self, event, current_view):
        event.view['resource_id'] = current_view['resource_id']
        event.view['text'] = current_view['text']
        event.view['content_description'] = current_view['content_description']
        event.view['class'] = current_view['class']
        event.view['instance'] = current_view['instance']
        event.view['bounds'] = current_view['bounds']

    def check_which_exists(self, event):
        resource_id = MatchingPolicy.__safe_dict_get(event.view, 'resource_id')
        text = MatchingPolicy.__safe_dict_get(event.view, 'text')
        content_description = MatchingPolicy.__safe_dict_get(event.view, 'content_description')
        class_name = MatchingPolicy.__safe_dict_get(event.view, 'class')
        instance = MatchingPolicy.__safe_dict_get(event.view, 'instance')

        u2 = self.device.u2
        

        if content_description is not None:
            if u2.exists(description=content_description, instance=instance):
                for current_view in self.current_state.views:
                    if self.check_if_same(current_view['content_description'], content_description) and self.check_if_same(current_view['instance'], instance):
                        self.replace_view(event, current_view)
                        break
                return 'content_description', content_description
        elif text is not None:
            if u2.exists(text=text, instance=instance):
                for current_view in self.current_state.views:
                    if self.check_if_same(current_view['text'], text) and self.check_if_same(current_view['instance'], instance):
                        self.replace_view(event, current_view)
                        break
                return 'text', text
        elif resource_id is not None:
            if u2.exists(resourceId=resource_id, instance=instance):
                for current_view in self.current_state.views:
                    if self.check_if_same(current_view['resource_id'], resource_id) and self.check_if_same(current_view['instance'], instance):
                        self.replace_view(event, current_view)
                        break
                return 'resource_id', resource_id
        elif class_name is not None:
            if u2.exists(className=class_name, instance=instance):
                for current_view in self.current_state.views:
                    if self.check_if_same(current_view['class'], class_name) and self.check_if_same(current_view['instance'], instance):
                        self.replace_view(event, current_view)
                        break
                return 'class_name', class_name
        elif class_name is not None and resource_id is not None and instance is not None:
            if u2.exists(className=class_name, resourceId=resource_id, instance=instance):
                for current_view in self.current_state.views:
                    if self.check_if_same(current_view['class'], class_name) and self.check_if_same(current_view['resource_id'], resource_id) and self.check_if_same(current_view['instance'], instance):
                        self.replace_view(event, current_view)
                        break
                return 'class_resource_instance', (class_name, resource_id, instance)
        
        return None, None
    

    @staticmethod
    def __safe_dict_get(view_dict, key, default=None):
        value = view_dict[key] if key in view_dict else None
        return value if value is not None else default


class GuiderPolicy(InputPolicy):
    """
    Replay DroidBot output generated by Guider policy

    One Baseline Policy

    find the target element
    """

    def __init__(self, device, app, replay_output, failed_replay_output, output_dir):
        super(GuiderPolicy, self).__init__(device, app)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.replay_output = replay_output # Normal record output
        self.failed_replay_output = failed_replay_output # Failed replay output
        self.output_dir = output_dir # Output directory

        event_dir = os.path.join(replay_output, "events")
        files = [os.path.join(event_dir, x) for x in
                 next(os.walk(event_dir))[2]
                 if x.endswith(".json")]
        def _event_index(path):
            base = os.path.basename(path)
            name, _ = os.path.splitext(base)
            try:
                return int(name.split('_')[-1])
            except Exception:
                return float('inf')
        # Natural sort: sort by <num> in event_<num>.json in ascending order
        self.event_paths = sorted(files, key=_event_index)
        # skip HOME and start app intent
        self.device = device
        self.app = app
        self.event_idx = 1
        self.num_replay_tries = 0
        self.utg = UTG(device=device, app=app, random_input=None)
        self.last_event = None
        self.last_state = None
        self.current_state = None

        # Failed event related
        self.failed_event_number = 0
        self.failed_event_path = None
        self.failed_event_json = None
        self.failed_event_xml_tree = None
        self.failed_event_png_path = None
        self.failed_event_png = None
        self.failed_event_png_next = None
        self.failed_event_png_next_path = None
        self.load_failed_event() # Load failed_event_number, failed_event_json, failed_event_xml_tree, failed_event_png

        # Mode management, first replay mode, then repair mode
        self.mode = "replay"  # Two modes: "replay" (replay mode) and "repair" (repair mode)

        # Repair process tracking
        self.repair_trace = []  # Complete trace of repair process, each step includes: {step_number, matched_element, screenshot}
        self.try_count = 0  # Number of attempts to find target element
        self.exploration_step = 0  # Exploration step counter, globally managed
        # Exploration state management (for random exploration fallback)
        self.visited_states = set()  # Visited state hashes, to avoid repeated exploration

        # Feedback mechanism: record incorrect match results, exclude them during retry
        self.excluded_views = []  # Excluded views (previously matched successfully but failed later)
        self.last_repaired_view = None  # View matched in the last repair attempt
        self.exploration_retry_count = 0  # Exploration retry count
        self.max_exploration_retries = 3  # Maximum retry count

        # Activity path tracking (for determining back feasibility)
        self.activity_trace = []  # Record activity changes after each event execution
        self.exploration_activity_trace = []  # Record activity changes during exploration

        self.original_next_screen_summary = None # Summary of the original next screen

        # Temporary file storage directory
        self.exploration_tmp_dir = os.path.join(self.output_dir, "exploration_tmp/")
        os.makedirs(self.exploration_tmp_dir, exist_ok=True)

        # Prevent DFS infinite loops by recording visited navigation elements
        self.visited_navigation_elements = set()  # Record previously selected navigation elements (activity, resource_id, class, content_desc, click_type)

        # Configure logger output to file
        self._setup_file_logger()

        # LLM configuration
        self.llm_api_key = os.getenv("API_KEY")  # OpenAI API Key


    def _setup_file_logger(self):
        """Configure logger output to a file in the exploration_tmp directory"""
        log_file = os.path.join(self.exploration_tmp_dir, "repair.log")

        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Set format
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        # Add to logger
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)

        self.logger.info(f"Logger initialized, log file: {log_file}")

    def load_failed_event(self):

        # 1. Load failed_event_number
        all_event_files = glob.glob(os.path.join(self.failed_replay_output, "events/*.json"))
        event_numbers = [int(os.path.basename(file).split('_')[-1].split('.')[0]) for file in all_event_files]
        self.failed_event_number = max(event_numbers) + 1

        # 2. Load failed_event_json
        failed_event_json_path = os.path.join(self.replay_output, f"events/event_{self.failed_event_number}.json")
        self.failed_event_path = failed_event_json_path
        with open(failed_event_json_path, 'r') as f:
            self.failed_event_json = json.load(f)

        # 3. Load failed_event_xml_tree
        failed_event_xml_tree_path = os.path.join(self.replay_output, f"xmls/xml_{self.failed_event_number-1}.xml")
        with open(failed_event_xml_tree_path, 'r') as f:
            self.failed_event_xml_tree = ET.parse(f)

        # 4. Load failed_event_png
        failed_event_png_path = os.path.join(self.replay_output, f"states/screen_{self.failed_event_number-1}.png")
        self.failed_event_png_path = failed_event_png_path
        self.failed_event_png = read_image(failed_event_png_path) # PIL.Image.Image

        # 5. Load failed_event_png_next
        failed_event_png_next_path = os.path.join(self.replay_output, f"states/screen_{self.failed_event_number}.png")
        self.failed_event_png_next_path = failed_event_png_next_path
        self.failed_event_png_next = read_image(failed_event_png_next_path) # PIL.Image.Image
    
   
    def normalize(self, value):
        if value is None:
            return ""
        else:
            return value

    def compare_bounds(self, current_bounds, gt_bounds):
        str_cur_bounds = '['+str(current_bounds[0][0])+","+str(current_bounds[0][1])+"]["+str(current_bounds[1][0])+","+str(current_bounds[1][1])+']'
        return str_cur_bounds == str(gt_bounds)


    def generate_event(self):
        """
        generate an event based on replay_output
        @return: InputEvent
        """
        while self.event_idx < len(self.event_paths) and \
              self.num_replay_tries < MAX_REPLY_TRIES:
            self.num_replay_tries += 1
            current_state = self.device.get_current_state()
            if current_state is None:
                time.sleep(5)
                self.num_replay_tries = 0
                return KeyEvent(name="BACK")

            curr_event_idx = self.event_idx
            # self.__update_utg()
            self.current_state = current_state
            self.current_state.tag = str(curr_event_idx) # Named by events count for easy reference later
            self.current_state.save2dir()

            # Update the activity_after of the previous event (replay phase)
            if len(self.activity_trace) > 0 and self.activity_trace[-1]['activity_after'] is None:
                self.activity_trace[-1]['activity_after'] = current_state.foreground_activity

            if curr_event_idx < len(self.event_paths):
                event_path = self.event_paths[curr_event_idx]
                with open(event_path, "r") as f:
                    curr_event_idx += 1

                    time.sleep(1) # Wait 1 second for app to load

                    self.logger.info("debug curr_event_idx: " + str(curr_event_idx))

                    if curr_event_idx!= 2:
                        try:
                            event_dict = json.load(f)
                        except Exception as e:
                            self.logger.info("Loading %s failed" % event_path + "curr_event_idx: " + str(curr_event_idx))
                            continue

                    self.logger.info("Replaying %s" % event_path + "curr_event_idx: " + str(curr_event_idx))
                    self.event_idx = curr_event_idx
                    self.num_replay_tries = 0
                    
                    # Skip the 2nd event, directly return the app launch Intent
                    if curr_event_idx == 2: # Some second events are empty
                        return IntentEvent(self.app.get_start_intent())

                    if self.app.get_package_name() == "com.appmindlab.nano" and curr_event_idx == 3:
                        # This app has an issue, restart it once more after restarting
                        self.device.adb.shell("am force-stop %s" % self.app.get_package_name())
                        self.device.start_app(self.app)
                        time.sleep(2)

                    
                    
                    event = InputEvent.from_dict(event_dict["event"])
                    event.u2 = self.device.u2
                    if isinstance(event, IntentEvent):
                        return event
                    elif isinstance(event, KeyEvent):
                        return event

                    # If reaching the failed event, switch to exploration mode
                    if curr_event_idx == self.failed_event_number:
                        self.logger.info("Reached failed event, switching to exploration mode")
                        self.mode = "explore"
                        self.target_event = event
                        time.sleep(1)
                        return self.start_exploration()
                    check_result = self.check_which_exists(event)
                    print("debug check_result", check_result)
                    if check_result[0] is None:
                        self.logger.warning(f"Widget not found for event: {event_path}")

                        # Check if the repaired event failed (indicating previous match was wrong, need to retry)
                        if curr_event_idx == self.failed_event_number + 1:
                            self.logger.info("Repaired event failed! Adding to excluded list and retrying...")
                            # Add incorrect view to exclusion list
                            if self.last_repaired_view:
                                self.excluded_views.append(self.last_repaired_view)
                            self.last_repaired_view = None
                            self.exploration_retry_count += 1
                            self.logger.info(f"Retry {self.exploration_retry_count}/{self.max_exploration_retries}, excluded {len(self.excluded_views)} views")

                            # Check if there are retry chances left
                            # if self.exploration_retry_count < self.max_exploration_retries:
                            #     # Restart app and replay to before failed_event
                            #     self.logger.info("Restarting app and replaying to retry exploration...")

                            #     # Reset event_idx to failed_event
                            #     self.event_idx = self.failed_event_number

                            #     # Clear activity trace and start over
                            #     self.activity_trace = []
                            #     self.exploration_activity_trace = []

                            #     # Restart exploration (will first replay to failed_event)
                            #     return self._restart_and_replay_to_failed_event()
                            # else:
                            #     self.logger.warning(f"Max retries ({self.max_exploration_retries}) reached, giving up")

                        self.logger.info("Stopping replay due to widget not found")
                        self.current_state.tag = str(curr_event_idx) # Named by events count for easy reference later
                        self.current_state.save2dir() # save the current state
                        self.input_manager.enabled = False
                        self.input_manager.stop()
                        break

                    if curr_event_idx == self.failed_event_number + 2:
                        # Fixed successfully, stop directly, no need to continue replay
                        self.logger.info("Repaired event succeeded! Stopping replay...")
                        self.input_manager.enabled = False
                        self.input_manager.stop()
                        break



                    self.last_state = self.current_state
                    self.last_event = event

                    # Record activity changes (for later back decision)
                    activity_before = self.current_state.foreground_activity
                    self.activity_trace.append({
                        'event_idx': curr_event_idx,
                        'activity_before': activity_before,
                        'activity_after': None  # Updated after execution
                    })

                    return event

            time.sleep(5)

    def start_exploration(self, max_steps=15):
        """
        Guider-style exploration mode: brute-force search.

        Args:
            max_steps: Maximum exploration steps

        Flow:
        Step 0 (Same page): Try to match the target element on the current page
        Step 1+ (Cross page): Click each clickable element in turn -> search for target -> press BACK if not found -> try next element
        """

        # Record activity at start of exploration, for later return navigation and retry
        start_activity = self.current_state.foreground_activity
        self.start_activity_before_repair = start_activity  # Save as instance variable for retry
        self.logger.info(f"[Guider] Exploration starting from activity: {start_activity}")

        # Clear activity trace for exploration phase
        self.exploration_activity_trace = []

        # Reset exploration step counter
        self.exploration_step = 0

        # ========== Step 0: Same page matching ==========
        self.logger.info(f"=== [Guider] Step 0: Same page matching ===")

        # Find target element on current page (Guider does not use LLM)
        matched_view, matching_method = self.find_target_element_in_page(
            self.current_state, self.exploration_step, cross_page=False,
            is_has_next_screen_summary=False, is_guider=True
        )
        self.logger.info(f"[Guider] Same page match result: {matched_view is not None}, method: {matching_method}")

        if matched_view:
            self.logger.info("[Guider] ✓ Found target element on same page")

            self.repair_trace.append({
                'step': self.exploration_step,
                'action': 'match_found',
                'event': None,
                'state_after': self.current_state.state_str,
                'screenshot': os.path.join(self.exploration_tmp_dir, f"states/screen_same_page_{self.exploration_step}.png"),
                'xml': os.path.join(self.exploration_tmp_dir, f"xmls/xml_same_page_{self.exploration_step}.xml"),
                'found_target': True,
                'matched_view': matched_view,
                'matching_method': matching_method
            })

            # Generate marked image for matched_view
            matched_bounds = matched_view.get('bounds')
            if matched_bounds:
                current_screenshot_path = os.path.join(self.exploration_tmp_dir, f"states/screen_same_page_{self.exploration_step}.png")
                if os.path.exists(current_screenshot_path):
                    current_img = read_image(current_screenshot_path)
                    if isinstance(matched_bounds, list) and len(matched_bounds) == 2:
                        bounds_str = f"[{matched_bounds[0][0]},{matched_bounds[0][1]}][{matched_bounds[1][0]},{matched_bounds[1][1]}]"
                    else:
                        bounds_str = str(matched_bounds)
                    marked_img = draw_replay_element_on_image(current_img.copy(), bounds_str, id="1")
                    images_dir = os.path.join(self.exploration_tmp_dir, "images")
                    if not os.path.exists(images_dir):
                        os.makedirs(images_dir)
                    marked_img_path = os.path.join(images_dir, f"matched_view_step_{self.exploration_step}.png")
                    marked_img.save(marked_img_path)
                    self.logger.info(f"[Guider] Saved matched_view image to: {marked_img_path}")

            repaired_event = self._create_repaired_event(matched_view)
            self.last_repaired_view = matched_view

            repaired_event_info = {
                'tag': f"repaired_event_step_{self.exploration_step}",
                'event': {
                    'event_type': repaired_event.event_type if hasattr(repaired_event, 'event_type') else 'touch',
                    'log_lines': None,
                    'x': None,
                    'y': None,
                    'view': matched_view
                },
                'start_state': self.current_state.state_str if self.current_state else None,
                'stop_state': None,
                'event_str': repaired_event.get_event_str(self.current_state) if hasattr(repaired_event, 'get_event_str') else str(repaired_event)
            }
            event_dir = os.path.join(self.exploration_tmp_dir, "events")
            if not os.path.exists(event_dir):
                os.makedirs(event_dir)
            repaired_event_path = os.path.join(event_dir, f"event_repaired_step_{self.exploration_step}.json")
            with open(repaired_event_path, 'w') as f:
                json.dump(repaired_event_info, f, indent=2, ensure_ascii=False)
            self.logger.info(f"[Guider] Saved repaired event to: {repaired_event_path}")

            self.save_repair_trace()
            return repaired_event

        # ========== Step 1+: Cross page brute-force exploration ==========
        self.logger.info(f"=== [Guider] Cross page brute-force exploration ===")

        # Get all clickable elements on the current page
        possible_events = self.current_state.get_possible_input_only_leaf_nodes(self.app.get_package_name())
        if len(possible_events) == 0:
            possible_events = self.current_state.get_possible_input(package_name=self.app.get_package_name())

        self.logger.info(f"[Guider] Found {len(possible_events)} clickable elements for exploration")

        if len(possible_events) == 0:
            self.logger.warning("[Guider] No clickable elements found, exploration failed")
            self.save_repair_trace()
            return None

        # Deduplicate: get unique views
        unique_views = []
        view_to_event = {}  # view_index -> event

        for event in possible_events:
            if hasattr(event, 'view') and event.view:
                view = event.view
                bounds = view.get('bounds', [])
                view_class = view.get('class', '')
                text = view.get('text', '')
                resource_id = view.get('resource_id', '')
                content_desc = view.get('content_description', '')

                # Check if a view with the same attributes already exists
                existing_idx = None
                for idx, existing_view in enumerate(unique_views):
                    existing_bounds = existing_view.get('bounds', [])
                    existing_class = existing_view.get('class', '')
                    existing_text = existing_view.get('text', '')
                    existing_resource_id = existing_view.get('resource_id', '')
                    existing_content_desc = existing_view.get('content_description', '')

                    if (bounds == existing_bounds and
                        view_class == existing_class and
                        text == existing_text and
                        resource_id == existing_resource_id and
                        content_desc == existing_content_desc):
                        existing_idx = idx
                        break

                if existing_idx is None:
                    new_idx = len(unique_views)
                    unique_views.append(view)
                    view_to_event[new_idx] = event

        self.logger.info(f"[Guider] Extracted {len(unique_views)} unique views for brute-force exploration")

        # Try each clickable element in turn
        for idx, view in enumerate(unique_views):
            if self.exploration_step >= max_steps:
                self.logger.warning(f"[Guider] Reached max steps {max_steps}, stopping exploration")
                break

            self.exploration_step += 1
            self.logger.info(f"=== [Guider] Step {self.exploration_step}: Trying element {idx+1}/{len(unique_views)} ===")

            event = view_to_event.get(idx)
            if not event:
                self.logger.warning(f"[Guider] No event found for view index {idx}, skipping")
                continue

            view_desc = f"{view.get('class', '')} | {view.get('resource_id', '')} | {view.get('text', '')[:30] if view.get('text') else ''}"
            self.logger.info(f"[Guider] Clicking element: {view_desc}")

            # Record the activity before clicking
            activity_before = self.current_state.foreground_activity

            # Click the element
            self.device.send_event(event)
            time.sleep(1)

            # Get the new state after clicking
            new_state = self.device.get_current_state()
            activity_after = new_state.foreground_activity if new_state else None

            # Record activity changes
            self.exploration_activity_trace.append({
                'step': self.exploration_step,
                'action': 'navigation',
                'activity_before': activity_before,
                'activity_after': activity_after
            })
            self.logger.info(f"[Guider] Activity trace: {activity_before} → {activity_after}")

            # Record navigation step
            step_info = {
                'step': self.exploration_step,
                'action': 'navigation',
                'element_idx': idx,
                'event': {
                    'type': event.event_type if hasattr(event, 'event_type') else 'unknown',
                    'view': event.view if hasattr(event, 'view') else None
                },
                'state_after': new_state.state_str if new_state else None,
                'screenshot': os.path.join(self.exploration_tmp_dir, f"states/screen_step_{self.exploration_step}.png"),
                'xml': os.path.join(self.exploration_tmp_dir, f"xmls/xml_step_{self.exploration_step}.xml"),
                'found_target': False,
                'matched_view': None
            }
            self.repair_trace.append(step_info)

            # Save navigation event
            event_info = {
                'tag': f"navigation_step_{self.exploration_step}",
                'event': {
                    'event_type': event.event_type if hasattr(event, 'event_type') else 'touch',
                    'log_lines': None,
                    'x': None,
                    'y': None,
                    'view': event.view if hasattr(event, 'view') else None
                },
                'start_state': self.current_state.state_str if self.current_state else None,
                'stop_state': new_state.state_str if new_state else None,
                'event_str': event.get_event_str(self.current_state) if hasattr(event, 'get_event_str') else str(event)
            }
            event_dir = os.path.join(self.exploration_tmp_dir, "events")
            if not os.path.exists(event_dir):
                os.makedirs(event_dir)
            event_path = os.path.join(event_dir, f"event_navigation_step_{self.exploration_step}.json")
            with open(event_path, 'w') as f:
                json.dump(event_info, f, indent=2, ensure_ascii=False)

            # Update current state
            self.current_state = new_state

            # Find target element in the new page (Guider does not use LLM)
            self.logger.info(f"[Guider] Searching for target element in new page...")
            matched_view, matching_method = self.find_target_element_in_page(
                self.current_state, self.exploration_step, cross_page=True,
                is_has_next_screen_summary=False, is_guider=True
            )
            self.logger.info(f"[Guider] Cross page match result: {matched_view is not None}, method: {matching_method}")

            if matched_view:
                self.logger.info("[Guider] ✓ Found target element on cross page")

                # Update the last record in the trace
                self.repair_trace[-1]['found_target'] = True
                self.repair_trace[-1]['matched_view'] = matched_view
                self.repair_trace[-1]['matching_method'] = matching_method

                # Generate marked image for matched_view
                matched_bounds = matched_view.get('bounds')
                if matched_bounds:
                    current_screenshot_path = os.path.join(self.exploration_tmp_dir, f"states/screen_same_page_{self.exploration_step}.png")
                    if os.path.exists(current_screenshot_path):
                        current_img = read_image(current_screenshot_path)
                        if isinstance(matched_bounds, list) and len(matched_bounds) == 2:
                            bounds_str = f"[{matched_bounds[0][0]},{matched_bounds[0][1]}][{matched_bounds[1][0]},{matched_bounds[1][1]}]"
                        else:
                            bounds_str = str(matched_bounds)
                        marked_img = draw_replay_element_on_image(current_img.copy(), bounds_str, id="1")
                        images_dir = os.path.join(self.exploration_tmp_dir, "images")
                        if not os.path.exists(images_dir):
                            os.makedirs(images_dir)
                        marked_img_path = os.path.join(images_dir, f"matched_view_step_{self.exploration_step}.png")
                        marked_img.save(marked_img_path)
                        self.logger.info(f"[Guider] Saved matched_view image to: {marked_img_path}")
                    else:
                        self.logger.warning(f"[Guider] Screenshot not found at: {current_screenshot_path}")

                repaired_event = self._create_repaired_event(matched_view)
                self.last_repaired_view = matched_view

                repaired_event_info = {
                    'tag': f"repaired_event_step_{self.exploration_step}",
                    'event': {
                        'event_type': repaired_event.event_type if hasattr(repaired_event, 'event_type') else 'touch',
                        'log_lines': None,
                        'x': None,
                        'y': None,
                        'view': matched_view
                    },
                    'start_state': self.current_state.state_str if self.current_state else None,
                    'stop_state': None,
                    'event_str': repaired_event.get_event_str(self.current_state) if hasattr(repaired_event, 'get_event_str') else str(repaired_event)
                }
                repaired_event_path = os.path.join(event_dir, f"event_repaired_step_{self.exploration_step}.json")
                with open(repaired_event_path, 'w') as f:
                    json.dump(repaired_event_info, f, indent=2, ensure_ascii=False)
                self.logger.info(f"[Guider] Saved repaired event to: {repaired_event_path}")

                self.save_repair_trace()
                return repaired_event

            # Target element not found, press BACK to return
            # But if currently at main page (start_activity), cannot press back or the app will exit
            current_activity = self.current_state.foreground_activity if self.current_state else None
            if current_activity and current_activity == start_activity:
                self.logger.info(f"[Guider] Currently at main activity ({start_activity}), skipping BACK to avoid exiting app")
                # Do not perform back, continue trying the next element
                continue

            self.logger.info("[Guider] Target not found, pressing BACK to return...")
            back_event = KeyEvent(name="BACK")
            self.device.send_event(back_event)
            time.sleep(1)

            # Get the state after pressing back
            back_state = self.device.get_current_state()
            self.current_state = back_state

            # Record BACK action
            back_step_info = {
                'step': self.exploration_step,
                'action': 'back',
                'state_after': back_state.state_str if back_state else None,
                'screenshot': None,
            }
            self.repair_trace.append(back_step_info)

            self.logger.info(f"[Guider] Returned to activity: {back_state.foreground_activity if back_state else 'unknown'}")

        # Exploration complete, target not found
        self.logger.warning("[Guider] Exploration finished, target element not found")

        # Record the final failure result
        self.repair_trace.append({
            'step': self.exploration_step,
            'action': 'exploration_failed',
            'found_target': False,
            'matched_view': None,
            'matching_method': None
        })

        self.save_repair_trace()
        return None


    def _create_repaired_event(self, matched_view):
        """
        Create the repaired event.

        Args:
            matched_view: The matched view dictionary

        Returns:
            TouchEvent: The repaired touch event
        """
        # Record this matched view for subsequent feedback mechanism
        self.last_repaired_view = matched_view

        repaired_event = TouchEvent(view=matched_view)
        # repaired_event.u2 = self.device.u2
        self.mode = "replay"  # Switch back to replay mode
        return repaired_event

   
    def find_target_element_in_page(self, current_state, step, cross_page=False, is_has_next_screen_summary=True, is_guider=False):
        """
        Find the target element in the current page using the UIMatch algorithm.

        Args:
            current_state: Current device state

        Returns:
            (matched_view, matching_method): Successfully matched view dictionary and matching method, or (None, None)
        """


        try:
            # Use the already loaded failed event data
            if self.failed_event_json is None or self.failed_event_xml_tree is None:
                self.logger.error("Failed event data not loaded")
                return None, None

            # Find the target element in the failed event's XML
            original_element = self._find_original_element(self.failed_event_path, self.failed_event_xml_tree)

            if original_element is None:
                self.logger.warning("Target element not found in original XML")
                return None, None

            # Save current state's XML and screenshot to the exploration_tmp directory
            current_state.tag = f"same_page_{step}"
            state_dir = os.path.join(self.exploration_tmp_dir, "states")
            current_state.save2dir(state_dir)
            current_xml_path = os.path.join(self.exploration_tmp_dir, f"xmls/xml_same_page_{step}.xml")
            current_png_path = os.path.join(self.exploration_tmp_dir, f"states/screen_same_page_{step}.png")

            # Parse current XML
            with open(current_xml_path, 'r', encoding='utf-8') as f:
                current_xml_tree = ET.parse(f)

            matcher = Matcher(
                original_png=self.failed_event_png_path,
                original_tree=self.failed_event_xml_tree,
                original_element=original_element,
                replay_png=current_png_path,
                replay_tree=current_xml_tree,
                logger=self.logger,
                cross_page=cross_page
            )

            # 1. Prepare the original element marked image
            marked_original_path = self.failed_event_png_path.replace(".png", "_marked_original_element.png")
            if not os.path.exists(marked_original_path):
                original_element = self._find_original_element(self.failed_event_path, self.failed_event_xml_tree)
                if original_element is None:
                    self.logger.warning("Cannot find original element for judge")
                    return False
                original_bounds = self._parse_bounds(original_element.attrib.get("bounds", ""))
                original_img = read_image(self.failed_event_png_path)
                marked_original_img = draw_original_element_on_image(original_img, original_bounds)
                marked_original_img.save(marked_original_path)
            else:
                marked_original_img = read_image(marked_original_path)

            marked_original_base64 = get_encoded_image(marked_original_img)

            # 2. Crop the original element image
            original_element = self._find_original_element(self.failed_event_path, self.failed_event_xml_tree)
            if original_element is None:
                self.logger.warning("Cannot find original element for cropping")
                return False
            original_bounds = self._parse_bounds(original_element.attrib.get("bounds", ""))
            # Convert to PIL crop format (x1, y1, x2, y2)
            crop_bounds = (original_bounds[0][0], original_bounds[0][1], original_bounds[1][0], original_bounds[1][1])
            original_full_img = read_image(self.failed_event_png_path)
            original_element_img = original_full_img.crop(crop_bounds)
            original_element_img_base64 = get_encoded_image(original_element_img)



            # v2: Have LLM summarize the changes from current screen to next screen, then pass the description to LLM to recommend the next element
            if is_has_next_screen_summary == False:
                original_next_screen_summary = None
            else:
                if self.failed_event_png_next and self.original_next_screen_summary is None:
                    next_screen_base64 = get_encoded_image(self.failed_event_png_next)
                    original_next_screen_summary = self.get_next_current_description(marked_original_base64, original_element_img_base64, next_screen_base64)
                    self.original_next_screen_summary = original_next_screen_summary
                elif self.original_next_screen_summary is not None:
                    original_next_screen_summary = self.original_next_screen_summary
                else:
                    original_next_screen_summary = None

            

            # Execute matching
            if is_guider:
                matching_result = matcher.guider_matching()
            else:
                matching_result = matcher.matching(app_name=self.app.get_package_name(), without_llm=self.without_llm, without_rule=self.without_rule, original_next_screen_summary=original_next_screen_summary)

            if matching_result.get("success"):
                matched_element = matching_result.get("matched_element")
                matching_method = matching_result.get('matching_method', 'unknown')
                self.logger.info(f"✓ Found target element using {matching_method} matching")

                # using the matched element to replace the event
                for current_view in current_state.views:
                    # normalize
                    resource_id = self.normalize(current_view['resource_id'])
                    text = self.normalize(current_view['text'])  # May change
                    content_description = self.normalize(current_view['content_description'])
                    class_name = self.normalize(current_view['class'])
                    bounds = current_view['bounds']

                    if self.check_if_same(resource_id, matched_element.get('resource-id')) and \
                    self.check_if_same(content_description, matched_element.get('content-desc')) and \
                    self.check_if_same(class_name, matched_element.get('class')) and \
                    self.compare_bounds(bounds, matched_element.get('bounds')):

                        # Check if in exclusion list
                        if self._is_view_excluded(current_view):
                            self.logger.info(f"Skipping excluded view: {current_view.get('view_str', 'unknown')}")
                            continue

                        return current_view, matching_method

                # If no matching view found, return None
                self.logger.warning("Matched element not found in current_state.views")
                return None, None
            else:
                self.logger.info("Target element not found in current page")
                return None, None

        except Exception as e:
            self.logger.error(f"Error in find_target_element_in_page: {e}")
            import traceback
            traceback.print_exc()
            return None, None


    def _find_original_element(self, event_path, xml_tree) -> ET.Element:
        """
        Find the corresponding element in xml_tree based on information in the event.

        Args:
            event_path: Event file path
            xml_tree: XML tree object

        Returns:
            The found element object or None
        """
        import json
        
        # 1. Extract bounds and class information from the event file
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
        
        # 2. Find the node with the same attributes in the XML tree
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

    

    def _is_view_excluded(self, view):
        """
        Check if a view is in the exclusion list.

        Args:
            view: The view dictionary to check

        Returns:
            True if excluded, False otherwise
        """
        if not self.excluded_views:
            return False

        view_str = view.get('view_str', '')
        view_bounds = view.get('bounds', [])

        for excluded in self.excluded_views:
            # Compare by view_str (unique identifier)
            if excluded.get('view_str') == view_str:
                return True
            # Or compare by bounds
            if excluded.get('bounds') == view_bounds:
                return True

        return False

    def _get_view_navigation_id(self, view, activity=None, click_type=None):
        """
        Generate a unique navigation identifier for the view, used to determine if it has been visited.

        Args:
            view: view dictionary
            activity: Current activity name

        Returns:
            (activity, resource_id, class, content_desc) tuple
        """
        resource_id = view.get('resource_id', '') or ''
        class_name = view.get('class', '') or ''
        content_desc = view.get('content_description', '') or ''
        activity = activity or ''
        click_type = click_type or ''
        return (activity, resource_id, class_name, content_desc, click_type)


    def _filter_events_by_rules(self, events):
        """
        Filter overlapping events. When multiple events' center points are covered by the same clickable element,
        only keep the topmost one (highest drawing order).

        Algorithm logic (reference: _HindenWidgetFilter):
        1. Traverse events in ascending drawing order
        2. Use rtree index to record each event's center point
        3. When encountering a clickable event, check if its bounds contain previous events' center points
        4. If so, mark previous events as covered (occluded)
        5. Finally return only the non-occluded events

        Args:
            events: Original event list

        Returns:
            filtered_events: Filtered event list
        """
        import rtree

        self.logger.info(f"Filtering events: original count = {len(events)}")

        # If the event list is empty, return directly
        if not events:
            return events

        try:
            # Sort events by drawing order
            def get_drawing_order(event):
                if hasattr(event, 'view') and event.view:
                    return event.view.get('drawing-order', 0) or 0
                return 0

            sorted_events = sorted(events, key=get_drawing_order)

            # Use rtree index
            idx = rtree.index.Index()
            event_nodes = []  # Store events and their covered status
            covered_set = set()  # Indices of occluded events

            for event in sorted_events:
                # Non-view events are added directly
                if not hasattr(event, 'view') or not event.view:
                    event_nodes.append(event)
                    continue

                bounds = event.view.get('bounds')
                if not bounds:
                    event_nodes.append(event)
                    continue

                # Parse bounds: [[x1, y1], [x2, y2]]
                x1, y1 = bounds[0]
                x2, y2 = bounds[1]

                # Check if the current event is clickable
                clickable = event.view.get('clickable', False)

                if clickable:
                    # Find previous events covered by the current event (center points within current bounds)
                    covered_ids = list(idx.intersection((x1, y1, x2, y2)))
                    for covered_id in covered_ids:
                        # Check if the covered event comes from the same view (same bounds)
                        # If it's different event types from the same view, they should not be filtered
                        covered_event = event_nodes[covered_id]
                        if hasattr(covered_event, 'view') and covered_event.view:
                            cb = covered_event.view.get('bounds')
                            if cb:
                                # If bounds are the same, it's the same view, skip
                                if cb == bounds:
                                    continue
                                # Different view, mark as covered
                                covered_set.add(covered_id)
                                # Remove the covered event from the index
                                cx = (cb[0][0] + cb[1][0]) / 2
                                cy = (cb[0][1] + cb[1][1]) / 2
                                idx.delete(covered_id, (cx, cy, cx, cy))

                # Calculate the current event's center point and insert into index
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                cur_id = len(event_nodes)
                idx.insert(cur_id, (center_x, center_y, center_x, center_y))
                event_nodes.append(event)

            # Filter out occluded events
            filtered_events = []
            for i, event in enumerate(event_nodes):
                if i not in covered_set:
                    filtered_events.append(event)

            self.logger.info(f"Filter stats: covered={len(covered_set)}, kept={len(filtered_events)}")
            self.logger.info(f"Events after filtering: {len(filtered_events)}")
            return filtered_events

        except Exception as e:
            self.logger.error(f"Error in _filter_events_by_uimatch_rules: {e}")
            import traceback
            traceback.print_exc()
            # Return original event list on error
            return events

    def _find_element_by_bounds(self, xml_tree, bounds_str):
        """
        Find the element with specified bounds in the XML tree.

        Args:
            xml_tree: XML tree
            bounds_str: Bounds string in format [x1,y1][x2,y2]

        Returns:
            The found element or None
        """
        for element in xml_tree.iter():
            if element.get("bounds") == bounds_str:
                return element
        return None

    
    def _parse_bounds(self, bounds_str):
        """
        Parse bounds string into coordinate list.

        Args:
            bounds_str: String in "[x1,y1][x2,y2]" format

        Returns:
            List in [[x1, y1], [x2, y2]] format
        """
        import re
        match = re.findall(r'\[(\d+),(\d+)\]', bounds_str)
        if len(match) == 2:
            return [[int(match[0][0]), int(match[0][1])],
                    [int(match[1][0]), int(match[1][1])]]
        return [[0, 0], [0, 0]]

    def save_repair_trace(self):
        """
        Save the complete repair trace to file.
        """
        try:
            repair_log_dir = os.path.join(self.exploration_tmp_dir, "repair_logs")
            os.makedirs(repair_log_dir, exist_ok=True)

            repair_log_path = os.path.join(
                repair_log_dir,
                f"repair_trace_event_{self.failed_event_number}.json"
            )

            # Determine if repair succeeded (last step's found_target is True)
            repair_success = False
            if self.repair_trace and self.repair_trace[-1].get('found_target'):
                repair_success = True

            # Complete repair trace
            repair_log = {
                'failed_event_number': self.failed_event_number,
                'repair_success': repair_success,
                'total_steps': len(self.repair_trace),
                'timestamp': str(__import__('datetime').datetime.now()),
                'trace': self.repair_trace  # Save the complete trace list
            }

            # Save to JSON file
            with open(repair_log_path, 'w', encoding='utf-8') as f:
                json.dump(repair_log, f, indent=2, ensure_ascii=False)

            self.logger.info(f"✓ Repair trace saved to: {repair_log_path}")

        except Exception as e:
            self.logger.error(f"Error saving repair trace: {e}")
            import traceback
            traceback.print_exc()

    def check_if_same(self, current, record):
        if current is None or record is None:
            return False
        if current == record:
            return True
        return False

    def replace_view(self, event, current_view):
        event.view['resource_id'] = current_view['resource_id']
        event.view['text'] = current_view['text']
        event.view['content_description'] = current_view['content_description']
        event.view['class'] = current_view['class']
        event.view['instance'] = current_view['instance']
        event.view['bounds'] = current_view['bounds']

    def check_which_exists(self, event):
        resource_id = GuiderPolicy.__safe_dict_get(event.view, 'resource_id')
        text = GuiderPolicy.__safe_dict_get(event.view, 'text')
        content_description = GuiderPolicy.__safe_dict_get(event.view, 'content_description')
        class_name = GuiderPolicy.__safe_dict_get(event.view, 'class')
        instance = GuiderPolicy.__safe_dict_get(event.view, 'instance')

        u2 = self.device.u2
        

        if content_description is not None:
            if u2.exists(description=content_description, instance=instance):
                for current_view in self.current_state.views:
                    if self.check_if_same(current_view['content_description'], content_description) and self.check_if_same(current_view['instance'], instance):
                        self.replace_view(event, current_view)
                        break
                return 'content_description', content_description
        elif text is not None:
            if u2.exists(text=text, instance=instance):
                for current_view in self.current_state.views:
                    if self.check_if_same(current_view['text'], text) and self.check_if_same(current_view['instance'], instance):
                        self.replace_view(event, current_view)
                        break
                return 'text', text
        elif resource_id is not None:
            if u2.exists(resourceId=resource_id, instance=instance):
                for current_view in self.current_state.views:
                    if self.check_if_same(current_view['resource_id'], resource_id) and self.check_if_same(current_view['instance'], instance):
                        self.replace_view(event, current_view)
                        break
                return 'resource_id', resource_id
        elif class_name is not None:
            if u2.exists(className=class_name, instance=instance):
                for current_view in self.current_state.views:
                    if self.check_if_same(current_view['class'], class_name) and self.check_if_same(current_view['instance'], instance):
                        self.replace_view(event, current_view)
                        break
                return 'class_name', class_name
        elif class_name is not None and resource_id is not None and instance is not None:
            if u2.exists(className=class_name, resourceId=resource_id, instance=instance):
                for current_view in self.current_state.views:
                    if self.check_if_same(current_view['class'], class_name) and self.check_if_same(current_view['resource_id'], resource_id) and self.check_if_same(current_view['instance'], instance):
                        self.replace_view(event, current_view)
                        break
                return 'class_resource_instance', (class_name, resource_id, instance)
        
        return None, None
    

    @staticmethod
    def __safe_dict_get(view_dict, key, default=None):
        value = view_dict[key] if key in view_dict else None
        return value if value is not None else default