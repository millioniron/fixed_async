"""Phone Platform Environment for agentic training."""

import logging
import copy
import re
import json
from typing import Optional, Dict, Any, List

from datasets import Dataset
from gem import Env
import ray

from roll.datasets.global_dataset import GlobalDataset, GlobalDatasetManager
from roll.utils.constants import RAY_NAMESPACE
from roll.pipeline.agentic.env.gz.utils import introspect_function_to_tool
from roll.pipeline.agentic.env.gz.base_api import BaseApi
from roll.pipeline.agentic.env.gz.message_api import MessageApi
from roll.pipeline.agentic.env.gz.food_platform import FoodPlatform
from roll.pipeline.agentic.env.gz.reminder_api import ReminderApi
from roll.pipeline.agentic.env.gz.travel_api import TravelApi

logger = logging.getLogger(__name__)


class PhonePlatform(Env):
    """Phone Platform Environment combining BaseApi, MessageApi, FoodPlatform, ReminderApi, and TravelApi."""

    def __init__(
            self,
            dataset_name: Optional[str] = "",
            split: Optional[str] = "train",
            dataset: Optional[Dataset] = None,
            question_key: str = "question",
            mile_stone_key: str = "mile_stone",
            ground_status_key: str = "ground_truth",
            initial_config_key: str = "initial_config",
            answer_key: str = "answer",
            max_steps: int = 100,
            seed: int = 0,
            mode: str = "train",
            format_penalty: float = 0.0,
            **_,
    ):
        super().__init__()

        self.seed = seed
        self.question_key = question_key
        self.answer_key = answer_key
        self.mile_stone_key = mile_stone_key
        self.ground_status_key = ground_status_key
        self.initial_config_key = initial_config_key
        self.mode = mode

        # Dataset setup
        global_dataset_mode = "sample" if self.mode == "train" else "traversal"
        self.dataset = GlobalDataset.options(
            name=f"{self.mode}_{dataset_name}",
            get_if_exists=True,
            namespace=RAY_NAMESPACE
        ).remote(dataset_name=dataset_name, split=split, mode=global_dataset_mode)

        self.dataset_manager = GlobalDatasetManager.options(
            name=f"{self.mode}_dataset_manager",
            get_if_exists=True,
            namespace=RAY_NAMESPACE
        ).remote()
        ray.get(self.dataset_manager.register.remote(dataset_name=dataset_name, dataset_ref=self.dataset))

        self.max_steps = max_steps
        self.step_count = 0
        self.render_cache = None
        self.format_penalty = format_penalty

        # Initialize sub-modules
        self._initialize_state()

    def _initialize_state(self):
        """Initialize all sub-modules."""
        self.base_api = BaseApi()
        self.message_api = MessageApi()
        self.food_platform = FoodPlatform()
        self.reminder_api = ReminderApi()
        self.travel_api = TravelApi()

        # Build available tools mapping
        self.available_tools = {
            # BaseApi tools
            "turn_on_wifi": self.base_api.turn_on_wifi,
            "turn_off_wifi": self.base_api.turn_off_wifi,
            "login_device": self.base_api.login_device,
            "logout_device": self.base_api.logout_device,
            # MessageApi tools
            "send_message": self.message_api.send_message,
            "delete_message": self.message_api.delete_message,
            "view_messages_between_users": self.message_api.view_messages_between_users,
            "search_messages": self.message_api.search_messages,
            "get_latest_message_id": self.message_api.get_latest_message_id,
            "get_earliest_message_id": self.message_api.get_earliest_message_id,
            # FoodPlatform tools
            "login_food_platform": self.food_platform.login_food_platform,
            "get_products": self.food_platform.get_products,
            "add_food_delivery_order": self.food_platform.add_food_delivery_order,
            "check_balance": self.food_platform.check_balance,
            "view_orders": self.food_platform.view_orders,
            # ReminderApi tools
            "view_reminder_by_title": self.reminder_api.view_reminder_by_title,
            "add_reminder": self.reminder_api.add_reminder,
            "delete_reminder": self.reminder_api.delete_reminder,
            "view_all_reminders": self.reminder_api.view_all_reminders,
            # TravelApi tools
            "get_flight_details": self.travel_api.get_flight_details,
            "find_transfer_flights": self.travel_api.find_transfer_flights,
            "reserve_flight": self.travel_api.reserve_flight,
            "get_reservation_details": self.travel_api.get_reservation_details,
            "modify_flight": self.travel_api.modify_flight,
            "cancel_reservation": self.travel_api.cancel_reservation,
            "get_user_details": self.travel_api.get_user_details,
            "get_baggage_allowance": self.travel_api.get_baggage_allowance,
        }

        self.env_instruction = ""
        self.mile_stone = []

    def _load_initial_config(self, initial_config: dict):
        """Load initial configuration for all sub-modules."""
        if not initial_config:
            return

        if "BaseApi" in initial_config:
            self.base_api.load_config(initial_config["BaseApi"])
            # Sync wifi/logged_in to other modules
            self.message_api.wifi = self.base_api.wifi
            self.message_api.logged_in = self.base_api.logged_in
            self.food_platform.wifi = self.base_api.wifi
            self.food_platform.logged_in = self.base_api.logged_in
            self.reminder_api.wifi = self.base_api.wifi
            self.reminder_api.logged_in = self.base_api.logged_in
            self.travel_api.wifi = self.base_api.wifi
            self.travel_api.logged_in = self.base_api.logged_in

        if "FoodPlatform" in initial_config:
            self.food_platform.load_config(initial_config["FoodPlatform"])

        if "MessageApi" in initial_config:
            self.message_api.load_config(initial_config["MessageApi"])

        if "ReminderApi" in initial_config:
            self.reminder_api.load_config(initial_config["ReminderApi"])

        if "TravelApi" in initial_config:
            self.travel_api.load_config(initial_config["TravelApi"])

    def reset(self, seed) -> None:
        """Reset the environment."""
        self.step_count = 0
        Env.reset(self, seed)

        data: Optional[Dict] = ray.get(self.dataset.get_data_item.remote(seed=seed))
        if data is None:
            return None, None

        self.first_obs = data[self.question_key]
        self.ground_mile_stone = data.get(self.mile_stone_key, '')
        self.ground_status = data.get(self.ground_status_key, '')
        self.ground_answer = data.get(self.answer_key, '')

        # Re-initialize state
        self._initialize_state()

        # Load initial config from data
        initial_config = data.get(self.initial_config_key, {})
        self._load_initial_config(initial_config)

        self.prepare_render_cache(self.first_obs)

        # Build tools list
        tools = []
        for name, func in self.available_tools.items():
            tool_schema = introspect_function_to_tool(func)
            tools.append(tool_schema)

        return self.render(), {"env_instruction": self.env_instruction, "tools": tools}

    def render(self, mode=None):
        """Render the environment."""
        return self.render_cache

    def prepare_render_cache(self, observation: str):
        """Prepare the render cache for the environment."""
        self.render_cache = observation

    def parse_action_tool(self, text: str):
        """Parse text to extract tool calls."""
        pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)

        if not matches:
            return []

        tool_calls = []
        for match in matches:
            try:
                tool_data = json.loads(match.strip())
                action_info = {
                    "tool_name": tool_data.get("name", ""),
                    "parsed_action": f'<tool_call>{match}</tool_call>',
                    "parameters": tool_data.get("arguments", {}),
                    "is_valid": True
                }
                tool_calls.append(action_info)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool JSON: {match}")
                action_info = {
                    "tool_name": "",
                    "parsed_action": f'<tool_call>{match}</tool_call>',
                    "parameters": {},
                    "is_valid": False
                }
                tool_calls.append(action_info)

        return tool_calls

    def step(self, action: str):
        """Execute one step in the environment."""
        metrics_agg_mode = {
            "action_is_effective": "mean",
            "action_is_valid": "mean",
            "success": "last",
            "format_penalty": "mean",
        }
        self.step_count += 1

        tool_calls = self.parse_action_tool(action)
        observations = []
        individual_results = []
        all_valid = True
        any_effective = False

        if not tool_calls:
            action_desc = f"At turn {self.step_count}, no valid tool call format detected."
            metrics = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
                "format_penalty": self.format_penalty,
            }
            info = {
                "metrics": metrics,
                "metrics_agg_mode": metrics_agg_mode,
                "action_desc": action_desc,
                "tool_calls": [],
                "use_tool": True,
            }
            truncated = self.step_count >= self.max_steps
            observation = action_desc + "The task is not yet resolved; please continue."
            return observation, self.format_penalty, truncated, truncated, info

        # Process each tool call
        for idx, action_info in enumerate(tool_calls):
            tool_name = action_info["tool_name"]
            parameters = action_info["parameters"]
            is_valid = action_info["is_valid"]

            result = {
                "index": idx,
                "tool_name": tool_name,
                "parameters": parameters,
                "is_valid_format": is_valid,
                "observation": "",
                "is_executed": False,
                "error": None,
            }

            if not is_valid or not tool_name:
                all_valid = False
                obs = f"[Tool {idx+1}] Invalid tool call format."
                observations.append(obs)
                result["observation"] = obs
                individual_results.append(result)
                continue

            if tool_name not in self.available_tools:
                all_valid = False
                obs = f"[Tool {idx+1}] Tool '{tool_name}' is not available."
                observations.append(obs)
                result["observation"] = obs
                individual_results.append(result)
                continue

            # Execute the tool
            try:
                tool_func = self.available_tools[tool_name]
                output = tool_func(**parameters)
                obs = f"[Tool {idx+1}: {tool_name}] {output}"
                observations.append(obs)
                result["observation"] = obs
                result["is_executed"] = True
                any_effective = True

                # Sync state after BaseApi operations
                if tool_name in ["turn_on_wifi", "turn_off_wifi", "login_device", "logout_device"]:
                    self.message_api.wifi = self.base_api.wifi
                    self.message_api.logged_in = self.base_api.logged_in
                    self.food_platform.wifi = self.base_api.wifi
                    self.food_platform.logged_in = self.base_api.logged_in
                    self.reminder_api.wifi = self.base_api.wifi
                    self.reminder_api.logged_in = self.base_api.logged_in
                    self.travel_api.wifi = self.base_api.wifi
                    self.travel_api.logged_in = self.base_api.logged_in

            except Exception as e:
                all_valid = False
                obs = f"[Tool {idx+1}: {tool_name}] Error: {str(e)}"
                observations.append(obs)
                result["observation"] = obs
                result["error"] = str(e)

            individual_results.append(result)

        # Build final observation
        combined_observation = "\n".join(observations)
        action_desc = f"At turn {self.step_count}, executed {len(tool_calls)} tool(s)."

        # Check success
        success = self.judge_success()
        truncated = self.step_count >= self.max_steps

        # Calculate reward
        format_penalty_value = 0.0 if all_valid else self.format_penalty
        reward = 1.0 if success else format_penalty_value

        metrics = {
            "action_is_effective": any_effective,
            "action_is_valid": all_valid,
            "success": success,
            "format_penalty": format_penalty_value,
        }

        info = {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode,
            "action_desc": action_desc,
            "tool_calls": individual_results,
            "use_tool": True,
        }

        if not success:
            combined_observation += "\nThe task is not yet resolved; please continue."

        return combined_observation, reward, success, truncated, info

    def judge_success(self) -> bool:
        """Judge if the task is successfully completed by comparing current state with ground truth."""
        if not self.ground_status:
            return False

        try:
            ground_status = json.loads(self.ground_status) if isinstance(self.ground_status, str) else self.ground_status
        except json.JSONDecodeError:
            logger.error(f"Failed to parse ground_status: {self.ground_status}")
            return False

        # Get current states from all sub-modules
        current_state = {
            "BaseApi": self.base_api.get_state(),
            "MessageApi": self.message_api.get_state(),
            "FoodPlatform": self.food_platform.get_state(),
            "ReminderApi": self.reminder_api.get_state(),
            "TravelApi": self.travel_api.get_state(),
        }

        # Compare each module's state with ground truth
        for module_name, expected_state in ground_status.items():
            if module_name not in current_state:
                continue

            actual_state = current_state[module_name]
            if not self._compare_state(expected_state, actual_state):
                return False

        return True

    def _compare_state(self, expected: Any, actual: Any) -> bool:
        """Recursively compare expected state with actual state."""
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False
            for key, value in expected.items():
                if key not in actual:
                    return False
                if not self._compare_state(value, actual[key]):
                    return False
            return True
        elif isinstance(expected, list):
            if not isinstance(actual, list):
                return False
            if len(expected) != len(actual):
                return False
            for exp_item, act_item in zip(expected, actual):
                if not self._compare_state(exp_item, act_item):
                    return False
            return True
        else:
            return expected == actual