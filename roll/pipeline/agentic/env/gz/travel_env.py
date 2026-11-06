import logging
import multiprocessing
import random
from typing import Optional, Tuple, Any, SupportsFloat, Dict

from datasets import load_dataset, Dataset, DatasetDict
from gem import Env
import ray

from roll.datasets.global_dataset import GlobalDataset, GlobalDatasetManager
from roll.utils.constants import RAY_NAMESPACE
from roll.pipeline.agentic.env.gz.utils import introspect_function_to_tool
import re

logger = logging.getLogger(__name__)



import json

from datetime import datetime, timedelta

# 待优化工具

class Travel(Env):
    def __init__(
            self,
            dataset_name: Optional[str] = "",
            split: Optional[str] = "train",
            dataset: Optional[Dataset] = None,
            question_key: str = "question",
            mile_stone_key: str = "mile_stone",
            ground_status_key: str = "ground_status",
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
        self.mile_stone_key=mile_stone_key
        self.ground_status_key=ground_status_key
        self.mode = mode

        # Convert train/val mode to sample/traversal for GlobalDataset
        global_dataset_mode = "sample" if self.mode == "train" else "traversal"
        self.dataset = GlobalDataset.options(name=f"{self.mode}_{dataset_name}",
                                             get_if_exists=True,
                                             namespace=RAY_NAMESPACE).remote(dataset_name=dataset_name,
                                                                             split=split,
                                                                             mode=global_dataset_mode)
        self.dataset_manager = GlobalDatasetManager.options(name=f"{self.mode}_dataset_manager",
                                                            get_if_exists=True,
                                                            namespace=RAY_NAMESPACE).remote()
        ray.get(self.dataset_manager.register.remote(dataset_name=dataset_name, dataset_ref=self.dataset))
        self.max_steps = max_steps
        self.step_count = 0
        self.render_cache = None
        self.format_penalty = format_penalty
        """
        Initialize the system, including user profiles and flight information
        """
        ####状态数据库
        # Initialize user information
        self._initialize_state()
    def _initialize_state(self):
        """Initialize the state databases with default values"""
        self.users = {
            "user1": {"user_name": "Eve", "password": "password123", "cash_balance": 2000.0, "bank_balance": 50000.0, "membership_level": "regular"},
            "user2": {"user_name": "Frank", "password": "password456", "cash_balance": 8000.0, "bank_balance": 8000.0, "membership_level": "silver"},
            "user3": {"user_name": "Grace", "password": "password789", "cash_balance": 1000.0, "bank_balance": 5000.0, "membership_level": "gold"}
        }

        self.flights = [
            {
                "flight_no": "CA1234",
                "origin": "Beijing",
                "destination": "Shanghai",
                "depart_time": "2024-07-15 08:00:00",
                "arrival_time": "2024-07-15 10:30:00",
                "status": "available",
                "seats_available": 5,
                "economy_price": 1200,
                "business_price": 3000
            },
            {
                "flight_no": "MU5678",
                "origin": "Shanghai",
                "destination": "Beijing",
                "depart_time": "2024-07-16 09:00:00",
                "arrival_time": "2024-07-16 11:30:00",
                "status": "available",
                "seats_available": 3,
                "economy_price": 1900,
                "business_price": 3000
            },
            {
                "flight_no": "CZ4321",
                "origin": "Shanghai",
                "destination": "Beijing",
                "depart_time": "2024-07-16 20:00:00",
                "arrival_time": "2024-07-16 22:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 2500,
                "business_price": 4000
            },
            {
                "flight_no": "CZ4352",
                "origin": "Shanghai",
                "destination": "Beijing",
                "depart_time": "2024-07-17 20:00:00",
                "arrival_time": "2024-07-17 22:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1600,
                "business_price": 2500
            },
            {
                "flight_no": "MU3561",
                "origin": "Beijing",
                "destination": "Nanjing",
                "depart_time": "2024-07-18 08:00:00",
                "arrival_time": "2024-07-18 10:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 4000
            },
            {
                "flight_no": "MU1566",
                "origin": "Beijing",
                "destination": "Nanjing",
                "depart_time": "2024-07-18 20:00:00",
                "arrival_time": "2024-07-18 22:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 4000
            },
            {
                "flight_no": "CZ1765",
                "origin": "Nanjing",
                "destination": "Shenzhen",
                "depart_time": "2024-07-17 20:30:00",
                "arrival_time": "2024-07-17 22:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 2500
            },
            {
                "flight_no": "CZ1765",
                "origin": "Nanjing",
                "destination": "Shenzhen",
                "depart_time": "2024-07-18 12:30:00",
                "arrival_time": "2024-07-18 15:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 2500
            },
            {
                "flight_no": "MH1765",
                "origin": "Xiamen",
                "destination": "Chengdu",
                "depart_time": "2024-07-17 12:30:00",
                "arrival_time": "2024-07-17 15:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 2500
            },
            {
                "flight_no": "MH2616",
                "origin": "Chengdu",
                "destination": "Xiamen",
                "depart_time": "2024-07-18 18:30:00",
                "arrival_time": "2024-07-18 21:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 2500
            },
            {
                "flight_no": "MH2616",
                "origin": "Chengdu",
                "destination": "Fuzhou",
                "depart_time": "2024-07-16 18:30:00",
                "arrival_time": "2024-07-16 21:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 2500
            }
        ]

        self.reservations = [
            {
                "reservation_id": "res_1",
                "user_id": "user1",
                "flight_no": "CA1234",
                "payment_method": "bank",
                "cabin": "Economy Class",
                "baggage": 1,
                "origin": "Beijing",
                "destination": "Shanghai",
            },
            {
                "reservation_id": "res_2",
                "user_id": "user1",
                "flight_no": "MU5678",
                "payment_method": "bank",
                "cabin": "Business Class",
                "baggage": 1,
                "origin": "Shanghai",
                "destination": "Beijing",
            },
            {
                "reservation_id": "res_3",
                "user_id": "user2",
                "flight_no": "MH1765",
                "payment_method": "bank",
                "cabin": "Business Class",
                "baggage": 1,
                "origin": "Xiamen",
                "destination": "Chengdu",
            },
            {
                "reservation_id": "res_4",
                "user_id": "user2",
                "flight_no": "MU2616",
                "payment_method": "bank",
                "cabin": "Business Class",
                "baggage": 1,
                "origin": "Chengdu",
                "destination": "Xiamen",
            },
        ]

        # Save initial state for reset
        self.init_users = self.users.copy()
        self.init_flights = self.flights.copy()
        self.init_reservations = self.reservations.copy()
        self.available_tools = {
            "get_flight_details": self.get_flight_details,
            "get_user_details": self.get_user_details,
            "get_reservation_details": self.get_reservation_details,
            "find_transfer_flights": self.find_transfer_flights,
            "reserve_flight": self.reserve_flight,
            "modify_flight": self.modify_flight,
            "cancel_reservation": self.cancel_reservation,
        }

        self.env_instruction = ""

        self.mile_stone = []

    def reset(self,seed) -> None:
        self.step_count = 0
        Env.reset(self, seed)
        data: Optional[Dict] = ray.get(self.dataset.get_data_item.remote(seed=seed))
        if data is None:
            return None, None

        self.first_obs = data[self.question_key]

        self.ground_mile_stone = data.get(self.mile_stone_key,'')      ### 通常是正确执行的顺序（mile_stone）或者数据库的状态（status）操作或者最终的回答（answer）。
        self.ground_status = data.get(self.ground_status_key,'')
        self.ground_answer = data.get(self.answer_key,'')
        self.prepare_render_cache(self.first_obs)

        # self.users=self.init_users.copy()
        # self.flights=self.init_flights.copy()
        # self.reservations=self.init_reservations.copy()
        # self.mile_stone=[]
        self._initialize_state()

        tools = []
        for name, func in self.available_tools.items():
            tool_schema = introspect_function_to_tool(func)
            tools.append(tool_schema)
        return self.render(), {"env_instruction": self.env_instruction,"tools":tools}

    def render(self, mode=None):
        """
        Render the environment.
        """
        return self.render_cache

    def prepare_render_cache(self, observation: str):
        """
        Prepare the render cache for the environment.
        """
        self.render_cache = observation

    def parse_action_tool(self,text: str):
        """Parse text to extract multiple tool calls in new format.

        Expected format: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>

        Args:
            text: Raw text string from agent

        Returns:
            list: 包含多个工具信息的列表
        """
        # New format pattern - matches JSON objects inside tool_call tags
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
                logger.error(f"Error: {e}")
                action_info = {
                    "tool_name": "",
                    "parsed_action": f'<tool_call>{match}</tool_call>',
                    "parameters": {},
                    "is_valid": False
                }
                tool_calls.append(action_info)

        return tool_calls
#### 处理tool的输出和observation的输出，确定好
#### 调用工具成功，就是工具的输出。调用工具失败（没有这个工具/参数不对等等），就是user的输出observation
    def step(self, action: str):
        metrics_agg_mode = {
            "action_is_effective": "mean",
            "action_is_valid": "mean",
            "success": "last",
            "format_penalty": "mean",
        }
        self.step_count += 1

        # Parse action to extract tool calls
        tool_calls = self.parse_action_tool(action)
        observations = []
        individual_results = []
        all_valid = True
        any_effective = False
        if not tool_calls:
            # No tool calls found → invalid format
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
                "current_state": {
                "users": self.users,
                "flights": self.flights,
                "reservations": self.reservations
            }
            }
            truncated = self.step_count >= self.max_steps
            observation = action_desc+"The task is not yet resolved; please continue."
            return observation, self.format_penalty, truncated, truncated, info

         # Process each tool call
        for idx, action_info in enumerate(tool_calls):
            tool_name = action_info["tool_name"]
            parameters = action_info["parameters"]
            is_valid = action_info["is_valid"]
            parsed_action = action_info["parsed_action"]

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

            # Try to execute the tool
            try:
                #调用一下
                tool_func = self.available_tools[tool_name]
                output = tool_func(**parameters)
                obs = f"[Tool {idx+1}: {tool_name}] {output}"
                observations.append(obs)
                result["observation"] = obs
                result["is_executed"] = True
                any_effective = True
            except Exception as e:
                all_valid = False
                error_msg = f"[Tool {idx+1}: {tool_name}] Execution error: {str(e)}"
                observations.append(error_msg)
                result["observation"] = error_msg
                result["error"] = str(e)
                # Depending on your definition, execution error may still be "valid format"
                # but not "effective". We keep is_valid_format=True if JSON was valid.
                logger.error(f"Tool execution error: {e}")

            individual_results.append(result)

        # Build final observation
        observation = "\n".join(observations)
        action_desc = f"At turn {self.step_count}, executed {len(tool_calls)} tool(s)."

        # Metrics
        # - action_is_valid: all tool calls must be valid (format + available)
        # - action_is_effective: at least one tool executed successfully
        metrics = {
            "action_is_effective": any_effective, #可以修成有效数，或者总共调用了多少函数？
            "action_is_valid": all_valid,
            "success": self.judge_success(),
            "format_penalty": 0.0 if all_valid else self.format_penalty,
        }

        info = {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode,
            "action_desc": action_desc,
            "tool_calls": individual_results,  # detailed per-tool info
            "use_tool": True,
            "current_state": {
                "users": self.users,
                "flights": self.flights,
                "reservations": self.reservations
            }
        }

        terminated = metrics['success']

        truncated = self.step_count >= self.max_steps

        reward_format_penality = 0.0 if all_valid else self.format_penalty
        reward_success = metrics['success']

        return observation, reward_success+reward_format_penality, terminated, truncated, info


    def judge_success(self) -> bool:
        """Judge if the current state matches the ground truth, skipping comparisons for any field that is None in ground truth."""

        def _is_equal_skip_none(current, truth):
            """
            Recursively compare two objects (dict, list, or primitive).
            - If `truth` is a dict and a value is None, skip that key.
            - Lists are compared element-wise (must be same length).
            - For dicts, `current` must not have extra keys beyond those in `truth` (unless truth has them as None).
            """
            if isinstance(truth, dict):
                if not isinstance(current, dict):
                    return False
                # Check all keys in truth
                for key, val in truth.items():
                    if val is None:
                        continue  # Skip None fields in ground truth
                    if key not in current or not _is_equal_skip_none(current[key], val):
                        return False
                # Ensure current doesn't have extra keys
                for key in current:
                    if key not in truth:
                        return False
                return True

            elif isinstance(truth, list):
                if not isinstance(current, list) or len(current) != len(truth):
                    return False
                # Compare each item pairwise
                return all(_is_equal_skip_none(c_item, t_item) for c_item, t_item in zip(current, truth))

            else:
                # Primitive type comparison
                return current == truth

        mile_stone_success = True  # assuming milestone logic is handled elsewhere or not used

        status_success = True
        if isinstance(self.ground_status, dict):
            # Compare each top-level section (users, flights, reservations) with _is_equal_skip_none
            for key in ['users', 'flights', 'reservations']:
                if key in self.ground_status:
                    current_val = getattr(self, key, None)
                    truth_val = self.ground_status[key]
                    if not _is_equal_skip_none(current_val, truth_val):
                        status_success = False
                        break  # Early exit on first mismatch

        answer_success = True  # adjust based on your actual answer checking logic

        return 1 if mile_stone_success and status_success and answer_success else 0

############################################################################################################################################################

    def _load_scenario(self, scenario: dict, long_context=False) -> None:
        pass
    def get_flight_details(self, origin: str = None, destination: str = None) -> list:
        """
        Retrieve flight details based on origin and destination.

        Args:
            origin (str, optional): The origin city of the flight.
            destination (str, optional): The destination city of the flight.
        """
        flights = self.flights

        if origin:
            flights = [flight for flight in flights if flight["origin"] == origin]

        if destination:
            flights = [flight for flight in flights if flight["destination"] == destination]
        if len(flights) == 0:
            return f'There are no direct flights that meet the criteria.'

        return [{"flight_no": flight["flight_no"], "origin": flight["origin"], "destination": flight["destination"],
                "depart_time": flight["depart_time"], "arrival_time": flight["arrival_time"],
                "status": flight["status"], "seats_available": flight["seats_available"],
                "economy_price": flight["economy_price"], "business_price": flight["business_price"]}
                for flight in flights]


    def get_user_details(self, user_id: str, password: str) -> dict:
        """
        Retrieve user details based on user ID and password.

        Args:
            user_id (str): The id of the user.
            password (str): The password of the user.
        """
        user = self.users.get(user_id)
        if user and user["password"] == password:
            return {key: value for key, value in user.items() if key != "password"}
        return {"status": "error", "message": "Incorrect username or password."}

    def get_reservation_details(self, reservation_id: str = None, user_id: str = None) -> list:
        """
        Retrieve reservation details based on reservation ID or user ID.

        Args:
            reservation_id (str, optional): The reservation ID of the booking.
            user_id (str, optional): The ID of the user.
        """
        # Filter reservations by reservation ID or user ID
        if reservation_id:
            reservations = [reservation for reservation in self.reservations if reservation["reservation_id"] == reservation_id]
        elif user_id:
            reservations = [reservation for reservation in self.reservations if reservation["user_id"] == user_id]
        else:
            return {"status": "error", "message": "Please provide a valid reservation ID or user ID"}

        # Add flight information for each reservation
        detailed_reservations = []
        for reservation in reservations:
            flight_info = next((flight for flight in self.flights if flight["flight_no"] == reservation["flight_no"]), None)
            detailed_reservation = {**reservation, "flight_info": flight_info}
            detailed_reservations.append(detailed_reservation)

        return detailed_reservations

    def authenticate_user(self, user_id, password):
        user = self.users.get(user_id)
        if user and user["password"] == password:
            return user
        return {"status": "error", "message": "Incorrect username or password."}


    def get_baggage_allowance(self, membership_level, cabin_class):
        """
        Get free checked baggage allowance based on membership level and cabin class.

        Parameters:
        - membership_level (str): Membership level ("regular", "silver", "gold")
        - cabin_class (str): Cabin class ("Economy Class", "Business Class")

        Returns:
        - int: Number of free checked baggage
        """
        allowance = {
            "regular": { "Economy Class": 1, "Business Class": 2},
            "silver": { "Economy Class": 2, "Business Class": 3},
            "gold": {"Economy Class": 3, "Business Class": 3}
        }
        return allowance.get(membership_level, {}).get(cabin_class, 0)

    # 查询中转航班
    def find_transfer_flights(self, origin_city: str, transfer_city: str, destination_city: str):
        """
        Find transfer flights from an origin city to a destination city with a stopover in a transfer city, ensuring the first flight arrives before the second flight departs.

        Args:
            origin_city (str): The starting city for the flight.
            transfer_city (str): The city where the transfer takes place.
            destination_city (str): The final destination city for the flight.
        """
        # Get flights from departure city to transfer city
        first_leg_flights = [
            flight for flight in self.flights
            if flight["origin"] == origin_city and flight["destination"] == transfer_city and flight["status"] == "available"
        ]

        # Get flights from transfer city to destination city
        second_leg_flights = [
            flight for flight in self.flights
            if flight["origin"] == transfer_city and flight["destination"] == destination_city and flight["status"] == "available"
        ]

        # Store valid transfer flight combinations
        transfer_flights = []

        # Check each combination of first and second leg flights for valid connections
        for first_flight in first_leg_flights:
            first_arrival = datetime.strptime(first_flight["arrival_time"], "%Y-%m-%d %H:%M:%S")

            for second_flight in second_leg_flights:
                second_departure = datetime.strptime(second_flight["depart_time"], "%Y-%m-%d %H:%M:%S")

                # Check if first flight arrives before second flight departs
                if first_arrival < second_departure:
                    transfer_flights.append({
                        "first_leg": first_flight,
                        "second_leg": second_flight
                    })

        # Return list of valid transfer flights
        if transfer_flights:
            return transfer_flights
        else:
            return "No connecting flights that meet the criteria were found."


    def calculate_baggage_fee(self, membership_level, cabin_class, baggage_count):
        free_baggage = {"regular": {"Economy Class": 1, "Business Class": 2}, "silver": {"Economy Class": 2, "Business Class": 3}, "gold": {"Economy Class": 3, "Business Class": 3}}
        free_limit = free_baggage[membership_level][cabin_class]
        additional_baggage = max(baggage_count - free_limit, 0)
        return additional_baggage * 50


    def update_balance(self, user, payment_method, amount):
        """
        Update user's balance.
        :param user: User information
        :param payment_method: Payment method ("cash" or "bank")
        :param amount: Update amount (positive for increase, negative for decrease)
        :return: Returns True if balance is sufficient and update successful, False otherwise.
        """
        if payment_method == "cash":
            if user["cash_balance"] + amount < 0:
                return False  # Insufficient balance
            user["cash_balance"] += amount
        elif payment_method == "bank":
            if user["bank_balance"] + amount < 0:
                return False  # Insufficient balance
            user["bank_balance"] += amount
        return True

    # 预定航班
    def reserve_flight(self, user_id: str, password: str, flight_no: str, cabin: str, payment_method: str, baggage_count: int):
        """
        Book a flight for the user, including passenger details and payment methods.

        Args:
            user_id (str): The ID of the user making the reservation.ssss
            password (str): The password of the user for authentication.
            flight_no (str): The flight number the user wants to book.
            cabin (str): The cabin class ("Economy Class", "Business Classsss") for the booking.
            payment_method (str): the payment method(e.g., bank, cash) to reserve_flight
            baggage_count (int): The total number of checked baggage items.
        """
        user = self.authenticate_user(user_id, password)
        if not user:
            return "Authentication failed. Please check your user ID and password."

        # Check flight and seats availability
        flight = next((f for f in self.flights if f["flight_no"] == flight_no and f["status"] == "available"), None)

        # Calculate flight price
        price = flight["economy_price"] if cabin == "Economy Class" else flight["business_price"]
        total_cost = price

        # Calculate baggage fee
        baggage_fee = self.calculate_baggage_fee(user["membership_level"], cabin, baggage_count)
        total_cost += baggage_fee

        # Check payment method
        if payment_method not in ["cash", "bank"]:
            return "Invalid payment method"

        # Update balance after reservation
        if payment_method == "cash":
            if total_cost > self.users.get(user_id)["cash_balance"]:
                return f"Your cash balance is insufficient. Please consider using another payment method."
            self.users.get(user_id)["cash_balance"] -= total_cost
        else:
            if total_cost > self.users.get(user_id)["bank_balance"]:
                return f"Your bank balance is insufficient. Please consider using another payment method."
            self.users.get(user_id)["bank_balance"] -= total_cost

        # Update flight information and generate reservation
        flight["seats_available"] -= 1
        reservation_id = f"res_{len(self.reservations) + 1}"
        reservation = {
            "reservation_id": reservation_id,
            "user_id": user_id,
            "flight_no": flight_no,
            "payment_method": payment_method,
            "cabin": cabin,
            "baggage": baggage_count,
        }
        self.reservations.append(reservation)

        return f"Booking successful. Reservation ID: {reservation_id}. Total cost: {total_cost} yuan (including baggage fees)."


    def modify_flight(self, user_id: str, reservation_id: str, new_flight_no: str = None, new_cabin: str = None, add_baggage: int = 0, new_payment_method: str = None):
        """
        Modify a flight reservation, including changing the flight number, cabin class, and additional baggage, and handling payment method for balance updates if required.

        Args:
            user_id (str): The ID of the user making the modification.
            reservation_id (str): The reservation ID of the booking to be modified.
            new_flight_no (str, optional): The new flight number to change to (optional).
            new_cabin (str, optional): The new cabin class to change to (optional).
            add_baggage (int, optional): The number of additional checked baggage items to add (default is 0).
            new_payment_method (str, optional): Optional payment method for any additional charges (either 'cash' or 'bank'). If not provided, the original payment method will be used.
        """
        # Get corresponding reservation
        reservation = next((r for r in self.reservations if r['reservation_id'] == reservation_id and r['user_id'] == user_id), None)
        if not reservation:
            return "Reservation not found or user ID does not match."

        # Check current flight information
        current_flight = next((f for f in self.flights if f['flight_no'] == reservation['flight_no']), None)
        if not current_flight:
            return "Flight information not found."

        # Get original payment method or new provided payment method
        payment_method = new_payment_method if new_payment_method else reservation['payment_method']
        user = self.users[user_id]
        if not user:
            return "User information not found."

        # Store processing results
        result_messages = []

        # Update flight number if provided and matches origin/destination
        if new_flight_no and new_flight_no != reservation['flight_no']:
            new_flight = next((f for f in self.flights if f['flight_no'] == new_flight_no), None)
            if new_flight and new_flight['origin'] == current_flight['origin'] and new_flight['destination'] == current_flight['destination']:
                reservation['flight_no'] = new_flight_no
                result_messages.append("Flight number has been changed.")
            else:
                return f"Flight change failed: Invalid new flight number or destination does not match."

        # Update cabin if provided and calculate price difference
        if new_cabin and new_cabin != reservation.get('cabin'):
            price_difference = self.calculate_price_difference(current_flight, reservation['cabin'], new_cabin)
            reservation['cabin'] = new_cabin
            if price_difference > 0:
                # Deduct price difference
                if self.update_balance(user, payment_method, -price_difference):
                    result_messages.append(f"Cabin change successful. Price difference paid: {price_difference}.")
                else:
                    result_messages.append("Insufficient balance to pay the cabin price difference.")
            elif price_difference < 0:
                # Refund
                self.update_balance(user, payment_method, -price_difference)
                result_messages.append(f"Cabin change successful. Price difference refunded: {-price_difference}.")

        # Add checked baggage, check free allowance and calculate fees
        if add_baggage > 0:
            membership = user["membership_level"]
            max_free_baggage = self.get_baggage_allowance(membership, reservation['cabin'])
            current_baggage = reservation.get('baggage', 0)
            total_baggage = current_baggage + add_baggage
            extra_baggage = max(0, total_baggage - max_free_baggage)
            baggage_cost = extra_baggage * 50
            if baggage_cost > 0:
                # Deduct baggage fees
                if self.update_balance(user, payment_method, -baggage_cost):
                    result_messages.append(f"Baggage has been added. Additional fee to be paid: {baggage_cost}.")
                else:
                    result_messages.append("Insufficient balance to pay the additional baggage fees.")
            reservation['baggage'] = total_baggage

        # Return final result
        if not result_messages:
            result_messages.append("Modification completed with no additional fees.")
        return " ".join(result_messages)



    def cancel_reservation(self, user_id: str, reservation_id: str, reason: str):
        """
        Cancel a flight reservation based on reservation ID and reason for cancellation.

        Args:
            user_id (str): The ID of the user who made the reservation.
            reservation_id (str): The ID of the reservation to cancel.
            reason (str): Reason for cancellation (e.g., 'schedule change', 'airline canceled', 'other').
        """
        # Set the default current time to July 14, 2024, 6:00 AM
        current_time = datetime(2024, 7, 14, 6, 0, 0)

        # Verify that the user and reservation exist
        user = self.users.get(user_id, None)
        if not user:
            return "Invalid user ID."

        reservation = next((r for r in self.reservations if r["reservation_id"] == reservation_id and r["user_id"] == user_id), None)
        if not reservation:
            return "Invalid reservation ID or it does not belong to the user."

        # Check if flight information exists
        flight = next((f for f in self.flights if f["flight_no"] == reservation["flight_no"]), None)
        if not flight:
            return "Invalid flight information."

        # Check if the flight has already departed
        depart_time = datetime.strptime(flight["depart_time"], "%Y-%m-%d %H:%M:%S")
        if current_time > depart_time:
            return "The flight segment has been used and cannot be canceled."

        # Calculate the time until departure
        time_until_departure = depart_time - current_time
        cancel_fee = 0
        refund_amount = 0

        # Get the flight price
        flight_price = flight["economy_price"] if reservation["cabin"] == "Economy Class" else flight["business_price"]

        # Cancellation policy and refund calculation
        if reason == "The airline has canceled the flight.":
            # Airline cancels the flight, full refund
            refund_amount = flight_price
            self.process_refund(user, refund_amount)
            return f"The flight has been canceled. Your reservation will be canceled free of charge, and {refund_amount} yuan has been refunded."

        elif time_until_departure > timedelta(days=1):
            # More than 24 hours before departure, free cancellation
            refund_amount = flight_price
            self.process_refund(user, refund_amount)
            return f"More than 24 hours before departure. Free cancellation successful, {refund_amount} yuan has been refunded."

        else:
            # If not eligible for free cancellation, set a cancellation fee as needed
            cancel_fee = flight_price * 0.1  # Assume a cancellation fee of 10% of the ticket price
            refund_amount = flight_price - cancel_fee
            self.process_refund(user, refund_amount)
            return f"Less than 24 hours before departure. A cancellation fee of {cancel_fee} yuan has been deducted, and {refund_amount} yuan has been refunded."



    def process_refund(self, user, amount):
        """
        Add refund amount to user's cash balance.
        """
        user["cash_balance"] += amount
        print(f"Refund has been successfully processed. {user['user_name']}'s cash balance has been increased by {amount} yuan.")


    def calculate_price_difference(self, flight, old_cabin, new_cabin):
        """
        Calculate the price difference between different cabin classes
        """
        # 创建舱位价格映射字典，将舱位名称与航班数据中的价格字段关联
        cabin_prices = {
            "Economy Class": flight["economy_price"],
            "Business Class": flight["business_price"]
        }
        # 获取旧舱位对应的价格，若舱位不存在则返回默认值0
        old_price = cabin_prices.get(old_cabin, 0)
        # 获取新舱位对应的价格，若舱位不存在则返回默认值0
        new_price = cabin_prices.get(new_cabin, 0)
        # 计算新旧舱位价格差值并返回结果
        return new_price - old_price
