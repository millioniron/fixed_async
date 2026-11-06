"""
tu_pipeline_env.py - Universal dynamic environment loader for TU-Pipeline data.

Integrates heterogeneous simulation environments from TU-Pipeline into
ROLL-infer as a single GEM environment. Each reset() dynamically loads a
different environment .py file based on the dataset record.

Key features:
- Dynamic environment class loading with importlib + caching
- Composite-environment aware class selection for step6_2 bridge files
- Tool registration from pre-built tool_schemas or fallback introspection
- Dual verification: watched_fields state comparison + answer verification
- Compatible with travel_env's step/reset/reward interface
"""


import importlib
import importlib.util
import inspect as _inspect
import json
import logging
import os
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

import ray
from datasets import Dataset
from gem import Env

from roll.datasets.global_dataset import GlobalDataset, GlobalDatasetManager
from roll.utils.constants import RAY_NAMESPACE

logger = logging.getLogger(__name__)


def _try_parse_num(s: str) -> Any:
    """Try to parse *s* as int or float; return the original string on failure."""
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def _summarize_for_log(value: Any, limit: int = 300) -> str:
    """Return a short, log-safe preview of *value*."""
    text = value if isinstance(value, str) else repr(value)
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


class TUPipelineEnv(Env):
    """Universal dynamic environment loader for TU-Pipeline data.

    Each reset() fetches a task record from the dataset, dynamically loads the
    corresponding environment .py file, instantiates it, registers its methods
    as available tools, and uses watched_fields + answer_verification for
    dual reward verification.
    """

    def __init__(
        self,
        dataset_name: str = "",
        env_base_dir: str = "",
        split: str = "train",
        dataset: Optional[Dataset] = None,
        question_key: str = "question",
        ground_truth_state_key: str = "ground_truth_state",
        watched_fields_key: str = "watched_fields",
        ignored_fields_key: str = "ignored_fields",
        answer_verification_key: str = "answer_verification",
        tool_schemas_key: str = "tool_schemas",
        env_code_file_key: str = "env_code_file",
        environment_id_key: str = "environment_id",
        state_before_key: str = "state_before",
        task_category_key: str = "task_category",
        max_steps: int = 100,
        seed: int = 0,
        mode: str = "train",
        format_penalty: float = 0.0,
        **_,
    ):
        super().__init__()
        self.seed = seed
        self.mode = mode
        self.max_steps = max_steps
        self.format_penalty = format_penalty
        self.env_base_dir = env_base_dir

        # Data field key mappings
        self.question_key = question_key
        self.ground_truth_state_key = ground_truth_state_key
        self.watched_fields_key = watched_fields_key
        self.ignored_fields_key = ignored_fields_key
        self.answer_verification_key = answer_verification_key
        self.tool_schemas_key = tool_schemas_key
        self.env_code_file_key = env_code_file_key
        self.environment_id_key = environment_id_key
        self.state_before_key = state_before_key
        self.task_category_key = task_category_key

        # Ray dataset
        global_dataset_mode = "sample" if self.mode == "train" else "traversal"
        self.dataset = GlobalDataset.options(
            name=f"{self.mode}_{dataset_name}",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote(dataset_name=dataset_name, split=split, mode=global_dataset_mode)
        self.dataset_manager = GlobalDatasetManager.options(
            name=f"{self.mode}_dataset_manager",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote()
        ray.get(self.dataset_manager.register.remote(
            dataset_name=dataset_name, dataset_ref=self.dataset
        ))

        # === Runtime state ===
        self._env_class_cache: Dict[str, type] = {}
        self._current_env = None
        self.available_tools: Dict[str, Callable] = {}
        self.current_tool_schemas: List[Dict] = []

        # Per-episode verification info
        self.ground_truth_state = None
        self.watched_fields: List[str] = []
        self.ignored_fields: List[str] = []
        self.answer_verification: Dict[str, Any] = {}
        self.task_category = ""

        self.step_count = 0
        self.render_cache = None
        self.env_instruction = ""
        self.first_obs = ""
        self._last_llm_response = ""

    # ========== Environment Class Loading ==========

    def _load_env_class(self, env_id: str, code_file: str) -> type:
        """Dynamically load an environment class with caching."""
        if env_id in self._env_class_cache:
            return self._env_class_cache[env_id]

        abs_path = os.path.join(self.env_base_dir, code_file)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Environment file not found: {abs_path}")

        module_name = f"tu_env_{env_id}"
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        env_class = self._extract_main_class(module)
        if env_class is None:
            raise ValueError(f"No environment class found in {abs_path}")

        self._env_class_cache[env_id] = env_class
        return env_class

    def _extract_main_class(self, module) -> Optional[type]:
        """Extract the main environment class from a module.

        Skips Enum, dataclass, Exception, and other auxiliary classes.
        Selection priority:
        1. Composite/bridge environment classes (step6_2 style)
        2. Name contains 'Environment'
        3. Has get_state
        4. Last candidate
        """
        skip_bases = (
            "Enum", "IntEnum", "StrEnum", "Flag", "IntFlag",
            "Exception", "Error", "TypedDict", "NamedTuple", "ABC",
        )

        candidates = []
        for name, obj in _inspect.getmembers(module, _inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if name.startswith("_"):
                continue
            base_names = [b.__name__ for b in obj.__mro__[1:]]
            if any(sb in base_names for sb in skip_bases):
                continue
            candidates.append((name, obj))

        if not candidates:
            return None

        bridge_candidates = []
        for name, obj in candidates:
            if not name.lower().endswith("environment"):
                continue
            if self._looks_like_composite_environment(obj):
                bridge_candidates.append((name, obj))

        if bridge_candidates:
            if len(bridge_candidates) > 1:
                logger.warning(
                    "Multiple composite environment candidates found in %s: %s; using %s",
                    getattr(module, "__name__", "<unknown>"),
                    [name for name, _ in bridge_candidates],
                    bridge_candidates[0][0],
                )
            return bridge_candidates[0][1]

        for name, obj in candidates:
            if "environment" in name.lower():
                return obj

        for name, obj in candidates:
            if hasattr(obj, "get_state"):
                return obj

        return candidates[-1][1]

    @staticmethod
    def _looks_like_composite_environment(obj: type) -> bool:
        """Detect step6_2-style bridge classes without instantiating them."""
        try:
            init_src = _inspect.getsource(obj.__init__)
        except (OSError, TypeError):
            init_src = ""

        if "self.env_a" in init_src and "self.env_b" in init_src:
            return True

        for _, member in _inspect.getmembers(obj, _inspect.isfunction):
            try:
                src = _inspect.getsource(member)
            except (OSError, TypeError):
                continue
            if "self.env_a" in src and "self.env_b" in src:
                return True

        return False

    # ========== Tool Registration ==========

    # Only skip internal lifecycle methods that are used by the framework itself,
    # not by the agent.  All other public methods are exposed as tools — including
    # auth, manuals, system info, etc. — since trajectory data confirms they are
    # actively called during task execution.
    _SKIP_METHODS = frozenset({"get_state", "set_state", "reset", "__init__"})

    def _register_tools(self, env_instance: Any) -> None:
        """Register available tools by introspecting the environment instance.

        Always generates tool_schemas at runtime via introspection to ensure
        consistency across all records.  Only internal lifecycle methods
        (get_state/set_state/reset) are excluded.
        """
        from roll.pipeline.agentic.env.gz.utils import introspect_function_to_tool

        self.available_tools = {}
        for name in sorted(dir(env_instance)):
            if name.startswith("_") or name in self._SKIP_METHODS:
                continue
            method = getattr(env_instance, name)
            if callable(method):
                self.available_tools[name] = method

        self.current_tool_schemas = [
            introspect_function_to_tool(func) for func in self.available_tools.values()
        ]

    # ========== Core Interface: reset / step ==========

    def reset(self, seed) -> None:
        """Reset environment: load a new task from dataset, instantiate its environment."""
        self.step_count = 0
        self._last_judge_details: Dict[str, Any] = {}
        Env.reset(self, seed)
        self._last_llm_response = ""

        # 1. Fetch task data from GlobalDataset
        data: Optional[Dict] = ray.get(self.dataset.get_data_item.remote(seed=seed))
        if data is None:
            return None, None

        # 2. Parse task fields
        self.first_obs = data[self.question_key]
        self.task_category = data.get(self.task_category_key, "action")
        gts = data.get(self.ground_truth_state_key, {})
        self.ground_truth_state = json.loads(gts) if isinstance(gts, str) else gts
        wf = data.get(self.watched_fields_key, [])
        self.watched_fields = json.loads(wf) if isinstance(wf, str) else wf
        igf = data.get(self.ignored_fields_key, [])
        self.ignored_fields = json.loads(igf) if isinstance(igf, str) else igf
        av = data.get(self.answer_verification_key, {})
        self.answer_verification = json.loads(av) if isinstance(av, str) else av

        # 2.1. Inject answer format hint for tasks requiring answer verification
        if (
            self.task_category in ("query", "hybrid")
            and self.answer_verification.get("enabled", False)
        ):
            self.first_obs += "\n\n" + self._get_boxed_hint(self.answer_verification)

        # 3. Dynamically load environment class
        env_id = data[self.environment_id_key]
        self.environment_id = env_id
        code_file = data[self.env_code_file_key]
        try:
            env_class = self._load_env_class(env_id, code_file)
        except Exception as e:
            logger.error(f"Failed to load environment {env_id}: {e}")
            return None, None

        # 4. Instantiate environment
        try:
            self._current_env = env_class()
        except Exception as e:
            logger.error(f"Failed to instantiate environment {env_id}: {e}")
            return None, None

        # 5. Register tools (always via introspection)
        self._register_tools(self._current_env)

        # 6. Prepare render cache
        self.prepare_render_cache(self.first_obs)

        return self.render(), {
            "env_instruction": self.env_instruction,
            "tools": self.current_tool_schemas,
        }

    def step(self, action: str):
        """Execute one step: parse tool calls, execute on environment, verify success."""
        metrics_agg_mode = {
            "action_is_effective": "mean",
            "action_is_valid": "mean",
            "success": "last",
            "format_penalty": "mean",
        }
        self.step_count += 1
        self._last_llm_response = action

        # 1. Parse tool_call tags
        tool_calls = self.parse_action_tool(action)
        observations = []
        individual_results = []
        all_valid = True
        any_effective = False

        # 2. No valid tool calls
        if not tool_calls:
            # For query/hybrid tasks, the model may submit a final answer
            # (e.g. \boxed{}) without a tool call.  Check judge_success() so
            # that a correct answer can terminate the episode.
            if self.task_category in ("query", "hybrid"):
                success = self.judge_success()
            else:
                success = False

            if success:
                action_desc = f"At turn {self.step_count}, answer submitted."
                observation = action_desc
                reward = 1
            else:
                action_desc = f"At turn {self.step_count}, no valid tool call format detected."
                observation = action_desc + "The task is not yet resolved; please continue."
                reward = self.format_penalty

            terminated = bool(success)
            truncated = self.step_count >= self.max_steps
            metrics = {
                "action_is_effective": bool(success),
                "action_is_valid": bool(success),
                "success": success,
                "format_penalty": 0.0 if success else self.format_penalty,
            }
            info = {
                "metrics": metrics,
                "metrics_agg_mode": metrics_agg_mode,
                "action_desc": action_desc,
                "tool_calls": [],
                "use_tool": True,
                "current_state": self._get_current_state(),
                "judge_details": getattr(self, "_last_judge_details", {}),
            }
            return observation, reward, terminated, truncated, info

        # 3. Execute each tool call
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
                obs = f"[Tool {idx + 1}] Invalid tool call format."
                observations.append(obs)
                result["observation"] = obs
                individual_results.append(result)
                continue

            if not isinstance(tool_name, str) or not isinstance(parameters, dict):
                all_valid = False
                obs = f"[Tool {idx + 1}] Invalid tool call payload."
                observations.append(obs)
                result["observation"] = obs
                result["error"] = (
                    f"tool_name_type={type(tool_name).__name__}, "
                    f"parameters_type={type(parameters).__name__}"
                )
                individual_results.append(result)
                continue

            if tool_name not in self.available_tools:
                all_valid = False
                obs = f"[Tool {idx + 1}] Tool '{tool_name}' is not available."
                observations.append(obs)
                result["observation"] = obs
                individual_results.append(result)
                continue

            try:
                tool_func = self.available_tools[tool_name]
                output = tool_func(**parameters)
                obs = f"[Tool {idx + 1}: {tool_name}] {output}"
                observations.append(obs)
                result["observation"] = obs
                result["is_executed"] = True
                any_effective = True
            except Exception as e:
                all_valid = False
                obs = f"[Tool {idx + 1}: {tool_name}] Execution error: {str(e)}"
                observations.append(obs)
                result["observation"] = obs
                result["error"] = str(e)
                logger.warning("Tool execution error: %s", e)

            individual_results.append(result)

        # 4. Build observation
        observation = "\n".join(observations)
        action_desc = f"At turn {self.step_count}, executed {len(tool_calls)} tool(s)."

        # 5. Judge success and compute reward
        metrics = {
            "action_is_effective": any_effective,
            "action_is_valid": all_valid,
            "success": self.judge_success(),
            "format_penalty": 0.0 if all_valid else self.format_penalty,
        }

        info = {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode,
            "action_desc": action_desc,
            "tool_calls": individual_results,
            "use_tool": True,
            "current_state": self._get_current_state(),
            "judge_details": getattr(self, "_last_judge_details", {}),
        }

        terminated = metrics["success"]
        truncated = self.step_count >= self.max_steps
        reward_format_penalty = 0.0 if all_valid else self.format_penalty
        reward_success = metrics["success"]

        return observation, reward_success + reward_format_penalty, terminated, truncated, info

    # ========== Action Parsing ==========

    @staticmethod
    def _build_tool_call_result(
        parsed_action: str,
        tool_name: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        is_valid: bool = False,
    ) -> Dict[str, Any]:
        """Create a normalized tool-call parse result."""
        return {
            "tool_name": tool_name,
            "parsed_action": parsed_action,
            "parameters": parameters if isinstance(parameters, dict) else {},
            "is_valid": is_valid,
        }

    def _parse_standard_tool_call(self, match: str) -> Dict[str, Any]:
        """Parse and validate a single ``<tool_call>`` JSON payload."""
        parsed_action = f"<tool_call>{match}</tool_call>"

        try:
            tool_data = json.loads(match.strip())
        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse tool JSON: %s, error: %s",
                _summarize_for_log(match),
                e,
            )
            return self._build_tool_call_result(parsed_action=parsed_action)

        if not isinstance(tool_data, dict):
            logger.warning(
                "Tool payload must be a JSON object, got %s: %s",
                type(tool_data).__name__,
                _summarize_for_log(tool_data),
            )
            return self._build_tool_call_result(parsed_action=parsed_action)

        tool_name = tool_data.get("name", "")
        arguments = tool_data.get("arguments", {})
        is_valid = True

        if not isinstance(tool_name, str):
            logger.warning(
                "Tool name must be a string, got %s: %s",
                type(tool_name).__name__,
                _summarize_for_log(tool_name),
            )
            tool_name = ""
            is_valid = False
        else:
            tool_name = tool_name.strip()
            if not tool_name:
                is_valid = False

        if arguments is None:
            arguments = {}
        elif isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Failed to parse string tool arguments for %r: %s, error: %s",
                    tool_name,
                    _summarize_for_log(arguments),
                    e,
                )
                arguments = {}
                is_valid = False
            else:
                arguments = parsed_arguments

        if not isinstance(arguments, dict):
            logger.warning(
                "Tool arguments must be a JSON object, got %s for %r: %s",
                type(arguments).__name__,
                tool_name,
                _summarize_for_log(arguments),
            )
            arguments = {}
            is_valid = False

        return self._build_tool_call_result(
            parsed_action=parsed_action,
            tool_name=tool_name,
            parameters=arguments,
            is_valid=is_valid,
        )

    def parse_action_tool(self, text: str) -> List[Dict]:
        """Parse tool calls from LLM output.

        Supports two formats:
        1. Standard: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        2. MiniMax XML: <invoke name="..."><parameter name="...">value</parameter></invoke>
        """
        # Strip <think>...</think> blocks so that tool_call examples inside
        # the model's chain-of-thought reasoning are not executed.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # 1. Standard format: <tool_call>JSON</tool_call>
        pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            tool_calls = []
            for match in matches:
                tool_calls.append(self._parse_standard_tool_call(match))
            return tool_calls

        # 2. MiniMax XML format: <invoke name="..."><parameter>...</parameter></invoke>
        tool_calls = []
        for m in re.finditer(
            r'<invoke\s+name="([^"]+)">(.*?)</invoke>', text, re.DOTALL
        ):
            tool_name = m.group(1)
            params = {}
            for pm in re.finditer(
                r'<parameter\s+name="([^"]+)">(.*?)</parameter>',
                m.group(2), re.DOTALL,
            ):
                raw_val = pm.group(2).strip()
                try:
                    params[pm.group(1)] = json.loads(raw_val)
                except (json.JSONDecodeError, ValueError):
                    params[pm.group(1)] = _try_parse_num(raw_val)
            tool_calls.append({
                "tool_name": tool_name,
                "parsed_action": m.group(0),
                "parameters": params,
                "is_valid": True,
            })

        return tool_calls

    # ========== Success Verification ==========

    def judge_success(self) -> int:
        """Dual verification: state comparison + answer verification.

        Task category determines which checks must pass:
          - action: state check only
          - query:  answer check only
          - hybrid: both must pass
        """
        state_ok = True
        answer_ok = True

        # --- State verification ---
        needs_state_check = self.task_category in ("action", "hybrid")
        if needs_state_check and self.watched_fields and self._current_env is not None:
            field_details: Dict[str, Any] = {}
            try:
                current_state = self._normalize(self._current_env.get_state())
                ground_state = self._normalize(self.ground_truth_state)
                state_ok = self._compare_states(
                    current_state, ground_state,
                    self.watched_fields, self.ignored_fields
                )
                for fp in self.watched_fields:
                    if fp in self.ignored_fields:
                        continue
                    cv = self._get_nested_value(current_state, fp)
                    gv = self._get_nested_value(ground_state, fp)
                    field_details[fp] = {
                        "current": cv, "ground_truth": gv,
                        "match": self._values_equal(cv, gv),
                    }
            except Exception as e:
                import traceback as _tb
                field_details["_error"] = {
                    "exception": str(e),
                    "gts_type": type(self.ground_truth_state).__name__,
                    "gts_preview": repr(self.ground_truth_state)[:300],
                    "traceback": _tb.format_exc(),
                }
                state_ok = False
            self._last_judge_details: Dict[str, Any] = {
                "state_ok": state_ok, "fields": field_details,
            }

        # --- Answer verification ---
        needs_answer_check = self.task_category in ("query", "hybrid")
        av = self.answer_verification
        if needs_answer_check and av.get("enabled", False):
            agent_answer = self._extract_answer(self._last_llm_response)
            if agent_answer is not None:
                answer_ok = self._verify_answer(agent_answer, av)
            else:
                answer_ok = False

        return 1 if state_ok and answer_ok else 0

    # ========== State Comparison Engine (aligned with step5_4.py) ==========

    @staticmethod
    def _is_dynamic_value(val: Any) -> bool:
        """Detect dynamically generated values (UUID, timestamp, prefixed ID)."""
        if not isinstance(val, str):
            return False
        if re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            val, re.IGNORECASE,
        ):
            return True
        if re.match(r"^[a-zA-Z]{2,10}_[0-9a-f]{6,}$", val):
            return True
        # ISO 时间戳 — 仅匹配含微秒/毫秒的运行时生成值（如 datetime.now() 产生的）
        # 硬编码业务时间戳（如 "2024-01-20T10:00:00"）不应被视为动态值
        if re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\.\d{3,}", val):
            return True
        return False

    # Compiled patterns for bracket filter expressions
    _JSONPATH_FILTER_RE = re.compile(
        r"\[\?\(@\.(\w+)\s*==\s*'([^']*)'\)\]"   # string: [?(@.id=='val')]
        r"|"
        r"\[\?\(@\.(\w+)\s*==\s*([^'\)]+)\)\]"    # numeric: [?(@.deal_id==120)]
    )
    _KV_FILTER_RE = re.compile(r"\[(\w+)=([^\]]+)\]")
    _WILDCARD_BRACKET_RE = re.compile(r"\[\*\]|\[\]")
    _NUMERIC_IDX_RE = re.compile(r"\[(-?\d+)\]")

    @staticmethod
    def _split_path(path: str) -> List[str]:
        """Split *path* on ``.`` while respecting bracket nesting.

        ``deals[?(@.deal_id==120)].on_sale`` → ``["deals[?(@.deal_id==120)]", "on_sale"]``
        """
        segments: List[str] = []
        current: List[str] = []
        depth = 0
        for ch in path:
            if ch == "[":
                depth += 1
                current.append(ch)
            elif ch == "]":
                depth = max(depth - 1, 0)
                current.append(ch)
            elif ch == "." and depth == 0:
                if current:
                    segments.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            segments.append("".join(current))
        return segments

    @staticmethod
    def _get_nested_value(data: Any, path: str) -> Any:
        """Resolve a nested path against *data* (dict / list tree).

        Supported syntax:
        - dict key:       ``orders.status``
        - list index:     ``items[0]``, ``items[-1]``
        - JSONPath filter: ``items[?(@.id=='p14')]``
        - Key=Value:       ``acl[entity=user@example.com]``
        - Wildcard:        ``items[*].price`` (returns list)
        - Greedy key:      ``users.a@b.com`` (dot-containing dict keys)
        """
        if data is None:
            return None

        cls = TUPipelineEnv
        segments = cls._split_path(path)
        current = data
        i = 0

        while i < len(segments):
            if current is None:
                return None

            seg = segments[i]
            bracket_start = seg.find("[")

            if bracket_start != -1:
                # --- Segment with bracket(s) ---
                key = seg[:bracket_start] if bracket_start > 0 else None
                indices_part = seg[bracket_start:]

                # Look up the key portion in the current dict
                if key is not None:
                    if isinstance(current, dict):
                        if key in current:
                            current = current[key]
                        elif key.lstrip("-").isdigit():
                            current = current.get(int(key))
                        else:
                            current = None
                    else:
                        return None

                if current is None:
                    return None

                # (a) Wildcard [*] or [] — fan out remaining path to every element
                if cls._WILDCARD_BRACKET_RE.search(indices_part):
                    if not isinstance(current, list):
                        return None
                    remaining = segments[i + 1:]
                    if remaining:
                        rp = ".".join(remaining)
                        return [cls._get_nested_value(item, rp) for item in current]
                    return current

                # (b) JSONPath filter [?(@.field==value)]
                m = cls._JSONPATH_FILTER_RE.search(indices_part)
                if m:
                    if not isinstance(current, list):
                        current = None
                    else:
                        field = m.group(1) if m.group(1) else m.group(3)
                        raw = m.group(2) if m.group(1) else m.group(4)
                        value: Any = raw if m.group(1) else _try_parse_num(raw)
                        current = next(
                            (item for item in current
                             if isinstance(item, dict) and item.get(field) == value),
                            None,
                        )
                    i += 1
                    continue

                # (c) Key=Value filter [key=value]
                m = cls._KV_FILTER_RE.search(indices_part)
                if m and not cls._NUMERIC_IDX_RE.match(indices_part):
                    if not isinstance(current, list):
                        current = None
                    else:
                        fk, fv = m.group(1), m.group(2).strip("'\"")
                        current = next(
                            (item for item in current
                             if isinstance(item, dict) and str(item.get(fk, "")) == fv),
                            None,
                        )
                    i += 1
                    continue

                # (d) Normal numeric indices [0], [-1]
                for idx_match in cls._NUMERIC_IDX_RE.finditer(indices_part):
                    if current is None:
                        return None
                    idx = int(idx_match.group(1))
                    if isinstance(current, list):
                        try:
                            current = current[idx]
                        except IndexError:
                            return None
                    else:
                        return None
                i += 1
            else:
                # --- Pure key segment — greedy matching for dot-containing keys ---
                if isinstance(current, dict):
                    if seg in current:
                        current = current[seg]
                        i += 1
                    else:
                        combined = seg
                        found = False
                        for j in range(i + 1, len(segments)):
                            combined += "." + segments[j]
                            if combined in current:
                                current = current[combined]
                                i = j + 1
                                found = True
                                break
                        if not found:
                            if seg.lstrip("-").isdigit():
                                current = current.get(int(seg))
                            else:
                                current = None
                            i += 1
                else:
                    return None

        return current

    # Regex to strip all bracket index notations: [0], [-1], [], [*]
    _BRACKET_RE = re.compile(r"\[-?\d+\]|\[\]|\[\*\]")

    @staticmethod
    def _build_ignored_keys_for_path(
        field_path: str, ignored_fields: List[str]
    ) -> set:
        """Build the set of top-level dict keys to skip when comparing *field_path*.

        Handles:
          - Numeric indices: ``orders[-1].id`` → ``orders.id``
          - Empty brackets:  ``orders[].id``   → ``orders.id``
          - Star brackets:   ``items[*].name`` → ``items.name``
          - Wildcard ``*`` as path segment:    ``guilds.*.id`` matches ``guilds.guild_3``
        """
        ignored_keys: set = set()
        br = TUPipelineEnv._BRACKET_RE
        normalized_path = br.sub("", field_path)
        fp_segments = normalized_path.split(".")

        for ig in ignored_fields:
            suffix = None
            # 1. Exact prefix match (before normalization)
            if ig.startswith(field_path + "."):
                suffix = ig[len(field_path) + 1:]
            else:
                normalized_ig = br.sub("", ig)
                # 2. Normalized prefix match (brackets stripped)
                if normalized_ig.startswith(normalized_path + "."):
                    suffix = normalized_ig[len(normalized_path) + 1:]
                else:
                    # 3. Glob-style matching: * in path segments matches any key
                    ig_segments = normalized_ig.split(".")
                    if len(ig_segments) > len(fp_segments):
                        match = True
                        for j in range(len(fp_segments)):
                            if ig_segments[j] == "*" or fp_segments[j] == "*":
                                continue
                            if ig_segments[j] != fp_segments[j]:
                                match = False
                                break
                        if match:
                            suffix = ".".join(ig_segments[len(fp_segments):])

            if suffix:
                # Extract the first concrete (non-wildcard) key from the suffix.
                # e.g. "joined_at" → "joined_at"
                #      "*.joined_at" → "joined_at"  (skip wildcard levels)
                for part in suffix.split("."):
                    clean = br.sub("", part)
                    if clean and clean != "*":
                        ignored_keys.add(clean)
                        break

        return ignored_keys

    def _compare_states(
        self,
        current_state: Any,
        ground_truth_state: Any,
        watched_fields: List[str],
        ignored_fields: List[str],
    ) -> bool:
        """Compare states along watched_fields paths, excluding ignored_fields."""
        for field_path in watched_fields:
            if field_path in ignored_fields:
                continue

            current_val = self._get_nested_value(current_state, field_path)
            truth_val = self._get_nested_value(ground_truth_state, field_path)

            ignored_keys = self._build_ignored_keys_for_path(
                field_path, ignored_fields
            )

            if not self._values_equal(current_val, truth_val, ignored_keys):
                return False

        return True

    def _values_equal(self, a: Any, b: Any, ignored_keys: set = None) -> bool:
        """Recursively compare two values with type coercion and order tolerance."""
        if ignored_keys is None:
            ignored_keys = set()

        if a is None and b is None:
            return True
        if a is None or b is None:
            return False

        # Dynamic values (UUID, timestamp) — treat as equal
        if self._is_dynamic_value(a) and self._is_dynamic_value(b):
            return True

        # Numeric comparison (bool guard first — bool is subclass of int)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if isinstance(a, bool) or isinstance(b, bool):
                return a == b
            if isinstance(a, float) or isinstance(b, float):
                return abs(float(a) - float(b)) < 1e-9
            return a == b

        # String ↔ numeric cross-type
        if isinstance(a, str) and isinstance(b, (int, float)) and not isinstance(b, bool):
            try:
                return self._values_equal(float(a), b, ignored_keys)
            except (ValueError, TypeError):
                return False
        if isinstance(b, str) and isinstance(a, (int, float)) and not isinstance(a, bool):
            try:
                return self._values_equal(a, float(b), ignored_keys)
            except (ValueError, TypeError):
                return False

        if isinstance(a, str) and isinstance(b, str):
            return a.strip() == b.strip()

        # List — sort before comparing (order-independent)
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            try:
                sorted_a = sorted(a, key=lambda x: json.dumps(x, ensure_ascii=False, default=str))
                sorted_b = sorted(b, key=lambda x: json.dumps(x, ensure_ascii=False, default=str))
                return all(self._values_equal(ai, bi, ignored_keys) for ai, bi in zip(sorted_a, sorted_b))
            except (TypeError, ValueError):
                return all(self._values_equal(ai, bi, ignored_keys) for ai, bi in zip(a, b))

        # Dict — skip ignored_keys
        if isinstance(a, dict) and isinstance(b, dict):
            keys_a = set(a.keys()) - ignored_keys
            keys_b = set(b.keys()) - ignored_keys
            if keys_a != keys_b:
                return False
            return all(self._values_equal(a[k], b[k], ignored_keys) for k in keys_a)

        return a == b

    # ========== Answer Verification Engine ==========

    _LATEX_WRAP_RE = re.compile(
        r"^\\(?:text|textbf|textit|textrm|textsf|texttt|mathrm|mathbf|operatorname)\{(.+)\}$",
        re.DOTALL,
    )

    @classmethod
    def _strip_latex(cls, text: str) -> str:
        """Strip common LaTeX wrapper commands like \\text{...}, \\textbf{...}, etc."""
        changed = True
        while changed:
            changed = False
            m = cls._LATEX_WRAP_RE.match(text)
            if m:
                text = m.group(1).strip()
                changed = True
        return text

    @staticmethod
    def _get_boxed_hint(answer_verification: Dict[str, Any]) -> str:
        r"""Generate answer-type-specific \boxed{} format hint (aligned with step5_4)."""
        answer_type = answer_verification.get("answer_type", "single_value")
        hints = {
            "single_value": (
                "[IMPORTANT] Wrap your final answer in \\boxed{}, e.g. \\boxed{answer}."
            ),
            "number": (
                "[IMPORTANT] Wrap the numeric result in \\boxed{} (without units), e.g. \\boxed{1234.56}."
            ),
            "list": (
                "[IMPORTANT] Wrap the result list in \\boxed{} as a JSON array, "
                'e.g. \\boxed{["item1", "item2"]}.'
            ),
            "structured": (
                "[IMPORTANT] Output the result as a JSON object inside \\boxed{}. "
                "Since \\boxed{} uses curly braces as delimiters, you must add an extra pair of "
                'braces around the JSON. For example, for {"name": "Alice"}, '
                'write: \\boxed{{"name": "Alice"}}.'
            ),
        }
        return hints.get(answer_type, hints["single_value"])

    def _extract_answer(self, text: str) -> Optional[str]:
        r"""Extract answer from LLM response.

        Priority: \boxed{} → [ANSWER] tag.
        Post-processing: strip LaTeX wrappers, fix JSON brace collision.
        """
        # 1. Try \boxed{} (last occurrence)
        idx = text.rfind("\\boxed{")
        if idx != -1:
            start = idx + len("\\boxed{")
            depth = 1
            i = start
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            if depth == 0:
                content = text[start:i - 1].strip()
                content = self._strip_latex(content)
                # Fix JSON brace collision: \boxed{"key": "val"} extracts as "key": "val"
                if content.startswith('"') and '":' in content:
                    wrapped = "{" + content + "}"
                    try:
                        json.loads(wrapped)
                        content = wrapped
                    except (json.JSONDecodeError, ValueError):
                        pass
                return content

        # 2. Fallback: [ANSWER] tag
        answer_match = re.search(r"\[ANSWER\]\s*(.+)", text, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()

        return None

    def _verify_answer(self, agent_answer: str, verification: Dict[str, Any]) -> bool:
        """Verify answer using the configured match_strategy.

        Strategies: exact, json_kv, numeric_tolerance, set_match.
        """
        expected = verification.get("expected_answer", "")
        # Dataset records may store answers as numeric types; normalize them
        # before string-oriented matchers call .replace() / .strip().
        if not isinstance(expected, str):
            expected = str(expected)
        if not isinstance(agent_answer, str):
            agent_answer = str(agent_answer)
        strategy = verification.get("match_strategy", "exact")
        tolerance = verification.get("tolerance", 0.01)

        try:
            if strategy == "exact":
                return self._match_exact(agent_answer, expected)
            elif strategy == "json_kv":
                return self._match_json_kv(
                    agent_answer, expected, verification.get("required_keys", [])
                )
            elif strategy == "numeric_tolerance":
                return self._match_numeric(agent_answer, expected, tolerance)
            elif strategy == "set_match":
                return self._match_set(agent_answer, expected)
            else:
                logger.warning(f"Unknown match_strategy: {strategy}, falling back to exact")
                return self._match_exact(agent_answer, expected)
        except Exception as e:
            logger.warning(f"Answer verification error: {e}")
            return False

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace: convert literal escapes, collapse runs, strip."""
        text = text.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _match_exact(self, agent: str, expected: str) -> bool:
        """Exact match: try JSON semantic comparison, fall back to normalized string comparison."""
        try:
            return json.loads(agent) == json.loads(expected)
        except (json.JSONDecodeError, TypeError):
            return self._normalize_whitespace(agent).lower() == self._normalize_whitespace(expected).lower()

    def _match_json_kv(self, agent: str, expected: str, required_keys: List[str]) -> bool:
        """JSON key-value comparison."""
        try:
            agent_obj = json.loads(agent)
            expected_obj = json.loads(expected)
        except (json.JSONDecodeError, TypeError):
            return False

        if required_keys:
            for key in required_keys:
                if key not in agent_obj:
                    return False
                if not self._values_equal(agent_obj[key], expected_obj.get(key)):
                    return False
            return True

        return self._values_equal(agent_obj, expected_obj)

    def _match_numeric(self, agent: str, expected: str, tolerance: float) -> bool:
        """Numeric comparison with tolerance."""
        try:
            a = float(agent.strip())
            b = float(expected.strip())
            return abs(a - b) <= tolerance
        except (ValueError, TypeError):
            return False

    def _match_set(self, agent: str, expected: str) -> bool:
        """Unordered set comparison."""
        try:
            a = json.loads(agent)
            b = json.loads(expected)
            if isinstance(a, list) and isinstance(b, list):
                return sorted(a, key=str) == sorted(b, key=str)
            return a == b
        except (json.JSONDecodeError, TypeError):
            return False

    # ========== Helpers ==========

    @staticmethod
    def _extract_step_action(step: Dict[str, Any]) -> tuple:
        """Extract function name and parameters from a trajectory step.

        Handles the many key-naming conventions found across TU-Pipeline data:
        function name: function, function_name, action, step_description
        parameters:    parameters, params, function_params, arguments,
                       action_input, action_args, kwargs, args
        """
        func = (
            step.get("function")
            or step.get("function_name")
            or step.get("action")
            or step.get("step_description")
            or "?"
        )
        params = None
        for _pk in (
            "parameters", "params", "function_params", "arguments",
            "action_input", "action_args", "kwargs", "args",
        ):
            if _pk in step:
                params = step[_pk]
                break
        if not isinstance(params, dict):
            params = {}
        return func, params

    def replay_trajectory(
        self, trajectory: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Replay a full trajectory on the current environment.

        Args:
            trajectory: list of step dicts from the dataset.

        Returns:
            dict with keys: success (bool), executed (int), errors (list),
            final_state (dict).
        """
        errors: List[str] = []
        executed = 0

        for idx, step in enumerate(trajectory):
            func_name, params = self._extract_step_action(step)

            if func_name == "?" or func_name not in self.available_tools:
                errors.append(f"Step {idx}: tool '{func_name}' not available")
                continue

            try:
                self.available_tools[func_name](**params)
                executed += 1
            except Exception as e:
                errors.append(f"Step {idx}: {func_name}() raised {e}")

        success = bool(self.judge_success())
        return {
            "success": success,
            "executed": executed,
            "errors": errors,
            "final_state": self._get_current_state(),
        }

    def _get_current_state(self) -> Dict[str, Any]:
        """Get current environment state for info/debugging (mirrors travel_env's current_state)."""
        if self._current_env is not None:
            try:
                return self._current_env.get_state()
            except Exception:
                return {}
        return {}

    def render(self, mode=None):
        """Render the environment."""
        return self.render_cache

    def prepare_render_cache(self, observation: str) -> None:
        """Prepare the render cache for the environment."""
        self.render_cache = observation

    def _normalize(self, obj: Any) -> Any:
        """Recursively convert to JSON-safe types (datetime, Decimal, set, bytes, etc.)."""
        if isinstance(obj, dict):
            return {str(k): self._normalize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._normalize(v) for v in obj]
        elif isinstance(obj, set):
            return sorted([self._normalize(v) for v in obj], key=lambda x: str(x))
        elif isinstance(obj, float):
            if obj != obj:  # NaN
                return None
            if obj == float("inf") or obj == float("-inf"):
                return str(obj)
            return obj
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        elif isinstance(obj, (int, str, bool)) or obj is None:
            return obj
        else:
            return repr(obj)
