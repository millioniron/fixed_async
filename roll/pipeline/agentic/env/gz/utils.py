
# 初始化tokenizer
import json
import inspect
import typing
import re
from typing import get_origin, get_args, Union, Dict, Any

import inspect
import re
from typing import Any, Dict, get_origin, get_args

def extract_function_description(docstring: str) -> str:
    """
    Extract only the main description from a Google/NumPy-style docstring,
    excluding the Args:, Returns:, etc. sections.
    """
    if not docstring:
        return "No description provided."

    # Clean and split into lines
    lines = inspect.cleandoc(docstring).splitlines()
    description_lines = []

    for line in lines:
        # Stop when we hit a section header like "Args:", "Parameters:", "Returns:", etc.
        if re.match(r"^(Args?|Parameters?|Returns?|Raises?|Examples?):\s*$", line.strip(), re.IGNORECASE):
            break
        description_lines.append(line)

    return " ".join(description_lines).strip() or "No description provided."


def _type_to_json_schema_type(python_type):
    """Map Python types to JSON Schema types."""
    from typing import Any
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }
    origin = get_origin(python_type)
    if origin is not None:
        if origin is list:
            return "array"
        elif origin is dict:
            return "object"
        elif origin is Union:
            args = get_args(python_type)
            for arg in args:
                if arg is not type(None):
                    return _type_to_json_schema_type(arg)
    return type_map.get(python_type, "object")


def _parse_param_descriptions(docstring: str) -> Dict[str, str]:
    """Parse parameter descriptions from Google-style docstring."""
    if not docstring:
        return {}

    descriptions = {}
    lines = docstring.splitlines()
    in_args_section = False

    for line in lines:
        stripped = line.strip()
        if re.match(r"^(Args?|Parameters?):$", stripped, re.IGNORECASE):
            in_args_section = True
            current_param = None
            continue
        if in_args_section:
            # Match: "param_name (type): description" or "param_name: description"
            match = re.match(r"^(\w+)\s*(?:\(.*?\))?\s*:\s*(.*)", stripped)
            if match:
                current_param, desc = match.groups()
                descriptions[current_param] = desc.strip()
            elif stripped == "":
                continue
            elif current_param and line.startswith(("    ", "\t")):
                # Continuation line for current param
                descriptions[current_param] += " " + stripped
            elif re.match(r"^(Returns?|Raises?|Examples?|Notes?):", stripped, re.IGNORECASE):
                break

    return descriptions


def introspect_function_to_tool(func) -> Dict[str, Any]:
    """
    Introspects a method/function and generates an OpenAI-style tool schema.
    Assumes Google-style docstring and type hints are present.
    """
    sig = inspect.signature(func)
    docstring = func.__doc__ or ""

    # Extract clean function description (no Args section)
    func_description = extract_function_description(docstring)

    # Parse parameter descriptions
    param_descs = _parse_param_descriptions(docstring)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue

        # Type
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
        json_type = _type_to_json_schema_type(param_type)

        # Required?
        is_required = param.default == inspect.Parameter.empty

        # Description
        description = param_descs.get(param_name, "No description provided.")

        # Build property
        prop = {"type": json_type, "description": description}

        # Handle Literal enum
        origin = get_origin(param_type)
        if origin is not None and "Literal" in str(origin):
            prop["enum"] = list(get_args(param_type))

        properties[param_name] = prop
        if is_required:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }