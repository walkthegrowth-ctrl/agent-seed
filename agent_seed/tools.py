"""Tool registry and OpenAI function-schema generation.

A tool is a plain Python function decorated with @tool. The schema is
derived from its name, docstring, and type hints — no pydantic needed.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints


class ToolError(RuntimeError): ...


@dataclass
class ToolSpec:
    """Registered tool: executor + LLM-facing schema."""

    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict[str, Any]

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_PY_TO_JSON: dict[type, str] = {
    str: "string",
    int: "number",
    float: "number",
    bool: "boolean",
}

_ARITH: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def type_to_schema(tp: Any, name: str = "") -> dict[str, Any]:
    """Convert a Python annotation to a JSON-schema fragment.

    ``name`` is the parameter being mapped. It is accepted for future
    name-aware extensions and a stable public signature; the mapping today is
    purely type-driven and ignores it.
    """
    import typing

    origin = typing.get_origin(tp)
    if origin is None:
        if tp in _PY_TO_JSON and isinstance(tp, type):
            return {"type": _PY_TO_JSON[tp]}
        if tp is dict or (isinstance(tp, type) and issubclass(tp, dict)):
            return {"type": "object"}
        return {}
    if origin is list:
        args = typing.get_args(tp)
        item = args[0] if args else Any
        return {"type": "array", "items": type_to_schema(item)}
    if origin is dict:
        return {"type": "object"}
    if origin is type(None):
        return {}
    if tp is Any:
        return {}
    return {}


def build_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build the OpenAI function-schema for a decorated tool function."""
    doc = fn.__doc__ or ""
    lines = doc.strip().splitlines()
    description = lines[0].strip() if lines else fn.__name__

    try:
        hints = {k: v for k, v in get_type_hints(fn).items() if k != "return"}
    except (TypeError, NameError):  # pragma: no cover - annotation resolution
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, param_type in hints.items():
        properties[param_name] = type_to_schema(param_type)
        required.append(param_name)

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": parameters,
        },
    }


class ToolRegistry:
    """Holds the set of callable tools exposed to the LLM."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        spec = ToolSpec(
            name=fn.__name__,
            description=(fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else fn.__name__,
            fn=fn,
            parameters=build_schema(fn)["function"]["parameters"],
        )
        self._tools[spec.name] = spec
        return fn

    def tool(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator form: @registry.tool"""
        return self.register(fn)

    def get(self, name: str) -> ToolSpec | None:
        """Public lookup: the :class:`ToolSpec` for ``name``, or ``None``.

        Public API for external/programmatic use (dispatch, schema access). The
        core's tool-policy path uses :meth:`schemas_for` / :meth:`call` instead.
        """
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.schema for spec in self._tools.values()]

    def has(self, name: str) -> bool:
        """True if a tool with ``name`` is registered."""
        return name in self._tools

    def schemas_for(self, names: list[str] | None) -> list[dict[str, Any]]:
        """Schemas for the names in ``names`` (None or empty -> []).

        Registration order is preserved. Names that are not registered are
        ignored (they simply do not exist to offer). This is the tool-policy
        filter the core applies against ``ctx.allowed_tools``: a strict
        whitelist — whatever is not named is neither offered nor callable.
        """
        if not names:
            return []
        return [spec.schema for spec in self._tools.values() if spec.name in names]

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        spec = self._tools.get(name)
        if spec is None:
            raise ToolError(f"unknown tool: {name!r}")
        kwargs = {k: v for k, v in (arguments or {}).items() if k in spec.parameters["properties"]}
        return spec.fn(**kwargs)


# ---------------------------------------------------------------------------
# starter tools
# ---------------------------------------------------------------------------


def _ast_value(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ToolError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITH:
        try:
            return _ARITH[type(node.op)](_ast_value(node.left), _ast_value(node.right))
        except ZeroDivisionError as exc:
            raise ToolError("division by zero") from exc
        except OverflowError as exc:
            raise ToolError("value overflow") from exc
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _ast_value(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ToolError(f"unsupported expression node: {type(node).__name__}")


def calculator(expression: str) -> float:
    """Evaluate an arithmetic expression (numbers, + - * / // % **)."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {expression!r}") from exc
    result = _ast_value(tree.body)
    if math.isnan(result) or math.isinf(result):
        raise ToolError("result is not a finite number")
    return result


# The default registry ships with the starter tools.
registry = ToolRegistry()
registry.register(calculator)
