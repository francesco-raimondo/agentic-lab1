"""Tools the agent can call. Each tool is a plain Python function that returns a string.

Exercise 1.3: complete the safe calculator (_eval_node).
"""
from __future__ import annotations

import ast
import json
import math
import operator
from dataclasses import dataclass, field
from typing import Callable

from config import DATA_DIR

FLEET = json.loads((DATA_DIR / "fleet.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- tool definition
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)  # argument name -> description
    func: Callable[..., str] = print

    def __call__(self, **kwargs) -> str:
        return self.func(**kwargs)

    def signature(self) -> str:
        """One line describing the tool, used inside the agent's system prompt."""
        args = ", ".join(self.parameters)
        details = "; ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"- {self.name}({args}): {self.description}" + (f" Arguments: {details}." if details else "")


# ---------------------------------------------------------------- calculator
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {"round": round, "min": min, "max": max, "abs": abs, "sqrt": math.sqrt}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate an arithmetic expression tree.

    Allowed: numbers, + - * / // % **, unary + and -, and the functions in _FUNCTIONS.
    Anything else (names, attributes, strings, imports...) must raise ValueError.
    """
    # TODO 1.3: evaluate the node. Handle ast.Constant, ast.BinOp, ast.UnaryOp and ast.Call; reject everything else.
    # HINT: for ast.BinOp use _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    # HINT: for ast.Call check that node.func is an ast.Name whose id is in _FUNCTIONS
    raise NotImplementedError("TODO 1.3 (see the comment above)")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(str(expression).replace("^", "**").replace("×", "*"), mode="eval")
        value = _eval_node(tree.body)
    except ZeroDivisionError:
        return "Error: division by zero"
    except (SyntaxError, ValueError, TypeError) as exc:
        return f"Error: cannot evaluate {expression!r} ({exc})"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    elif isinstance(value, float):
        value = round(value, 6)
    return str(value)


# ---------------------------------------------------------------- fleet tools
def list_robots() -> str:
    return ", ".join(FLEET["robots"])


def get_robot(name: str) -> str:
    key = str(name).strip().strip("\"'").capitalize()
    robot = FLEET["robots"].get(key)
    if robot is None:
        return f"Error: no robot called {name!r}. Known robots: {list_robots()}"
    return json.dumps({"name": key, **robot})


def get_distance(origin: str, destination: str) -> str:
    a = str(origin).replace("depot", "").strip().capitalize()
    b = str(destination).replace("depot", "").strip().capitalize()
    if a == b:
        return "0"
    for key in (f"{a}-{b}", f"{b}-{a}"):
        if key in FLEET["distances_km"]:
            return str(FLEET["distances_km"][key])
    depots = sorted({d for k in FLEET["distances_km"] for d in k.split("-")})
    return f"Error: unknown depot pair {origin!r}, {destination!r}. Depots: {', '.join(depots)}"


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("calculator",
             "Evaluates an arithmetic expression and returns the number. Supports + - * / ** % and round(), min(), max().",
             {"expression": "the expression, e.g. \"(1200 - 720) / 1200 * 100\""}, calculator),
        Tool("list_robots", "Returns the names of all the robots in the fleet.", {}, list_robots),
        Tool("get_robot",
             "Returns the specification of one robot as JSON: battery_wh, consumption_wh_per_km, max_payload_kg, speed_kmh, depot.",
             {"name": "the robot name, e.g. \"Borea\""}, get_robot),
        Tool("get_distance", "Returns the road distance in km between two depots (North, Port, Campus, Hospital).",
             {"origin": "first depot", "destination": "second depot"}, get_distance),
    ]
}
