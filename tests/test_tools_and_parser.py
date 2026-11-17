"""Run with:  python -m pytest -q
These tests do not need the language model, so they are fast.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from react_agent import ParseError, ReActAgent, parse_step  # noqa: E402
from tools import calculator, get_distance, get_robot  # noqa: E402


# ---------------------------------------------------------------- Exercise 1.3
def test_calculator_basic():
    assert calculator("37 * 48 + 112") == "1888"
    assert calculator("(1200 - 720) / 1200 * 100") == "40"
    assert calculator("2 ** 10") == "1024"
    assert calculator("-3 + 5") == "2"
    assert calculator("round(10 / 3, 2)") == "3.33"
    assert calculator("max(3, 9, 4)") == "9"


def test_calculator_rejects_code():
    for dangerous in ["__import__('os').system('ls')", "open('x')", "a + 1", "'abc' * 3", "(1).__class__"]:
        assert calculator(dangerous).startswith("Error"), dangerous


def test_calculator_errors():
    assert calculator("1 / 0") == "Error: division by zero"
    assert calculator("2 +").startswith("Error")


# ---------------------------------------------------------------- fleet tools (already implemented)
def test_fleet_tools():
    assert json.loads(get_robot("cirrus"))["max_payload_kg"] == 60
    assert get_distance("Hospital", "Port") == "7"
    assert get_robot("Zephyr").startswith("Error")


# ---------------------------------------------------------------- Exercise 1.4
def test_parse_action():
    step = parse_step('Thought: I need Atlas.\nAction: get_robot\nAction Input: {"name": "Atlas"}')
    assert step.thought == "I need Atlas."
    assert step.action == "get_robot"
    assert step.action_input == {"name": "Atlas"}
    assert step.final_answer is None


def test_parse_final_answer():
    step = parse_step("Thought: I now know the final answer\nFinal Answer: 15")
    assert step.final_answer == "15"
    assert step.action is None


def test_parse_action_wins_if_first():
    text = 'Thought: x\nAction: calculator\nAction Input: {"expression": "1+1"}\nFinal Answer: 2'
    assert parse_step(text).action == "calculator"


def test_parse_bare_input_and_missing_input():
    assert parse_step("Action: get_robot\nAction Input: Atlas").action_input == "Atlas"
    assert parse_step("Thought: list them\nAction: list_robots").action_input == {}


def test_parse_error():
    try:
        parse_step("I think the answer is probably 15.")
    except ParseError:
        return
    raise AssertionError("ParseError was not raised")


# ---------------------------------------------------------------- Exercise 1.5 (with a scripted fake model)
class FakeLLM:
    """Plays back prepared replies instead of running a real model."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(list(messages))  # copy: the agent keeps appending to its list
        return self.replies.pop(0)


def test_agent_loop_uses_tools_and_stops():
    fake = FakeLLM([
        'Thought: I need Atlas.\nAction: get_robot\nAction Input: {"name": "Atlas"}',
        'Thought: 1200 / 80.\nAction: calculator\nAction Input: {"expression": "1200 / 80"}',
        "Thought: I now know the final answer\nFinal Answer: 15",
    ])
    result = ReActAgent(fake, verbose=False).run("How far can Atlas go?")
    assert result.answer == "15"
    assert result.tool_calls == 2
    last_messages = fake.calls[-1]
    assert last_messages[-1] == {"role": "user", "content": "Observation: 15"}
    assert last_messages[0]["role"] == "system"


def test_agent_recovers_from_format_error_and_unknown_tool():
    fake = FakeLLM([
        "The answer is 15.",
        'Thought: try\nAction: fly\nAction Input: {}',
        "Thought: I now know the final answer\nFinal Answer: 15",
    ])
    result = ReActAgent(fake, verbose=False).run("q")
    assert result.answer == "15"
    assert any("did not follow the format" in m["content"] for m in fake.calls[1] if m["role"] == "user")
    assert "unknown tool" in result.steps[1]["observation"]


def test_agent_gives_up_after_max_steps():
    fake = FakeLLM(['Thought: again\nAction: list_robots\nAction Input: {}'] * 3)
    result = ReActAgent(fake, max_steps=3, verbose=False).run("q")
    assert result.answer is None and result.tool_calls == 3
