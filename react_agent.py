"""A ReAct agent written from first principles: Thought -> Action -> Observation -> ... -> Final Answer.

Exercise 1.4: complete parse_step().
Exercise 1.5: complete ReActAgent.run().

Run:  python react_agent.py "How many kilometres can robot Atlas travel on a full battery?"
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field

from config import MAX_STEPS
from tools import TOOLS, Tool

FORMAT_REMINDER = (
    "Your reply did not follow the format. Reply with 'Thought:', then either "
    "'Action:' and 'Action Input:' (a JSON object), or 'Final Answer:'."
)


def build_system_prompt(tools: dict[str, Tool]) -> str:
    tool_lines = "\n".join(t.signature() for t in tools.values())
    names = ", ".join(tools)
    return f"""You answer questions about a fleet of delivery robots.
You do not know the robot data and you must not guess numbers: use the tools, and use the calculator for every calculation.

You can use these tools:
{tool_lines}

Always use exactly this format:
Thought: what you need to do next
Action: one tool name from [{names}]
Action Input: the arguments as a JSON object
Then STOP and wait. The result will be sent to you as "Observation: ...".
Repeat Thought/Action/Action Input as many times as needed. When you know the answer, write:
Thought: I now know the final answer
Final Answer: a single number without units, or a single word

Example:
Question: A robot drives at 12 km/h from the North depot to the Port depot. How many minutes does the trip take?
Thought: I need the distance between North and Port.
Action: get_distance
Action Input: {{"origin": "North", "destination": "Port"}}
Observation: 6
Thought: 6 km at 12 km/h, so the time in minutes is 6 / 12 * 60.
Action: calculator
Action Input: {{"expression": "6 / 12 * 60"}}
Observation: 30
Thought: I now know the final answer
Final Answer: 30"""


# ---------------------------------------------------------------- parsing
@dataclass
class Step:
    thought: str = ""
    action: str | None = None
    action_input: dict | str | None = None
    final_answer: str | None = None


class ParseError(ValueError):
    pass


def _parse_json_object(raw: str) -> dict | str:
    """Return the first JSON object found in raw, or the stripped raw text if there is none."""
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            value = json.loads(raw[start : end + 1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return raw.strip().strip("\"'`")


def parse_step(text: str) -> Step:
    """Parse one model reply into a Step.

    Rules:
      * 'Thought:' text goes into step.thought (may be missing).
      * If an 'Action:' line appears BEFORE any 'Final Answer:', the step is an action:
        step.action = the tool name, step.action_input = the JSON object after 'Action Input:'
        (use _parse_json_object; if 'Action Input:' is missing, use {}).
      * Otherwise, if 'Final Answer:' appears, step.final_answer = the text after it (first line only).
      * If there is neither an action nor a final answer, raise ParseError.
    """
    # TODO 1.4: implement the rules above. Regular expressions (re.search) make this short.
    # HINT: re.search(r"Action\s*:\s*([A-Za-z_]\w*)", text) finds the tool name
    # HINT: compare match.start() of the action and of the final answer to know which comes first
    raise NotImplementedError("TODO 1.4 (see the comment above)")


# ---------------------------------------------------------------- the agent
@dataclass
class AgentResult:
    question: str
    answer: str | None
    steps: list[dict] = field(default_factory=list)
    tool_calls: int = 0
    seconds: float = 0.0


class ReActAgent:
    def __init__(self, llm, tools: dict[str, Tool] = TOOLS, max_steps: int = MAX_STEPS, verbose: bool = True):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.verbose = verbose
        self.system_prompt = build_system_prompt(tools)

    def log(self, text: str) -> None:
        if self.verbose:
            print(text)

    def call_tool(self, name: str, args: dict | str) -> str:
        """Run a tool and return its result as a string. Never raises: errors become observations."""
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}. Available tools: {', '.join(self.tools)}"
        if isinstance(args, str):  # the model wrote a bare value instead of a JSON object
            args = {next(iter(tool.parameters)): args} if len(tool.parameters) == 1 else {}
        try:
            return tool(**args)
        except TypeError as exc:
            return f"Error: wrong arguments for {name}: {exc}. Expected: {list(tool.parameters)}"

    def run(self, question: str) -> AgentResult:
        """Answer one question with the Thought/Action/Observation loop.

        1. Start the conversation with the system prompt and the user message 'Question: ...'.
        2. Repeat at most self.max_steps times:
           a. ask the model for the next step, stopping at 'Observation:'
           b. parse it (parse_step); on ParseError send FORMAT_REMINDER as the next user message
           c. if it is a final answer, stop and return it
           d. otherwise run the tool (self.call_tool) and send 'Observation: <result>' as the next user message
        3. If no final answer was given within max_steps, return answer=None.
        Record every step in result.steps (thought, action, input, observation) and count the tool calls.
        """
        result = AgentResult(question=question, answer=None)
        start = time.perf_counter()
        # TODO 1.5: implement the loop described in the docstring.
        # HINT: messages = [{"role": "system", ...}, {"role": "user", "content": f"Question: {question}"}]
        # HINT: reply = self.llm.chat(messages, stop=["Observation:"]); then append {"role": "assistant", "content": reply}
        raise NotImplementedError("TODO 1.5 (see the comment above)")
        result.seconds = round(time.perf_counter() - start, 2)
        return result


if __name__ == "__main__":
    from llm import LocalLLM

    question = " ".join(sys.argv[1:]) or "How many kilometres can robot Atlas travel on a full battery?"
    agent = ReActAgent(LocalLLM())
    res = agent.run(question)
    print(f"\nAnswer: {res.answer}   ({res.tool_calls} tool calls, {res.seconds} s)")
