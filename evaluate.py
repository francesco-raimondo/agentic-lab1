"""Exercise 1.6: does the agent beat the same model answering directly?

Run:  python evaluate.py                 (all 12 questions, both modes)
      python evaluate.py --limit 4       (first 4 questions only, for a quick test)
Writes results/<model>_results.csv and prints a summary table you can paste into your README.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time

from config import DATA_DIR, RESULTS_DIR
from llm import LocalLLM
from react_agent import ReActAgent

DIRECT_SYSTEM = (
    "Answer the question about the delivery-robot fleet. Reply with a single number without units, "
    "or a single word. Do not explain."
)


def extract_number(text: str) -> float | None:
    match = re.search(r"-?\d[\d,]*\.?\d*", text or "")
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def is_correct(prediction: str | None, gold, kind: str) -> bool:
    if prediction is None:
        return False
    if kind == "number":
        value = extract_number(prediction)
        return value is not None and abs(value - float(gold)) <= max(1e-6, 0.01 * abs(float(gold)))
    if kind == "yesno":
        return prediction.strip().lower().startswith(str(gold).lower())
    return re.search(rf"\b{re.escape(str(gold))}\b", prediction, re.I) is not None


def run_direct(llm: LocalLLM, question: str) -> tuple[str, float]:
    start = time.perf_counter()
    answer = llm.chat([{"role": "system", "content": DIRECT_SYSTEM}, {"role": "user", "content": question}],
                      max_new_tokens=20)
    return answer, round(time.perf_counter() - start, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true", help="do not print the agent's reasoning")
    args = parser.parse_args()

    questions = json.loads((DATA_DIR / "questions.json").read_text(encoding="utf-8"))[: args.limit]
    llm = LocalLLM()
    agent = ReActAgent(llm, verbose=not args.quiet)

    rows = []
    for q in questions:
        direct, direct_s = run_direct(llm, q["question"])
        res = agent.run(q["question"])
        rows.append({
            "id": q["id"], "question": q["question"], "gold": q["answer"],
            "direct_answer": direct, "direct_ok": is_correct(direct, q["answer"], q["type"]), "direct_s": direct_s,
            "agent_answer": res.answer, "agent_ok": is_correct(res.answer, q["answer"], q["type"]),
            "agent_s": res.seconds, "tool_calls": res.tool_calls, "steps": len(res.steps),
        })
        r = rows[-1]
        print(f"[{q['id']:>2}] gold={q['answer']!s:<6} direct={direct!s:<10} {'OK ' if r['direct_ok'] else 'X  '}"
              f" agent={res.answer!s:<10} {'OK' if r['agent_ok'] else 'X'}  ({res.tool_calls} tools, {res.seconds} s)")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{llm.model_id.split('/')[-1]}_results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    acc_d = sum(r["direct_ok"] for r in rows) / n
    acc_a = sum(r["agent_ok"] for r in rows) / n
    print(f"\nModel: {llm.model_id}   questions: {n}   saved: {out}")
    print("| Mode   | Accuracy | Avg. time (s) | Avg. tool calls |")
    print("|--------|----------|---------------|-----------------|")
    print(f"| Direct | {acc_d:.0%}      | {sum(r['direct_s'] for r in rows) / n:.1f}           | 0               |")
    print(f"| Agent  | {acc_a:.0%}      | {sum(r['agent_s'] for r in rows) / n:.1f}           "
          f"| {sum(r['tool_calls'] for r in rows) / n:.1f}             |")


if __name__ == "__main__":
    main()
