"""Exercise 1.2 (part b): talk to the local model and measure its speed.

Run:  python chat_demo.py            (three fixed experiments)
      python chat_demo.py --chat     (free chat; type 'quit' to stop)
"""
import sys

from llm import LocalLLM

SYSTEM = {"role": "system", "content": "You are a concise assistant. Answer in at most three sentences."}


def show(llm: LocalLLM, messages: list[dict], **kwargs) -> str:
    reply = llm.chat(messages, **kwargs)
    print(f"\n>>> {messages[-1]['content']}")
    print(reply)
    s = llm.last_stats
    print(f"[{s['new_tokens']} new tokens in {s['seconds']} s = {s['tokens_per_second']} tokens/s]")
    return reply


def experiments(llm: LocalLLM) -> None:
    # Experiment A: a general question the model can answer from its training data.
    show(llm, [SYSTEM, {"role": "user", "content": "What is an intelligent agent, in one sentence?"}])

    # Experiment B: the same creative prompt three times with temperature 0 and three times with 1.0.
    prompt = [SYSTEM, {"role": "user", "content": "Invent a name for a delivery robot."}]
    for t in (0.0, 0.0, 0.0, 1.0, 1.0, 1.0):
        print(f"\n(temperature {t})", end="")
        show(llm, prompt, temperature=t, max_new_tokens=20)

    # Experiment C: questions the model CANNOT know (fictional fleet data) or may compute wrongly.
    for q in [
        "What is the battery capacity of robot Atlas in Wh?",
        "What is 37 * 48 + 112?",
    ]:
        show(llm, [SYSTEM, {"role": "user", "content": q}])


def chat_loop(llm: LocalLLM) -> None:
    history = [SYSTEM]
    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"quit", "exit"}:
            break
        history.append({"role": "user", "content": user})
        reply = llm.chat(history)
        history.append({"role": "assistant", "content": reply})
        print(f"Model: {reply}\n[{llm.last_stats['tokens_per_second']} tokens/s]")


if __name__ == "__main__":
    model = LocalLLM()
    chat_loop(model) if "--chat" in sys.argv else experiments(model)
