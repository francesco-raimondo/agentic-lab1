"""A small wrapper around a Hugging Face chat model that runs on your own computer.

Exercise 1.2: complete LocalLLM.chat().
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEVICE, DTYPE, MAX_NEW_TOKENS, MODEL_ID


def pick_device() -> tuple[str, torch.dtype]:
    """Choose the best available device and a dtype that works well on it."""
    if DEVICE:
        return DEVICE, {"cuda": torch.bfloat16, "mps": torch.float16}.get(DEVICE, torch.float32)
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def cut_at_stop(text: str, stop: list[str] | None) -> str:
    """Return text up to (not including) the first stop string it contains."""
    if not stop:
        return text
    positions = [text.find(s) for s in stop if s in text]
    return text[: min(positions)] if positions else text


class LocalLLM:
    """Loads a causal language model once and answers chat messages with it."""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        auto_device, auto_dtype = pick_device()
        self.device = device or auto_device
        dtype = getattr(torch, DTYPE) if DTYPE else auto_dtype
        self.model_id = model_id
        print(f"Loading {model_id} on {self.device} ({dtype}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device)
        self.model.eval()
        self.last_stats: dict = {}

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = MAX_NEW_TOKENS,
        stop: list[str] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate the assistant's reply to a list of chat messages.

        messages:   [{"role": "system" | "user" | "assistant", "content": "..."}, ...]
        stop:       strings that end the reply early (they are not included in the result)
        temperature: 0.0 = always pick the most likely token (deterministic);
                     higher values = more random replies.
        Also stores timing information in self.last_stats.
        """
        # TODO 1.2: turn the messages into model inputs with the chat template, generate, and decode ONLY the new tokens.
        # HINT: self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        # HINT: self.model.generate(**inputs, max_new_tokens=..., do_sample=..., stop_strings=stop, tokenizer=self.tokenizer)
        # HINT: the output contains the prompt tokens followed by the new tokens.
        raise NotImplementedError("TODO 1.2 (see the comment above)")

        self.last_stats = {
            "prompt_tokens": int(prompt_len),
            "new_tokens": int(len(new_tokens)),
            "seconds": round(elapsed, 2),
            "tokens_per_second": round(len(new_tokens) / elapsed, 1) if elapsed > 0 else 0.0,
        }
        return cut_at_stop(text, stop).strip()
