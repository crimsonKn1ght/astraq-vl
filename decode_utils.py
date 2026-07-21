"""Shared, dependency-free helpers for turning raw generation output into a
clean model response, plus a leakage check.

Three model families produce completions under two different contracts:

* **Marker contract** (AstraQ-VL): ``model.generate`` decodes from
  ``inputs_embeds`` and returns completion-only ids, but ``inference.py`` decodes
  the full sequence *with* special tokens and slices out the assistant turn by
  string markers. Robust whether or not the prompt is echoed.
* **Echoed-prompt contract** (Qwen2.5-VL): ``generate`` echoes the full (padded)
  prompt ahead of the completion, so the completion is everything past the prompt
  length. Correct for left-padded batches because every row shares the padded
  length.
* **Conditional contract** (AstroLLaVA / haotian-liu LLaVA): ``generate`` decodes
  from ``inputs_embeds`` and normally returns completion-only ids, but some
  legacy paths echo the prompt. Strip the prompt only when the output actually
  begins with it (this is the fix from commit b773be2).

Keeping the slice/split logic here — free of torch and of any model object — lets
it be unit-tested on crafted inputs, including batched/padded cases, with no GPU
or weight download.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

ASSISTANT_MARKER = "<|im_start|>assistant\n"
END_MARKER = "<|im_end|>"

# Tokens that must never survive into a scored response: chat scaffolding and the
# image placeholder (in any of the vendor spellings we feed a model).
LEAK_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<image>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "[INST]",
    "[/INST]",
)


def split_assistant_response(
    decoded_text: str,
    assistant_marker: str = ASSISTANT_MARKER,
    end_marker: str = END_MARKER,
) -> str:
    """Marker contract: pull the assistant turn out of a fully-decoded sequence.

    Behaviour matches ``inference.py``: if the assistant marker is present keep
    everything after the last one; if the end marker is present keep everything
    before the first one; then strip. Robust to completion-only output (neither
    marker present -> the text is returned unchanged apart from stripping).
    """
    text = decoded_text
    if assistant_marker in text:
        text = text.split(assistant_marker)[-1]
    if end_marker in text:
        text = text.split(end_marker)[0]
    return text.strip()


def trim_completion_ids(input_ids_row: Sequence[int], output_ids_row: Sequence[int]):
    """Echoed-prompt contract: completion is everything past the prompt length.

    Works on python lists or torch tensors (only ``len`` and slicing are used).
    For left-padded batches every ``input_ids_row`` shares the padded length, so
    slicing at that length correctly drops the (padded) prompt for every row.
    """
    return output_ids_row[len(input_ids_row):]


def _starts_with(output_row: Sequence[int], input_row: Sequence[int]) -> bool:
    n = len(input_row)
    if len(output_row) < n:
        return False
    return all(int(a) == int(b) for a, b in zip(output_row[:n], input_row))


def strip_prompt_if_echoed(input_row: Sequence[int], output_row: Sequence[int]):
    """Conditional contract: strip the prompt only if the output begins with it.

    Returns the same sequence type it is given (list in, list out; tensor slice in,
    tensor slice out). Guards against deleting the answer's leading tokens when the
    model returns completion-only ids.
    """
    if _starts_with(output_row, input_row):
        return output_row[len(input_row):]
    return output_row


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("<image>", " ")).strip().lower()


def response_leak_flags(response: str, prompt: Optional[str] = None) -> List[str]:
    """Return reasons a decoded response is *not* clean; empty list means clean.

    Flags (a) any surviving chat/image scaffolding token, and (b) a verbatim echo
    of the prompt (only when the normalized prompt is long enough that an echo is
    not plausibly incidental word overlap).
    """
    flags: List[str] = []
    text = response or ""
    for marker in LEAK_MARKERS:
        if marker in text:
            flags.append(f"marker:{marker}")
    if prompt:
        prompt_n = _normalize(prompt)
        if len(prompt_n) >= 20 and prompt_n in _normalize(text):
            flags.append("prompt_echo")
    return flags
