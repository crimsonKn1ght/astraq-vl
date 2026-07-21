"""Decode-correctness tests for every model family's output extraction.

These pin the reviewer's requirement that only newly generated tokens are decoded
and that decoded responses contain no system message, user prompt, or image
tokens. They run on crafted id sequences / strings with no torch, no GPU, and no
model download, including a batched + left-padded case.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decode_utils import (  # noqa: E402
    response_leak_flags,
    split_assistant_response,
    strip_prompt_if_echoed,
    trim_completion_ids,
)


# --- AstraQ-VL marker contract ------------------------------------------------

def test_marker_split_full_sequence_with_prompt_echo():
    """Full ChatML decode: keep only the assistant turn, drop scaffolding."""
    decoded = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<image>\nWhat is shown?<|im_end|>\n"
        "<|im_start|>assistant\nA spiral galaxy.<|im_end|>"
    )
    assert split_assistant_response(decoded) == "A spiral galaxy."


def test_marker_split_completion_only():
    """Completion-only text (no markers) is returned unchanged apart from strip."""
    assert split_assistant_response("  A spiral galaxy.  ") == "A spiral galaxy."


def test_marker_split_does_not_truncate_leading_tokens():
    """The assistant answer must survive intact, not lose its first word."""
    decoded = "<|im_start|>assistant\nNGC 1300 is a barred spiral.<|im_end|>"
    assert split_assistant_response(decoded) == "NGC 1300 is a barred spiral."


# --- Qwen echoed-prompt contract ---------------------------------------------

def test_trim_completion_single():
    input_ids = [10, 11, 12, 13]
    output_ids = [10, 11, 12, 13, 40, 41]
    assert list(trim_completion_ids(input_ids, output_ids)) == [40, 41]


def test_trim_completion_batched_left_padded():
    """Left-padded batch: every row shares the padded length, so slicing at that
    length yields each row's own completion (no cross-row leakage)."""
    pad = 0
    # two prompts of true length 2 and 4, left-padded to width 4
    input_ids = [
        [pad, pad, 12, 13],
        [20, 21, 22, 23],
    ]
    generated = [
        [pad, pad, 12, 13, 90, 91],   # completion 90, 91
        [20, 21, 22, 23, 80],         # completion 80
    ]
    trimmed = [list(trim_completion_ids(i, o)) for i, o in zip(input_ids, generated)]
    assert trimmed == [[90, 91], [80]]


# --- AstroLLaVA conditional contract -----------------------------------------

def test_strip_prompt_when_echoed():
    input_ids = [5, 6, 7]
    output_ids = [5, 6, 7, 100, 101]
    assert list(strip_prompt_if_echoed(input_ids, output_ids)) == [100, 101]


def test_no_strip_when_completion_only():
    """The haotian-liu LLaVA stack returns completion-only ids; stripping the
    prompt length would delete the answer's leading tokens. Guard against it."""
    input_ids = [5, 6, 7, 8, 9]      # long prompt
    output_ids = [100, 101]           # short completion, does NOT begin with prompt
    assert list(strip_prompt_if_echoed(input_ids, output_ids)) == [100, 101]


def test_no_strip_when_output_shorter_than_prompt():
    input_ids = [5, 6, 7, 8, 9]
    output_ids = [100]
    assert list(strip_prompt_if_echoed(input_ids, output_ids)) == [100]


# --- Response leakage check ---------------------------------------------------

def test_clean_response_has_no_flags():
    assert response_leak_flags("A spiral galaxy.", prompt="What is shown?") == []


def test_flags_surviving_scaffolding_and_image_token():
    flags = response_leak_flags("<image>\nA galaxy<|im_end|>", prompt="What is shown?")
    assert "marker:<image>" in flags
    assert "marker:<|im_end|>" in flags


def test_flags_prompt_echo():
    prompt = "Describe the astronomical object in this image in detail."
    response = "Describe the astronomical object in this image in detail. It is a nebula."
    assert "prompt_echo" in response_leak_flags(response, prompt=prompt)


def test_short_prompt_not_treated_as_echo():
    # short prompts can incidentally overlap answer wording; do not flag
    assert response_leak_flags("yes it is", prompt="Is it?") == []


if __name__ == "__main__":
    import traceback

    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(funcs) - failures}/{len(funcs)} passed")
    sys.exit(1 if failures else 0)
