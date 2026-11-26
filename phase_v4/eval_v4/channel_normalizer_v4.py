#!/usr/bin/env python3
"""
Deterministic Cross-Family Reasoning-Channel & Turn-Delimiter Normalizer (`eval_v4`).

Maps raw completions from `Inkling-2-Mini` (`inkling2`), `Gemma 4` (`gemma4`),
`Muse Glimmer` (`muse_glimmer`), and `Qwen` (`qwen`) into a uniform
`NormalizedChannelOutput(thinking_text, final_text, has_thinking, has_unclosed_thinking, has_template_leak)`
without altering mathematical answers or Lean 4 proof bodies.

Locked into `IMMUTABLE_MANIFEST_V4.json` prior to Plane-2 post-training.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

VALID_CHANNEL_FAMILIES = ("auto", "inkling2", "gemma4", "muse_glimmer", "qwen")

# Spoof / multi-turn prompt-injection markers that must NEVER appear inside extracted `final_text`
INJECTED_TURN_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|message_user|>",
    "<|message_system|>",
    "<|message_tool|>",
    "<|content_invoke_tool_json|>",
    "<|turn>user",
    "<|turn>system",
    "<|start|>user",
    "<|start|>system",
    "<|start|>tool",
    "<bos>",
    "</s>",
)


@dataclass(frozen=True)
class NormalizedChannelOutput:
    family: str
    thinking_text: str
    final_text: str
    has_thinking: bool
    has_unclosed_thinking: bool
    has_template_leak: bool


def detect_channel_family(raw_text: str) -> str:
    if any(tok in raw_text for tok in ("<|content_thinking|>", "<|content_text|>", "<|message_model|>", "<|end_message|>", "<|content_model_end_sampling|>")):
        return "inkling2"
    if any(tok in raw_text for tok in ("<|channel>", "<channel|>", "<turn|>")):
        return "gemma4"
    if any(tok in raw_text for tok in ("to=self<|message|>", "to=user<|message|>", "<|eot|>", "<|eom|>")):
        return "muse_glimmer"
    return "qwen"


def _strip_edge_tokens(s: str, tokens: tuple[str, ...]) -> str:
    out = s
    changed = True
    while changed:
        changed = False
        stripped = out.strip()
        for t in tokens:
            if stripped.startswith(t):
                out = stripped[len(t) :]
                changed = True
                break
            if stripped.endswith(t):
                out = stripped[: -len(t)]
                changed = True
                break
    return out.strip()


def normalize_reasoning_channel(
    raw_text: str,
    *,
    channel_family: str = "auto",
    expect_thinking: bool = False,
) -> NormalizedChannelOutput:
    if channel_family not in VALID_CHANNEL_FAMILIES:
        raise ValueError(f"Unsupported channel_family={channel_family!r}; expected one of {VALID_CHANNEL_FAMILIES}")

    fam = detect_channel_family(raw_text) if channel_family == "auto" else channel_family

    thinking_text = ""
    final_text = raw_text
    has_thinking = False
    has_unclosed = False

    if fam == "inkling2":
        # Format:
        #   (<|message_model|>)?(<|content_thinking|>...<|end_message|>)?
        #   (<|message_model|>)?(<|content_text|>...<|end_message|>)?(<|content_model_end_sampling|>)?
        s = raw_text
        if "<|content_thinking|>" in s:
            has_thinking = True
            after_t = s.split("<|content_thinking|>", 1)[1]
            if "<|end_message|>" not in after_t and "<|content_text|>" not in after_t:
                has_unclosed = True
                thinking_text = after_t
                final_text = ""
            elif "<|content_text|>" in after_t:
                t_part, c_part = after_t.rsplit("<|content_text|>", 1)
                thinking_text = t_part.split("<|end_message|>", 1)[0].strip()
                final_text = c_part
            else:
                t_part, rem = after_t.split("<|end_message|>", 1)
                thinking_text = t_part.strip()
                final_text = rem
        elif "<|content_text|>" in s:
            final_text = s.rsplit("<|content_text|>", 1)[1]
        else:
            final_text = s

        if not has_unclosed:
            for end_tok in ("<|content_model_end_sampling|>", "<|end_message|>"):
                if end_tok in final_text:
                    final_text = final_text.split(end_tok, 1)[0]
            final_text = _strip_edge_tokens(
                final_text,
                ("<|message_model|>", "<|content_text|>", "<|end_message|>", "<|content_model_end_sampling|>"),
            )
            if expect_thinking and not has_thinking:
                # Note: at low reasoning_effort Inkling-2-Mini may skip `<|content_thinking|>` on trivial steps;
                # only flag `has_unclosed` when a thinking block was opened and never closed, or `final_text` is empty.
                if not final_text:
                    has_unclosed = True

    elif fam == "gemma4":
        # Format:
        #   enable_thinking=True: prompt opens `<|channel>thought\n`, so completion is `THOUGHT<channel|>FINAL<turn|>`
        #                         (or `<|channel>thought\nTHOUGHT<channel|>FINAL<turn|>`). If `<channel|>` is missing,
        #                         generation truncated inside the open thought channel!
        #   enable_thinking=False: prompt already ends with `<|channel>thought\n<channel|>` on 31B/26B,
        #                          so completion is `FINAL<turn|>`.
        s = raw_text
        if "<channel|>" in s:
            t_raw, c_raw = s.rsplit("<channel|>", 1)
            if "<|channel>" in t_raw:
                t_body = t_raw.rsplit("<|channel>", 1)[1]
            else:
                t_body = t_raw
            t_body = re.sub(r"^thought\s*\n?", "", t_body, count=1)
            thinking_text = t_body.strip()
            has_thinking = bool(thinking_text) or expect_thinking
            final_text = c_raw
        elif "<|channel>" in s or expect_thinking:
            has_thinking = True
            has_unclosed = True
            thinking_text = s.split("<|channel>", 1)[1] if "<|channel>" in s else s
            final_text = ""
        else:
            final_text = s

        if not has_unclosed:
            for end_tok in ("<turn|>", "<end_of_turn>", "<eos>"):
                if end_tok in final_text:
                    final_text = final_text.split(end_tok, 1)[0]
            final_text = _strip_edge_tokens(final_text, ("<|turn>model", "<turn|>", "<eos>", "<end_of_turn>", "<pad>"))

    elif fam == "muse_glimmer":
        # Prompt ends with `<|start|>assistant`.
        # Completion with thinking: ` to=self<|message|>REASONING<|eom|><|start|>assistant to=user<|message|>FINAL<|eot|>`
        # Completion direct:        ` to=user<|message|>FINAL<|eot|>`
        s = raw_text
        m_self = re.findall(r"(?:<\|start\|>assistant\s+)?to=self<\|message\|>(.*?)(?:<\|eom\|>|<\|eot\|>)", s, flags=re.DOTALL)
        if m_self:
            has_thinking = True
            thinking_text = "\n\n".join(x.strip() for x in m_self if x.strip())

        if "to=user<|message|>" in s:
            final_text = s.rsplit("to=user<|message|>", 1)[1]
        elif "to=self<|message|>" in s:
            # Opened `to=self` block without ever reaching `to=user<|message|>`!
            has_thinking = True
            has_unclosed = True
            thinking_text = s.split("to=self<|message|>", 1)[1]
            final_text = ""
        elif "<|message|>" in s and not s.lstrip().startswith("to=self"):
            final_text = s.rsplit("<|message|>", 1)[1]
        else:
            final_text = s

        if not has_unclosed:
            for end_tok in ("<|eot|>", "<|eom|>", "<|end_of_text|>"):
                if end_tok in final_text:
                    final_text = final_text.split(end_tok, 1)[0]
            final_text = _strip_edge_tokens(final_text, ("<|eot|>", "<|eom|>", "<|end_of_text|>"))

    else:
        # Qwen / `<think>...</think>` family
        s = raw_text
        if "</think>" in s:
            t_part, c_part = s.rsplit("</think>", 1)
            thinking_text = t_part.replace("<think>", "").strip()
            has_thinking = bool(thinking_text)
            final_text = c_part.strip()
        elif "<think>" in s:
            has_thinking = True
            has_unclosed = True
            thinking_text = s.replace("<think>", "").strip()
            final_text = ""
        elif expect_thinking:
            has_unclosed = True
            final_text = ""
        else:
            final_text = s.strip()

    # Strip edge padding tokens across all families before final inspection
    if final_text and not has_unclosed:
        final_text = _strip_edge_tokens(final_text, ("<pad>", "<|pad|>", "<|finetune_right_pad_id|>"))

    has_leak = any(tok in final_text for tok in INJECTED_TURN_MARKERS)
    if fam == "qwen" and ("<think>" in final_text or "</think>" in final_text):
        has_leak = True
    if fam == "inkling2" and any(t in final_text for t in ("<|content_thinking|>", "<|message_model|>")):
        has_leak = True
    if fam == "gemma4" and ("<|channel>" in final_text or "<channel|>" in final_text):
        has_leak = True
    if fam == "muse_glimmer" and ("to=self" in final_text or "<|start|>" in final_text or "<atem:" in final_text):
        has_leak = True

    return NormalizedChannelOutput(
        family=fam,
        thinking_text=thinking_text,
        final_text=final_text,
        has_thinking=has_thinking,
        has_unclosed_thinking=has_unclosed,
        has_template_leak=has_leak,
    )
