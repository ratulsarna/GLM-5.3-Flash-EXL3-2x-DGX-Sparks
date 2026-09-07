#!/usr/bin/env python3
"""CPU-only GLM template contracts; no tokenizer, processor or model imports."""

import json
import unittest
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment


def _tojson(value, ensure_ascii=False, indent=None, **_kwargs):
    """vLLM's renderer supplies a tojson accepting ensure_ascii; bare Jinja2
    does not. Register the same shape so the template renders standalone."""
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)


def _environment() -> ImmutableSandboxedEnvironment:
    env = ImmutableSandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = _tojson
    return env


TEMPLATE = Path(__file__).parents[1] / "files" / "chat_template.jinja"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    }
]

CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "Hi. How can I help?"},
    {"role": "user", "content": "and 3+3?"},
]


def render_generation_prompt(**kwargs: object) -> str:
    template = _environment().from_string(TEMPLATE.read_text())
    return template.render(
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        add_generation_prompt=True,
        **kwargs,
    )


def render_conversation(**kwargs: object) -> str:
    template = _environment().from_string(TEMPLATE.read_text())
    return template.render(add_generation_prompt=True, **kwargs)


class ChatTemplateTests(unittest.TestCase):
    def test_thinking_defaults_on(self) -> None:
        rendered = render_generation_prompt()
        self.assertIn("<|system|>Reasoning Effort: Max", rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think>"), rendered)

    def test_thinking_can_be_disabled(self) -> None:
        rendered = render_generation_prompt(enable_thinking=False)
        # The head line stays: it is what keeps the cached prefix stable across
        # a thinking toggle. Thinking is disabled by the closed block instead.
        self.assertIn("<|system|>Reasoning Effort: Max", rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think></think>"), rendered)

    def test_thinking_alias_matches_parser_behavior(self) -> None:
        rendered = render_generation_prompt(thinking=False)
        self.assertIn("<|system|>Reasoning Effort: Max", rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think></think>"), rendered)

    def test_explicit_thinking_preserves_reasoning_effort(self) -> None:
        rendered = render_generation_prompt(
            enable_thinking=True,
            reasoning_effort="low",
        )
        self.assertIn("<|system|>Reasoning Effort: Low", rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think>"), rendered)


class PrefixStabilityTests(unittest.TestCase):
    """Toggling thinking must not change the prompt before the final token.

    vLLM chains prefix-cache block hashes forward from token 0, so any
    divergence near the head invalidates the whole prompt. The off-shape must
    therefore be a strict extension of the on-shape.
    """

    def _assert_strict_extension(self, tools) -> None:
        on = render_conversation(
            messages=CONVERSATION, tools=tools, enable_thinking=True
        )
        off = render_conversation(
            messages=CONVERSATION, tools=tools, enable_thinking=False
        )
        self.assertTrue(
            off.startswith(on),
            "thinking-off prompt must extend thinking-on prompt, but they "
            f"diverge at char {len(_common_prefix(on, off))} of {len(on)}",
        )
        self.assertEqual(off[len(on) :], "</think>")

    def test_toggle_is_prefix_stable_without_tools(self) -> None:
        self._assert_strict_extension(None)

    def test_toggle_is_prefix_stable_with_tools(self) -> None:
        # Agent traffic always carries tools; the tools block renders after the
        # reasoning-effort line, so this is the case that actually regressed.
        self._assert_strict_extension(TOOLS)

    def test_effort_levels_share_the_prompt_up_to_the_effort_word(self) -> None:
        low = render_conversation(
            messages=CONVERSATION, tools=TOOLS, enable_thinking=True,
            reasoning_effort="low",
        )
        high = render_conversation(
            messages=CONVERSATION, tools=TOOLS, enable_thinking=True,
            reasoning_effort="high",
        )
        self.assertEqual(len(_common_prefix(low, high)), len("[gMASK]<sop><|system|>Reasoning Effort: "))


def call(identifier, name="lookup", arguments=None):
    return {"id": identifier, "type": "function", "function": {
        "name": name, "arguments": {"q": identifier} if arguments is None else arguments,
    }}


def assistant(calls=(), content="", **kwargs):
    return {"role": "assistant", "content": content, "tool_calls": list(calls), **kwargs}


def result(identifier, content):
    return {"role": "tool", "tool_call_id": identifier, "content": content}


class ToolHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = _environment().from_string(TEMPLATE.read_text())

    def render(self, messages, **kwargs):
        return self.template.render(messages=messages, tools=TOOLS,
                                    add_generation_prompt=True, **kwargs)

    def assert_responses(self, messages, expected):
        import re
        rendered = self.render(messages)
        self.assertEqual(re.findall(r"<tool_response>(.*?)</tool_response>",
                                    rendered, flags=re.S), expected)
        self.assertEqual(rendered.count("<|observation|>"), 1)

    def test_null_and_empty_tool_only_turns_are_identical(self):
        empty = self.render([assistant([call("a")]), result("a", "ONE")])
        for content in (None, "", []):
            with self.subTest(content=content):
                rendered = self.render([assistant([call("a")], content), result("a", "ONE")])
                self.assertEqual(rendered, empty)
                self.assertRegex(rendered, r"<\|assistant\|><think></think>\s*<tool_call>lookup")

    def test_valid_results_follow_call_order_in_all_supported_shapes(self):
        for order in (("a", "b"), ("b", "a")):
            for shape in ("messages", "outputs", "mixed"):
                with self.subTest(order=order, shape=shape):
                    outputs = {"a": "ONE", "b": "TWO"}
                    if shape == "messages":
                        block = [result(i, outputs[i]) for i in order]
                    elif shape == "outputs":
                        block = [{"role": "tool", "content": [
                            {"tool_call_id": i, "output": outputs[i]} for i in order]}]
                    else:
                        block = [result(order[0], outputs[order[0]]),
                                 {"role": "tool", "content": [
                                     {"id": order[1], "output": outputs[order[1]]}]}]
                    self.assert_responses([assistant([call("a"), call("b")])] + block,
                                          ["ONE", "TWO"])

    def test_invalid_blocks_preserve_every_result_in_input_order(self):
        for ids in (("b", "b"), ("unknown", "a"), (None, "a"), ("", "a")):
            for nested in (False, True):
                with self.subTest(ids=ids, nested=nested):
                    if nested:
                        block = [{"role": "tool", "content": [
                            {"tool_call_id": ids[0], "output": "FIRST"},
                            {"tool_call_id": ids[1], "output": "SECOND"}]}]
                    else:
                        block = [result(ids[0], "FIRST"), result(ids[1], "SECOND")]
                    self.assert_responses([assistant([call("a"), call("b")])] + block,
                                          ["FIRST", "SECOND"])

    def test_invalid_call_ids_and_orphan_results_preserve_order(self):
        for calls in ([call("a"), call("a")], [call(None), call("a")], []):
            with self.subTest(calls=calls):
                self.assert_responses([assistant(calls), result("b", "FIRST"),
                                       result("a", "SECOND")], ["FIRST", "SECOND"])
        # Both result IDs are valid members: duplicate call validation itself fails.
        self.assert_responses([assistant([call("a"), call("b"), call("a")]),
                               result("b", "FIRST"), result("a", "SECOND")],
                              ["FIRST", "SECOND"])
        self.assert_responses([assistant([call("a"), call("b"), call(None)]),
                               result("b", "FIRST"), result("a", "SECOND")],
                              ["FIRST", "SECOND"])
        self.assert_responses([result("b", "FIRST"), result("a", "SECOND")],
                              ["FIRST", "SECOND"])

    def test_partial_valid_results_are_not_duplicated_or_invented(self):
        self.assert_responses([assistant([call("a"), call("b")]), result("b", "TWO")],
                              ["TWO"])

    def test_arguments_keep_strings_and_json_values(self):
        rendered = self.render([assistant([call("a", arguments={
            "text": "café", "number": 2, "flag": True, "nil": None,
            "list": [1, "x"], "object": {"nested": 3},
        })])])
        for key, value in {"text": "café", "number": "2", "flag": "true", "nil": "null",
                           "list": '[1, "x"]', "object": '{"nested": 3}'}.items():
            self.assertIn(f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>", rendered)

    def test_numeric_tool_name_is_coerced(self):
        self.assertIn("<tool_call>42<arg_key>", self.render([assistant([call("a", 42)])]))

    def test_growing_tool_image_history_through_sixteen_turns(self):
        # Placeholder contract only: no image decoding, resize, tokenization or inference.
        messages = []
        for index in range(1, 17):
            identifier = str(index)
            messages.extend([
                {"role": "user", "content": f"inspect {index}"},
                assistant([call(identifier)], None, reasoning_content=f"REASON_{index};"),
                result(identifier, [{"type": "text", "text": f"SCREEN_{index};"},
                                    {"type": "image_url", "image_url": {
                                        "url": "synthetic-2560x1440-not-fetched"}}]),
            ])
            rendered = self.render(messages)
            with self.subTest(turns=index):
                self.assertEqual(rendered.count("<|image|>"), index)
                self.assertEqual(rendered.count("<|begin_of_image|>"), index)
                self.assertEqual(rendered.count("<|end_of_image|>"), index)
                self.assertEqual(rendered.count("<|observation|>"), index)
                self.assertEqual(rendered.count("<tool_response>"), index)
                for prior in range(1, index + 1):
                    self.assertEqual(rendered.count(f"SCREEN_{prior};"), 1)
                    self.assertEqual(rendered.count(f"REASON_{prior};"), 1)

    def test_supported_media_and_deferred_tool_references(self):
        media = ["image", "image_url", "video", "video_url", "audio", "audio_url", "input_audio"]
        rendered = self.render([{"role": "user", "content": [{"type": t} for t in media]}])
        self.assertEqual(rendered.count("<|image|>"), 2)
        self.assertEqual(rendered.count("<|video|>"), 2)
        self.assertEqual(rendered.count("<|begin_of_audio|>"), 3)
        rendered = self.render([assistant([call("a")]), result("a", [
            {"type": "tool_reference", "name": "multiply"}])])
        self.assertIn('<tool_response><tools>\n{"name": "multiply"', rendered)


class ReasoningHistoryTests(unittest.TestCase):
    def test_effort_contract_including_off_and_unsupported_values(self):
        for effort, label in [("low", "Low"), ("high", "High"), ("max", "Max"),
                              ("medium", "Max"), ("off", "Max"), (None, "Max")]:
            for enabled in (True, False):
                with self.subTest(effort=effort, enabled=enabled):
                    rendered = render_generation_prompt(reasoning_effort=effort,
                                                        enable_thinking=enabled)
                    self.assertIn(f"Reasoning Effort: {label}", rendered)
                    suffix = "<think>" if enabled else "<think></think>"
                    self.assertTrue(rendered.endswith("<|assistant|>" + suffix))

    def test_aliases_keep_existing_or_semantics(self):
        for thinking in (True, False):
            for enable_thinking in (True, False):
                rendered = render_generation_prompt(thinking=thinking, enable_thinking=enable_thinking)
                suffix = "<think>" if thinking or enable_thinking else "<think></think>"
                self.assertTrue(rendered.endswith("<|assistant|>" + suffix))

    def test_retention_and_clear_thinking_last_user_boundary(self):
        for embedded in (False, True):
            old = (assistant(content="<think>OLD_REASON</think>OLD_ANSWER") if embedded else
                   assistant(content="OLD_ANSWER", reasoning_content="OLD_REASON"))
            messages = [{"role": "user", "content": "first"}, old,
                        {"role": "user", "content": "last"},
                        assistant(content="NEW_ANSWER", reasoning_content="NEW_REASON")]
            for clear in (None, False, True):
                for enabled in (True, False):
                    with self.subTest(embedded=embedded, clear=clear, enabled=enabled):
                        kwargs = {} if clear is None else {"clear_thinking": clear}
                        rendered = render_conversation(messages=messages, tools=None,
                                                       enable_thinking=enabled, **kwargs)
                        self.assertEqual("OLD_REASON" in rendered, clear is not True)
                        self.assertIn("NEW_REASON", rendered)
                        self.assertIn("OLD_ANSWER", rendered)
                        self.assertIn("NEW_ANSWER", rendered)
            # Pi serializes images from a tool as an additional user message.
            messages.append({"role": "user", "content": [{"type": "image_url"}]})
            rendered = render_conversation(messages=messages, tools=None, clear_thinking=True)
            self.assertNotIn("OLD_REASON", rendered)
            self.assertNotIn("NEW_REASON", rendered)
            self.assertEqual(rendered.count("<|image|>"), 1)


def _common_prefix(a: str, b: str) -> str:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return a[:index]


if __name__ == "__main__":
    unittest.main()
