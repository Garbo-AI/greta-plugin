"""vLLM reasoning parser for Greta's round-based think markers.

An assistant round is one turn:
``<|ai|><|think|>reasoning<|text|>answer<|tool:call|>...<|tool:call|>...<|eot|>``.
The chat template opens the round in the prompt (``<|ai|><|think|>`` with
thinking, ``<|ai|><|text|>`` without), so generation starts inside a channel
and ``<|eot|>`` closes the whole round.
"""

from __future__ import annotations

from collections.abc import Sequence
import re

try:
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage
except ImportError:  # vLLM 0.30 compatibility
    from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser, ReasoningParserManager


class GretaReasoningParser(ReasoningParser):
    """Route ``<|think|>`` segments of a round to vLLM's reasoning field.

    Text and tool-call segments go to content. ``<|tool:call|>`` markers are
    kept in the content for the tool parser, and so is a ``<|text|>`` that
    follows a call, since it ends that call's last parameter value.
    """

    THINK = "<|think|>"
    EOT = "<|eot|>"
    TEXT = "<|text|>"
    AI = "<|ai|>"
    HUMAN = "<|human|>"
    CALL = "<|tool:call|>"
    RESPONSE = "<|tool:response|>"
    # Markers that end the round: <|eot|> normally, the others if a stop was missed.
    _END = (EOT, HUMAN, RESPONSE)
    _markers = re.compile(r"<\|(?:think|text|ai|eot|human|tool:call|tool:response)\|>")

    def __init__(self, tokenizer, *args, chat_template_kwargs=None, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        # vLLM passes the request's kwargs merged with --default-chat-template-kwargs.
        # The template defaults to enable_thinking=True.
        options = chat_template_kwargs or {}
        self._initial_mode = "think" if options.get("enable_thinking", True) else "text"

    def _id(self, token: str) -> int | None:
        return self.vocab.get(token)

    def adjust_request(self, request):
        request.skip_special_tokens = False
        options = getattr(request, "chat_template_kwargs", None) or {}
        if "enable_thinking" in options:
            self._initial_mode = "think" if options["enable_thinking"] else "text"
        return request

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        if prompt_token_ids:
            if prompt_token_ids[-1] == self._id(self.THINK):
                self._initial_mode = "think"
            elif prompt_token_ids[-1] == self._id(self.TEXT):
                self._initial_mode = "text"

    @property
    def reasoning_start_str(self) -> str:
        return self.THINK

    @property
    def reasoning_end_str(self) -> str:
        return self.TEXT

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # The latest channel marker decides; works on the prompt (which ends in
        # <|think|> or <|text|>) as well as on generated ids alone.
        think = self._id(self.THINK)
        ends = {self._id(t) for t in (self.TEXT, self.CALL, *self._END)}
        for token in reversed(input_ids):
            if token == think:
                return False
            if token in ends:
                return True
        return self._initial_mode == "text"

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids) -> bool:
        # Hand over to the content phase only once the answer has a non-blank token: after
        # the hand-over vLLM streams raw text, and the token after <|text|> is a lone space.
        if not self.is_reasoning_end(input_ids):
            return False
        markers = {self._id(t) for t in (self.TEXT, self.CALL, *self._END)}
        last = max((i for i, t in enumerate(input_ids) if t in markers), default=-1)
        if last < 0 or input_ids[last] != self._id(self.TEXT):
            return True
        return bool(self.model_tokenizer.decode(list(input_ids[last + 1:])).strip())

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        think, text, call, ai = (self._id(t) for t in (self.THINK, self.TEXT, self.CALL, self.AI))
        ends = {self._id(t) for t in self._END}
        mode, called = self._initial_mode, False
        result = []
        for token in input_ids:
            if token in ends:
                break
            if token == think:
                mode = "think"
            elif token == text:
                mode = "text"
                if called:
                    result.append(token)
            elif token == call:
                mode, called = "text", True
                result.append(token)
            elif token != ai and mode == "text":
                result.append(token)
        return result

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        think, text, call = (self._id(t) for t in (self.THINK, self.TEXT, self.CALL))
        ends = {self._id(t) for t in self._END}
        mode, count = self._initial_mode, 0
        for token in token_ids:
            if token in ends:
                break
            if token == think:
                mode = "think"
            elif token in (text, call):
                mode = "text"
            elif mode == "think":
                count += 1
        return count

    def _parse(self, output: str, streaming: bool = False):
        if streaming:
            partial = re.search(r"<(?:\|[^<>]*)?$", output)
            if partial:
                output = output[:partial.start()]
        mode, called = self._initial_mode, False
        reasoning, content = [], []
        offset = 0
        for match in self._markers.finditer(output):
            segment = output[offset:match.start()]
            (reasoning if mode == "think" else content).append(segment)
            offset = match.end()
            marker = match.group()
            if marker in self._END:
                break
            if marker == self.THINK:
                mode = "think"
            elif marker == self.TEXT:
                mode = "text"
                if called:
                    content.append(marker)
            elif marker == self.CALL:
                mode, called = "text", True
                content.append(marker)
        else:
            (reasoning if mode == "think" else content).append(output[offset:])
        # The tokenizer starts text after a special token with a space token.
        return "".join(reasoning).lstrip() or None, "".join(content).lstrip() or None

    def extract_reasoning(self, model_output: str, request):
        if request is not None:
            self.adjust_request(request)
        return self._parse(model_output)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        reasoning, content = self._parse(current_text, streaming=True)
        old_reasoning, old_content = self._parse(previous_text, streaming=True)
        reasoning_delta = (reasoning or "")[len(old_reasoning or ""):]
        content_delta = (content or "")[len(old_content or ""):]
        if not reasoning_delta and not content_delta:
            return None
        return DeltaMessage(reasoning=reasoning_delta or None, content=content_delta or None)


ReasoningParserManager.register_module(["greta", "greta_markers"], module=GretaReasoningParser)

__all__ = ["GretaReasoningParser"]
