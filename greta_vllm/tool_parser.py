"""vLLM tool parser for Greta's marker-based tool-call format.

All calls of an assistant round share one turn:
``<|ai|><|think|>...<|text|>...<|tool:call|><|tool:name|>name<|tool:params|>...<|tool:call|>...<|eot|>``.
A call runs until the next channel marker (``<|tool:call|>``, ``<|text|>``,
``<|think|>``) or the end of the round. Each parameter is delimited by
``<|tool:param|>``, ``<|tool:param:name|>`` and ``<|tool:param:value|>``.
Values emitted by the template are either raw strings or JSON values, so
parsing tries JSON first and falls back to a string. The template writes no
call ids; a legacy textual ``<|tool:id|>`` is still accepted.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser, ToolParserManager
from vllm.entrypoints.chat_utils import make_tool_call_id


class GretaToolParser(ToolParser):
    """Parse Greta marker calls for OpenAI-compatible vLLM responses."""

    supports_required_and_named = False
    CALL = "<|tool:call|>"
    NAME = "<|tool:name|>"
    ID = "<|tool:id|>"
    PARAMS = "<|tool:params|>"
    PARAM = "<|tool:param|>"
    PARAM_NAME = "<|tool:param:name|>"
    PARAM_VALUE = "<|tool:param:value|>"
    EOT = "<|eot|>"
    # Channel markers of a round. Everything after an end marker is ignored.
    _channels = re.compile(r"<\|(?:think|text|ai|eot|human|tool:call|tool:response)\|>")
    _END = ("<|eot|>", "<|human|>", "<|tool:response|>")

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self._streamed_calls: list[dict[str, Any]] = []
        self._streamed_content = ""
        self._streamed_reasoning = ""

    def adjust_request(self, request: ChatCompletionRequest | ResponsesRequest):
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Greta's markers are special tokens. Dropping them before this
            # parser runs makes the output impossible to disambiguate.
            request.skip_special_tokens = False
        return request

    @staticmethod
    def _value(raw: str) -> Any:
        raw = raw.strip()
        if not raw:
            return ""
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return raw

    def _parse_call(self, block: str) -> dict[str, Any] | None:
        name_start = block.find(self.NAME)
        if name_start < 0:
            return None
        name_start += len(self.NAME)
        id_pos = block.find(self.ID, name_start)
        params_pos = block.find(self.PARAMS, name_start)
        name_end = min([p for p in (id_pos, params_pos) if p >= 0], default=len(block))
        name = block[name_start:name_end].strip()
        if not name:
            return None
        call_id = ""
        if id_pos >= 0 and (params_pos < 0 or id_pos < params_pos):
            call_id = block[id_pos + len(self.ID) : params_pos if params_pos >= 0 else len(block)].strip()
        params = block[params_pos + len(self.PARAMS) :] if params_pos >= 0 else ""
        arguments: dict[str, Any] = {}
        for raw_param in params.split(self.PARAM):
            if self.PARAM_NAME not in raw_param or self.PARAM_VALUE not in raw_param:
                continue
            nstart = raw_param.index(self.PARAM_NAME) + len(self.PARAM_NAME)
            vpos = raw_param.index(self.PARAM_VALUE, nstart)
            pname = raw_param[nstart:vpos].strip()
            if pname:
                arguments[pname] = self._value(raw_param[vpos + len(self.PARAM_VALUE) :])
        return {
            "name": name,
            "id": call_id,
            "arguments": arguments,
            # A name is final once a later marker follows it.
            "name_done": name_end < len(block),
        }

    def _parse(self, output: str, ended: bool = False) -> tuple[str, list[dict[str, Any]], str]:
        """Split a round into (content, calls, reasoning).

        The input is either the reasoning parser's content (text and call
        segments) or raw output starting in the text channel. ``ended`` marks
        the last call closed even without a following marker (stream finished).
        Each call carries ``closed``: its arguments can no longer change.
        """
        content: list[str] = []
        reasoning: list[str] = []
        calls: list[dict[str, Any]] = []
        mode, offset = "text", 0
        markers = list(self._channels.finditer(output))
        for i, match in enumerate(markers + [None]):
            segment = output[offset : match.start() if match else len(output)]
            if mode == "text":
                content.append(segment)
            elif mode == "think":
                reasoning.append(segment)
            elif mode == "call":
                call = self._parse_call(segment)
                if call is not None:
                    call["closed"] = match is not None or ended
                    call["name_done"] = call["name_done"] or call["closed"]
                    calls.append(call)
            if match is None:
                break
            marker = match.group()
            offset = match.end()
            if marker in self._END:
                break
            if marker == self.CALL:
                mode = "call"
            elif marker == "<|think|>":
                mode = "think"
            elif marker == "<|text|>":
                mode = "text"
            # <|ai|> keeps the current channel.
        return "".join(content).strip(), calls, "".join(reasoning)

    @staticmethod
    def _tool_call(call: dict[str, Any], call_id: str | None = None) -> ToolCall:
        return ToolCall(
            id=call_id or call.get("id") or make_tool_call_id(),
            type="function",
            function=FunctionCall(
                name=call["name"],
                arguments=json.dumps(call.get("arguments", {}), ensure_ascii=False),
            ),
        )

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest):
        content, calls, _ = self._parse(model_output, ended=True)
        return ExtractedToolCallInformation(
            tools_called=bool(calls),
            tool_calls=[self._tool_call(c) for c in calls],
            content=content or None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # The round's <|eot|> is the EOS token and usually not in the text.
        end_ids = {self.vocab.get(t) for t in self._END}
        ended = any(t in end_ids for t in delta_token_ids)
        content, calls, reasoning = self._parse(current_text, ended=ended)
        deltas: list[DeltaToolCall] = []
        for i, call in enumerate(calls):
            if i == len(self._streamed_calls):
                if not call["name_done"]:
                    break
                call_id = call.get("id") or make_tool_call_id()
                self._streamed_calls.append({"id": call_id, "name": call["name"], "args": ""})
                self.prev_tool_call_arr.append({"name": call["name"], "arguments": {}})
                self.streamed_args_for_tool.append("")
                self.current_tool_id = i
                deltas.append(DeltaToolCall(
                    id=call_id, type="function", index=i,
                    function=DeltaFunctionCall(name=call["name"]).model_dump(exclude_none=True),
                ))
            self.prev_tool_call_arr[i]["arguments"] = call["arguments"]
            # Raw string values may still grow, so a call's arguments are sent
            # once it is closed by the next marker or the end of the round.
            if call["closed"] and not self._streamed_calls[i]["args"]:
                encoded = json.dumps(call["arguments"], ensure_ascii=False)
                self._streamed_calls[i]["args"] = encoded
                self.streamed_args_for_tool[i] = encoded
                deltas.append(DeltaToolCall(
                    index=i,
                    function=DeltaFunctionCall(arguments=encoded).model_dump(exclude_none=True),
                ))
        new_content = content[len(self._streamed_content) :] if content.startswith(self._streamed_content) else content
        self._streamed_content = content
        new_reasoning = reasoning[len(self._streamed_reasoning) :]
        self._streamed_reasoning = reasoning
        message = DeltaMessage(content=new_content or None, reasoning=new_reasoning or None, tool_calls=deltas)
        return message if message.content or message.reasoning or message.tool_calls else None


ToolParserManager.register_module(["greta", "greta_markers"], module=GretaToolParser)

__all__ = ["GretaToolParser"]
