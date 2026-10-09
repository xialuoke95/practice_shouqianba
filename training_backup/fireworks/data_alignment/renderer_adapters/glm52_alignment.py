"""GLM 5.2 renderer for preserve-history, tools-aware single-datum SFT."""

from __future__ import annotations

import json

from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import Message, ToolSpec
from tinker_cookbook.tokenizer_utils import Tokenizer
from training.renderer.glm5 import GLMMoeDsaRenderer


_TOOLS_INTRO = """
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, output the function name and arguments within the following XML format:
<tool_call>{{function-name}}<arg_key>{{arg-key-1}}</arg_key><arg_value>{{arg-value-1}}</arg_value><arg_key>{{arg-key-2}}</arg_key><arg_value>{{arg-value-2}}</arg_value>...</tool_call>"""


class GLM52PreserveHistorySingleDatumRenderer(GLMMoeDsaRenderer):
    """Match GLM-5.2 with ``clear_thinking=False`` and render tool declarations."""

    @property
    def has_extension_property(self) -> bool:
        return True

    def _assistant_think_block(
        self,
        *,
        before_last_user: bool,
        reasoning: str,
    ) -> str:
        del before_last_user
        if reasoning:
            return f"<think>{reasoning}</think>"
        return "<think></think>"

    def create_conversation_prefix_with_tools(
        self,
        tools: list[ToolSpec],
        system_prompt: str = "",
    ) -> list[Message]:
        messages: list[Message] = []
        if tools:
            encoded_tools = []
            for tool in tools:
                public_tool = {
                    key: value
                    for key, value in tool.items()
                    if key not in {"defer_loading", "strict"}
                }
                encoded_tools.append(json.dumps(public_tool, ensure_ascii=False))
            messages.append(
                Message(
                    role="system",
                    content=_TOOLS_INTRO.format(tools="\n".join(encoded_tools)),
                )
            )
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        return messages


def _factory(
    tokenizer: Tokenizer,
    image_processor=None,
) -> GLM52PreserveHistorySingleDatumRenderer:
    del image_processor
    return GLM52PreserveHistorySingleDatumRenderer(tokenizer)


register_renderer("glm52_preserve_history_single_datum", _factory)
