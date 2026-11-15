"""Claude models through the official Anthropic SDK.

Needs the extra: pip install 'nanoagent[anthropic]'
"""

from ..model import Call, Model, ModelError, Reply

# Models that accept server-side refusal fallbacks: when one declines a
# request, the API retries it on another model within the same call.
FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

STOPS = {"tool_use": "tool", "max_tokens": "length", "model_context_window_exceeded": "length", "refusal": "refusal"}


class Anthropic(Model):
    """A Claude model.

    model           any Claude model id
    client          an `anthropic.Anthropic` client; made from the environment if omitted
    max_tokens      the output limit for one reply
    effort          "low" to "max", or None for the model's default
    fallbacks       "default" lets the API route a declined request to another
                    model where the model supports it; None turns it off
    """

    key = "anthropic"
    images = True

    def __init__(self, model="claude-opus-5-5", *, client=None, max_tokens=64_000, effort=None,
                 fallbacks="default", context_window=1_000_000):
        if client is None:
            try:
                import anthropic
            except ImportError:
                raise ModelError("Claude models need the Anthropic SDK: pip install 'nanoagent[anthropic]'") from None
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.name = f"anthropic:{model}"
        self.max_tokens = max_tokens
        self.effort = effort
        self.fallbacks = fallbacks if model in FALLBACK_MODELS else None
        self.context_window = context_window

    def reply(self, system, tools, messages, on_text=None):
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "tools": [self._tool(each) for each in tools],
            "messages": self._messages(messages),
            "cache_control": {"type": "ephemeral"},  # reuse the unchanged start of the conversation
        }
        if self.effort:
            request["output_config"] = {"effort": self.effort}
        if self.fallbacks:
            stream = lambda: self.client.beta.messages.stream(  # noqa: E731
                **request, betas=[FALLBACK_BETA], fallbacks=self.fallbacks
            )
        else:
            stream = lambda: self.client.messages.stream(**request)  # noqa: E731

        for attempt in (1, 2):
            try:
                with stream() as events:
                    for text in events.text_stream:
                        if on_text:
                            on_text(text)
                    response = events.get_final_message()
                break
            except ValueError:  # a streamed tool input was not parseable JSON
                if attempt == 2:
                    raise

        usage = response.usage
        return Reply(
            text="".join(block.text for block in response.content if block.type == "text"),
            calls=[Call(b.id, b.name, b.input) for b in response.content if b.type == "tool_use"],
            stop=STOPS.get(response.stop_reason, "end"),
            context=usage.input_tokens
            + (usage.cache_read_input_tokens or 0)
            + (usage.cache_creation_input_tokens or 0)
            + usage.output_tokens,
            # The exact blocks, reasoning included, to send back unchanged.
            raw=[block.to_dict(mode="json") for block in response.content],
        )

    @staticmethod
    def _tool(tool):
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.schema,
            # Stream inputs as they are written instead of buffering them
            # server-side. The agent validates them before running anything.
            "eager_input_streaming": True,
        }

    def _messages(self, messages):
        wire = []
        for turn in messages:
            if turn["role"] == "user":
                wire.append({"role": "user", "content": turn["content"]})
            elif turn["role"] == "assistant":
                if turn.get("by") == self.key and turn.get("raw") is not None:
                    content = turn["raw"]
                else:  # written by another model: rebuild it from the neutral form
                    content = [{"type": "text", "text": turn["content"]}] if turn["content"] else []
                    content += [
                        {"type": "tool_use", "id": c["id"], "name": c["name"],
                         "input": c["arguments"] if isinstance(c["arguments"], dict) else {}}
                        for c in turn.get("calls", [])
                    ]
                if content:
                    wire.append({"role": "assistant", "content": content})
            elif turn["role"] == "tool":
                wire.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": r["id"], "content": _content(r["content"]), "is_error": r["error"]}
                        for r in turn["results"]
                    ],
                })
        return wire


def _content(content):
    if isinstance(content, str):
        return content
    return [
        {"type": "text", "text": part} if isinstance(part, str)
        else {"type": "image", "source": {"type": "base64", "media_type": part["media_type"], "data": part["data"]}}
        for part in content
    ]
