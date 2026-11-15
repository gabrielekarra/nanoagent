"""The agent: a model, some tools, and a loop.

This file is the whole engine. `tutorial/` builds the same thing one idea at
a time, if you want to see why each part is here.
"""

import json
import os
from pathlib import Path
from typing import Annotated

from .model import estimate, resolve
from .tools import tool

COMPACT_PROMPT = (
    "Pause the task. Without calling any tool, write a handoff note for an agent "
    "that will continue this work with no access to the conversation above: the "
    "user's goal, what is done, what is left, and every file path, command and "
    "decision it needs."
)


class Agent:
    """A model that works on a request with tools until it is done.

        agent = Agent("You are a travel assistant.", tools=[get_weather], model="ollama:qwen3:4b")
        answer = agent.run("Do I need an umbrella in Rome?")

    `run` can be called again: the agent keeps the conversation in
    `agent.messages`.

    system       instructions for the model, fixed for the agent's lifetime
    tools        functions decorated with `@tool`
    model        "provider:model" (e.g. "ollama:qwen3:4b", "anthropic:claude-opus-5-5")
                 or a `Model`; defaults to the NANOAGENT_MODEL environment variable
    approve      `approve(name, arguments) -> bool`, asked before each tool
                 declared with `@tool(approve=True)`; without it they just run
    on_text      called with each piece of text as the model writes it
    on_tool      `on_tool(name, arguments)`, called when the model uses a tool
    memory       path of a notes file: loaded into the instructions, and the
                 model gets a `remember` tool that appends to it
    subagents    give the model a `task` tool that delegates to a fresh agent
    compact_at   context size, in tokens, above which history is summarized;
                 by default 70% of the model's window, at most 150,000
    max_output   characters kept from a single tool result; by default about
                 a quarter of the model's window, at most 20,000
    """

    def __init__(
        self,
        system="You are a helpful assistant.",
        tools=(),
        *,
        model=None,
        approve=None,
        on_text=None,
        on_tool=None,
        memory=None,
        subagents=False,
        compact_at=None,
        max_output=None,
    ):
        self.model = resolve(model or os.environ.get("NANOAGENT_MODEL"))
        window = self.model.context_window
        compact_at = compact_at or min(150_000, int(window * 0.7))
        max_output = max_output or min(20_000, window)  # characters: about window / 4 tokens
        tools = list(tools)
        if memory:
            memory = Path(memory)
            if memory.exists():
                system += f"\n\nNotes from earlier sessions:\n{memory.read_text()}"
            tools.append(_remember_tool(memory))
        if subagents:
            tools.append(_task_tool(self))

        # The instructions and the tool set never change after this point.
        # A stable start of the conversation is what makes caching work.
        self.system = system
        self.tools = {each.name: each for each in tools}
        self.approve = approve
        self.on_text = on_text
        self.on_tool = on_tool
        self.compact_at = compact_at
        self.max_output = max_output
        self.messages = []
        self.stop_reason = None  # why the last reply ended: "end", "tool", "length" or "refusal"

    def run(self, request):
        """Work on a request until the model stops using tools. Returns its final text."""
        self.messages.append({"role": "user", "content": request})
        try:
            return self._loop()
        except KeyboardInterrupt:
            self._close_open_calls()
            raise

    # --- the loop -------------------------------------------------------------

    def _loop(self):
        while True:
            # Servers that run out of room drop the start of the conversation,
            # instructions included, without saying so. Make room first.
            if self._size() > self.compact_at:
                self._compact()
            reply = self.model.reply(self.system, list(self.tools.values()), self.messages, self.on_text)
            self.stop_reason = reply.stop
            calls = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in reply.calls]
            # Store the reply with the provider's own copy, which it may need
            # back unchanged (Claude's reasoning, for one). A declined request
            # can come back empty.
            if reply.text or calls or reply.raw:
                self.messages.append(
                    {"role": "assistant", "content": reply.text, "calls": calls, "raw": reply.raw, "by": self.model.key}
                )
            if calls:
                # A reply cut off by the output limit or a refusal may hold
                # incomplete calls: none of them runs.
                cut_off = reply.stop in ("length", "refusal")
                results = [self._execute(call, cut_off) for call in calls]
                self.messages.append({"role": "tool", "results": results})

            # Compact only here, when every tool call has its result.
            if max(reply.context, self._size()) > self.compact_at:
                self._compact()
            if not calls:
                return reply.text

    def _execute(self, call, cut_off):
        """Run one tool call. Whatever happens, the model gets a result back."""
        name, arguments = call["name"], call["arguments"]
        if self.on_tool:
            self.on_tool(name, arguments)
        each = self.tools.get(name)
        if cut_off:
            return self._result(call, "Your reply hit the output limit before this call was "
                                      "complete, so nothing ran. Try again with a smaller input.", True)
        if each is None:
            return self._result(call, f"There is no tool named `{name}`. Available: {', '.join(self.tools)}.", True)
        if not isinstance(arguments, dict):
            return self._result(call, f"Invalid input: expected a JSON object, got {str(arguments)[:200]!r}.", True)
        if problem := each.check(arguments):
            return self._result(call, f"Invalid input for {name}: {problem}. Schema: {json.dumps(each.schema)}", True)
        if each.approve and self.approve and not self.approve(name, arguments):
            return self._result(call, "The user declined this action. Ask them what to do instead.", True)
        try:
            return self._result(call, str(each.function(**arguments)))
        except Exception as error:  # the model reads the error and adapts
            return self._result(call, f"{type(error).__name__}: {error}", True)

    def _result(self, call, output, failed=False):
        if len(output) > self.max_output:  # keep the head and the tail
            half = self.max_output // 2
            omitted = len(output) - self.max_output
            output = f"{output[:half]}\n[... {omitted} characters omitted ...]\n{output[-half:]}"
        return {"id": call["id"], "name": call["name"], "content": output or "(no output)", "error": failed}

    # --- keeping the conversation healthy --------------------------------------

    def _size(self, messages=None):
        """Estimated tokens of the next request."""
        tools = [[t.name, t.description, t.schema] for t in self.tools.values()]
        return estimate(self.system, tools, self.messages if messages is None else messages)

    def _compact(self):
        """Replace the whole history with a handoff note written by the model."""
        if len(self.messages) < 2:
            return  # nothing to summarize: the history is already one message
        request = self.messages + [{"role": "user", "content": COMPACT_PROMPT}]
        request = self._fit(request, int(self.model.context_window * 0.8))
        # Same instructions and tools as always, so a cached prefix still hits.
        reply = self.model.reply(self.system, list(self.tools.values()), request)
        if reply.text:
            self.messages[:] = [
                {
                    "role": "user",
                    "content": "This session continues earlier work. Handoff note from "
                    f"that work:\n\n{reply.text}\n\nCarry on from where it left off.",
                }
            ]

    def _fit(self, messages, budget):
        """A copy of `messages` that fits `budget`, emptying the oldest tool results first.

        Only used for the summary request. The history itself is never edited:
        it is replaced whole by the summary.
        """
        messages = [dict(turn) for turn in messages]
        for turn in messages[:-1]:
            if self._size(messages) <= budget:
                break
            if turn["role"] == "tool":
                turn["results"] = [{**r, "content": "[omitted to fit the context window]"} for r in turn["results"]]
            elif turn["role"] in ("user", "assistant") and len(turn["content"]) > 400:
                turn["content"] = turn["content"][:200] + " [...] " + turn["content"][-200:]
                turn["raw"] = None
        return messages

    def _close_open_calls(self):
        """After an interruption, answer any tool call that never got a result.

        Most APIs reject a conversation where a call has no result, so an
        interrupted turn has to be closed before the next one starts.
        """
        last = self.messages[-1]
        if last["role"] == "assistant" and last["calls"]:
            results = [self._result(call, "Interrupted by the user.", True) for call in last["calls"]]
            self.messages.append({"role": "tool", "results": results})


def _remember_tool(path):
    @tool
    def remember(note: Annotated[str, "One self-contained sentence"]) -> str:
        """Save a lasting fact about this project or the user for future sessions."""
        with path.open("a") as file:
            file.write(f"- {note}\n")
        return f"Saved to {path}"

    return remember


def _task_tool(parent):
    @tool
    def task(prompt: Annotated[str, "The complete task description"]) -> str:
        """Hand a self-contained piece of work to a fresh agent and get back its
        final report. Good for research and side tasks whose details you do not
        need. The agent sees nothing of this conversation: put everything it
        needs in the prompt."""
        worker = Agent(
            parent.system,
            [each for each in parent.tools.values() if each.name != "task"],  # no recursion
            model=parent.model,
            approve=parent.approve,
            on_tool=parent.on_tool,
            compact_at=parent.compact_at,
            max_output=parent.max_output,
        )
        return worker.run(prompt)

    return task
