"""The agent: a model, some tools, and a loop.

This file is the whole engine. `tutorial/` builds the same thing one idea at
a time, if you want to see why each part is here.
"""

from pathlib import Path
from typing import Annotated

import anthropic

from .tools import tool

COMPACT_PROMPT = (
    "Pause the task. Without calling any tool, write a handoff note for an agent "
    "that will continue this work with no access to the conversation above: the "
    "user's goal, what is done, what is left, and every file path, command and "
    "decision it needs."
)


class Agent:
    """A model that works on a request with tools until it is done.

        agent = Agent("You are a travel assistant.", tools=[get_weather])
        answer = agent.run("Do I need an umbrella in Rome?")

    `run` can be called again: the agent keeps the conversation in
    `agent.messages`.

    system       instructions for the model, fixed for the agent's lifetime
    tools        functions decorated with `@tool`
    model        any Claude model id
    client       an `anthropic.Anthropic` client; made from the environment if omitted
    approve      `approve(name, arguments) -> bool`, asked before each tool
                 declared with `@tool(approve=True)`; without it they just run
    on_text      called with each piece of text as the model writes it
    on_tool      `on_tool(name, arguments)`, called when the model uses a tool
    memory       path of a notes file: loaded into the instructions, and the
                 model gets a `remember` tool that appends to it
    subagents    give the model a `task` tool that delegates to a fresh agent
    compact_at   context size, in tokens, above which history is summarized
    max_output   characters kept from a single tool result
    """

    def __init__(
        self,
        system="You are a helpful assistant.",
        tools=(),
        *,
        model="claude-opus-5-5",
        client=None,
        approve=None,
        on_text=None,
        on_tool=None,
        memory=None,
        subagents=False,
        compact_at=150_000,
        max_output=20_000,
        max_tokens=64_000,
    ):
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
        self.model = model
        self.client = client or anthropic.Anthropic()
        self.approve = approve
        self.on_text = on_text
        self.on_tool = on_tool
        self.compact_at = compact_at
        self.max_output = max_output
        self.max_tokens = max_tokens
        self.messages = []
        self.stop_reason = None  # why the last reply ended, e.g. "end_turn" or "refusal"

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
            response = self._ask()
            self.stop_reason = response.stop_reason
            # Store the reply exactly as it came back: it also carries the
            # model's tool calls and reasoning. A declined request can be empty.
            if response.content:
                self.messages.append({"role": "assistant", "content": response.content})

            calls = [block for block in response.content if block.type == "tool_use"]
            if calls:
                # Any stop reason other than "tool_use" means the reply ended
                # early and the calls in it may be incomplete.
                cut_off = response.stop_reason != "tool_use"
                results = [self._execute(call, cut_off) for call in calls]
                self.messages.append({"role": "user", "content": results})

            # Compact only here, when every tool call has its result.
            usage = response.usage
            size = (
                usage.input_tokens
                + (usage.cache_read_input_tokens or 0)
                + (usage.cache_creation_input_tokens or 0)
                + usage.output_tokens
            )
            if size > self.compact_at:
                self._compact()
            if not calls:
                return "".join(block.text for block in response.content if block.type == "text")

    def _ask(self):
        """One streamed model call."""
        for attempt in (1, 2):
            try:
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self.system,
                    tools=[each.spec for each in self.tools.values()],
                    messages=self.messages,
                    cache_control={"type": "ephemeral"},  # reuse the unchanged prefix
                ) as stream:
                    for text in stream.text_stream:
                        if self.on_text:
                            self.on_text(text)
                    return stream.get_final_message()
            except ValueError:  # a streamed tool input was not parseable JSON
                if attempt == 2:
                    raise

    def _execute(self, call, cut_off):
        """Run one tool call. Whatever happens, the model gets a result back."""
        if self.on_tool:
            self.on_tool(call.name, call.input)
        each = self.tools.get(call.name)
        if cut_off:
            return self._result(call, "Your reply hit the output limit before this call was "
                                      "complete, so nothing ran. Try again with a smaller input.", True)
        if each is None:
            return self._result(call, f"There is no tool named `{call.name}`.", True)
        if problem := each.check(call.input):
            return self._result(call, f"Invalid input: {problem}.", True)
        if each.approve and self.approve and not self.approve(call.name, call.input):
            return self._result(call, "The user declined this action. Ask them what to do instead.", True)
        try:
            return self._result(call, str(each.function(**call.input)))
        except Exception as error:  # the model reads the error and adapts
            return self._result(call, f"{type(error).__name__}: {error}", True)

    def _result(self, call, output, failed=False):
        if len(output) > self.max_output:  # keep the head and the tail
            half = self.max_output // 2
            omitted = len(output) - self.max_output
            output = f"{output[:half]}\n[... {omitted} characters omitted ...]\n{output[-half:]}"
        return {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": output or "(no output)",
            "is_error": failed,
        }

    # --- keeping the conversation healthy --------------------------------------

    def _compact(self):
        """Replace the whole history with a handoff note written by the model."""
        reply = self.client.messages.create(
            model=self.model,
            max_tokens=16_000,
            system=self.system,
            tools=[each.spec for each in self.tools.values()],
            messages=self.messages + [{"role": "user", "content": COMPACT_PROMPT}],
        )
        note = "".join(block.text for block in reply.content if block.type == "text")
        if note:
            self.messages[:] = [
                {
                    "role": "user",
                    "content": "This session continues earlier work. Handoff note from "
                    f"that work:\n\n{note}\n\nCarry on from where it left off.",
                }
            ]

    def _close_open_calls(self):
        """After an interruption, answer any tool call that never got a result.

        The API rejects a conversation where a call has no result, so an
        interrupted turn has to be closed before the next one starts.
        """
        last = self.messages[-1]
        if last["role"] == "assistant":
            calls = [block for block in last["content"] if block.type == "tool_use"]
            if calls:
                results = [self._result(call, "Interrupted by the user.", True) for call in calls]
                self.messages.append({"role": "user", "content": results})


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
            client=parent.client,
            approve=parent.approve,
            on_tool=parent.on_tool,
            compact_at=parent.compact_at,
            max_output=parent.max_output,
            max_tokens=parent.max_tokens,
        )
        return worker.run(prompt)

    return task
