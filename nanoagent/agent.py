"""The agent: a model, some tools, and a loop.

This file is the whole engine. `tutorial/` builds the same thing one idea at
a time, if you want to see why each part is here.
"""

import asyncio
import contextvars
import hashlib
import inspect
import json
import os
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Annotated

from .model import estimate, resolve
from .tools import Tool, tool

# The request being worked on, so tools such as `task` can see how it is approved.
CURRENT = contextvars.ContextVar("nanoagent_request", default=None)

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

    Abilities plug in with one line each: `agent.use(telegram())`,
    `agent.use(every("1h", "..."))`, then `agent.live()` keeps it running.

    system       instructions for the model, fixed for the agent's lifetime
    tools        functions decorated with `@tool`
    model        "provider:model" (e.g. "ollama:qwen3:4b", "anthropic:claude-opus-5-5")
                 or a `Model`; defaults to the NANOAGENT_MODEL environment variable
    approve      `approve(name, arguments) -> bool` (or async), asked before each
                 tool declared with `@tool(approve=True)`; without it they just run
    on_text      called with each piece of text as the model writes it
    on_tool      `on_tool(name, arguments)`, called when the model uses a tool
    memory       path of a notes file: loaded into the instructions, and the
                 model gets a `remember` tool that appends to it
    subagents    give the model a `task` tool that delegates to a fresh agent
    state        a directory where conversations (and trigger times) are saved,
                 so they survive restarts
    parallel     how many tool calls of one reply may run at the same time
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
        state=None,
        parallel=4,
        compact_at=None,
        max_output=None,
    ):
        self.model = resolve(model or os.environ.get("NANOAGENT_MODEL"))
        window = self.model.context_window
        self.compact_at = compact_at or min(150_000, int(window * 0.7))
        self.max_output = max_output or min(20_000, window)  # characters: about window / 4 tokens
        if memory:
            memory = Path(memory)
            if memory.exists():
                system += f"\n\nNotes from earlier sessions:\n{memory.read_text()}"
        # The instructions and the tool set never change once a conversation
        # has started. A stable start of the conversation is what makes
        # caching work, and some models reject a request where it changed.
        self.system = system
        self.tools = {}
        self.approve = approve
        self.on_text = on_text
        self.on_tool = on_tool
        self.parallel = max(1, parallel)
        self.state = Path(state).expanduser() if state else None
        self.sessions = {}  # session name -> list of messages
        self.stop_reason = None  # why the last reply ended: "end", "tool", "length" or "refusal"

        # Extension points, filled by `use()`.
        self.channels = []  # ways to talk to the agent
        self.triggers = []  # reasons for it to act on its own
        self.guards = []  # guard(name, arguments) -> "allow", "deny" or None, before approval
        self.hooks = []  # hook(event, data), for logging and replay
        self.closers = []  # called when the agent shuts down
        self.notifier = None  # set by the first channel; see notify.py
        self.runtime = None  # set while live() runs
        self.clock = datetime.now  # replaceable in tests

        self._add_tools(tools)
        if memory:
            self._add_tools([_remember_tool(memory)])
        if subagents:
            self._add_tools([_task_tool(self)])

    # --- plugging things in ------------------------------------------------------

    def use(self, thing, **options):
        """Add an ability in one line: a tool, a list of tools, a channel, a trigger or a plugin.

        Anything with an `attach(agent, **options)` method is a plugin.
        """
        if isinstance(thing, Tool):
            self._add_tools([thing])
        elif isinstance(thing, (list, tuple)) and thing and all(isinstance(each, Tool) for each in thing):
            self._add_tools(thing)
        elif hasattr(thing, "attach"):
            thing.attach(self, **options)
        else:
            raise TypeError(
                "use() takes a tool made with @tool, a list of tools, a channel, a trigger, "
                f"or a plugin (anything with an attach(agent) method), not {type(thing).__name__}"
            )
        return thing

    def notify_on(self, channel, peer=None):
        """Send unprompted messages (trigger answers, the `notify` tool) to `peer` on `channel`."""
        from .notify import enable

        enable(self).target = (channel, peer)

    def _add_tools(self, tools):
        tools = list(tools)
        if tools and any(self.sessions.values()):
            raise RuntimeError("Add tools before the agent's first run: the tool set is fixed once a conversation starts.")
        for each in tools:
            self.tools[each.name] = each

    # --- conversations -------------------------------------------------------------

    @property
    def messages(self):
        """The default conversation, the one `run` uses."""
        return self.session("default")

    @messages.setter
    def messages(self, value):
        self.sessions["default"] = value

    def session(self, name):
        """The messages of one conversation, loaded from `state` the first time."""
        if name not in self.sessions:
            self.sessions[name] = self._load(name)
        return self.sessions[name]

    def _session_file(self, name):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:60]
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        return self.state / "sessions" / f"{safe}-{digest}.json"

    def _load(self, name):
        if not self.state:
            return []
        path = self._session_file(name)
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text())["messages"]
        except (ValueError, KeyError, TypeError):
            aside = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
            path.rename(aside)
            warnings.warn(f"Session file {path} was unreadable; moved it to {aside.name} and started fresh.")
            return []

    def _save(self, name):
        if not self.state:
            return
        path = self._session_file(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps({"session": name, "messages": self.sessions[name]}, indent=1, default=str))
        temporary.replace(path)  # atomic: a crash never leaves half a file

    # --- running -------------------------------------------------------------------

    def run(self, request, **options):
        """Work on a request until the model stops using tools. Returns its final text."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(request, **options))
        raise RuntimeError("Agent.run() was called inside a running event loop. Use `await agent.arun(...)` there.")

    async def arun(self, request, *, session="default", on_text=None, approve=None):
        """The async form of `run`.

        session    which conversation this request belongs to
        on_text    overrides the agent's `on_text` for this request
        approve    overrides the agent's `approve` for this request
        """
        messages = self.session(session)
        messages.append({"role": "user", "content": request})
        context = {"session": session, "on_text": on_text or self.on_text, "approve": approve or self.approve}
        CURRENT.set(context)
        try:
            return await self._loop(messages, context)
        except (KeyboardInterrupt, asyncio.CancelledError):
            self._close_open_calls(messages)
            raise
        finally:
            self._save(session)

    def live(self, **options):
        """Run every channel and trigger until stopped (Ctrl-C or `stop()`). See runtime.py."""
        from .runtime import Runtime

        try:
            asyncio.run(Runtime(self, **options).run())
        except KeyboardInterrupt:
            pass

    async def alive(self, **options):
        """The async form of `live`."""
        from .runtime import Runtime

        await Runtime(self, **options).run()

    def stop(self):
        """Stop `live()`. Safe to call from any thread."""
        if getattr(self, "runtime", None):
            self.runtime.stop()

    def close(self):
        """Release what plugins started (servers, browsers, processes)."""
        while self.closers:
            self.closers.pop()()

    # --- the loop -------------------------------------------------------------

    async def _loop(self, messages, context):
        while True:
            # Servers that run out of room drop the start of the conversation,
            # instructions included, without saying so. Make room first.
            if self._size(messages) > self.compact_at:
                await self._compact(messages)
            tools = list(self.tools.values())
            self._emit("request", session=context["session"], system=self.system, messages=messages)
            reply = await self.model.areply(self.system, tools, messages, context["on_text"])
            self.stop_reason = reply.stop
            calls = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in reply.calls]
            self._emit("reply", session=context["session"], text=reply.text, calls=calls, stop=reply.stop,
                       context=reply.context, raw=reply.raw)
            # Store the reply with the provider's own copy, which it may need
            # back unchanged (Claude's reasoning, for one). A declined request
            # can come back empty.
            if reply.text or calls or reply.raw:
                messages.append(
                    {"role": "assistant", "content": reply.text, "calls": calls, "raw": reply.raw, "by": self.model.key}
                )
            if calls:
                # A reply cut off by the output limit or a refusal may hold
                # incomplete calls: none of them runs.
                cut_off = reply.stop in ("length", "refusal")
                results = await self._execute_all(calls, cut_off, context)
                messages.append({"role": "tool", "results": results})
            self._save(context["session"])

            # Compact only here, when every tool call has its result.
            if max(reply.context, self._size(messages)) > self.compact_at:
                await self._compact(messages)
            if not calls:
                return reply.text

    async def _execute_all(self, calls, cut_off, context):
        """Run one reply's calls. Approvals are asked first, one at a time; then the
        calls run together, up to `parallel` at once. Results keep the calls' order."""
        results = [None] * len(calls)
        ready = []  # (position, call, tool)
        for position, call in enumerate(calls):
            refusal = await self._check(call, cut_off, context)
            if refusal:
                results[position] = refusal
            else:
                ready.append((position, call, self.tools[call["name"]]))

        limit = asyncio.Semaphore(self.parallel)

        async def run(position, call, each):
            async with limit:
                results[position] = await self._invoke(call, each, context)

        await asyncio.gather(*(run(*item) for item in ready if item[2].parallel))
        for item in ready:
            if not item[2].parallel:  # these never overlap another call
                await run(*item)
        return results

    async def _check(self, call, cut_off, context):
        """The error result that stops a call before it runs, or None."""
        name, arguments = call["name"], call["arguments"]
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
        verdicts = [guard(name, arguments) for guard in self.guards]
        if "deny" in verdicts:
            return self._result(call, "A rule set by the user blocks this action. Do not retry it; "
                                      "find another way or ask the user.", True)
        if each.approve and "allow" not in verdicts and context["approve"]:
            answer = context["approve"](name, arguments)
            if inspect.isawaitable(answer):
                answer = await answer
            if not answer:
                return self._result(call, "The user declined this action. Ask them what to do instead.", True)
        return None

    async def _invoke(self, call, each, context):
        if self.on_tool:
            self.on_tool(call["name"], call["arguments"])
        self._emit("tool_start", session=context["session"], call=call)
        try:
            if inspect.iscoroutinefunction(each.function):
                output = await each.function(**call["arguments"])
            else:  # a plain function runs in a worker thread, so it never blocks the loop
                output = await asyncio.to_thread(each.function, **call["arguments"])
            result = self._result(call, output)
        except Exception as error:  # the model reads the error and adapts
            result = self._result(call, f"{type(error).__name__}: {error}", True)
        self._emit("tool", session=context["session"], call=call, result=result)
        return result

    def _result(self, call, output, failed=False):
        """A tool result. `output` is text, or a list of text and image parts (see Image)."""
        if isinstance(output, list):
            parts = [self._clip(p) if isinstance(p, str) else p.part() if hasattr(p, "part") else p for p in output]
            texts = [p for p in parts if isinstance(p, str)]
            content = parts if len(texts) != len(parts) else "\n".join(texts)
        elif hasattr(output, "part"):  # an Image
            content = [output.part()]
        else:
            content = self._clip(str(output))
        return {"id": call["id"], "name": call["name"], "content": content or "(no output)", "error": failed}

    def _clip(self, output):
        if len(output) > self.max_output:  # keep the head and the tail
            half = self.max_output // 2
            omitted = len(output) - self.max_output
            output = f"{output[:half]}\n[... {omitted} characters omitted ...]\n{output[-half:]}"
        return output

    def _emit(self, event, **data):
        for hook in self.hooks:
            hook(event, data)

    # --- keeping the conversation healthy --------------------------------------

    def _size(self, messages):
        """Estimated tokens of the next request."""
        tools = [[t.name, t.description, t.schema] for t in self.tools.values()]
        return estimate(self.system, tools, messages)

    async def _compact(self, messages):
        """Replace the whole history with a handoff note written by the model."""
        if len(messages) < 2:
            return  # nothing to summarize: the history is already one message
        request = messages + [{"role": "user", "content": COMPACT_PROMPT}]
        request = self._fit(request, int(self.model.context_window * 0.8))
        # Same instructions and tools as always, so a cached prefix still hits.
        reply = await self.model.areply(self.system, list(self.tools.values()), request)
        if reply.text:
            messages[:] = [
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

    def _close_open_calls(self, messages):
        """After an interruption, answer any tool call that never got a result.

        Most APIs reject a conversation where a call has no result, so an
        interrupted turn has to be closed before the next one starts.
        """
        last = messages[-1]
        if last["role"] == "assistant" and last["calls"]:
            results = [self._result(call, "Interrupted by the user.", True) for call in last["calls"]]
            messages.append({"role": "tool", "results": results})


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
    async def task(prompt: Annotated[str, "The complete task description"]) -> str:
        """Hand a self-contained piece of work to a fresh agent and get back its
        final report. Good for research and side tasks whose details you do not
        need. The agent sees nothing of this conversation: put everything it
        needs in the prompt."""
        current = CURRENT.get() or {}
        worker = Agent(
            parent.system,
            [each for each in parent.tools.values() if each.name != "task"],  # no recursion
            model=parent.model,
            approve=current.get("approve", parent.approve),  # asked the same way as the parent
            on_tool=parent.on_tool,
            parallel=parent.parallel,
            compact_at=parent.compact_at,
            max_output=parent.max_output,
        )
        worker.guards = parent.guards
        return await worker.arun(prompt)

    return task
