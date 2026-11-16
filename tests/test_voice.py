"""Issue #19: voice."""

import asyncio
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from nanoagent import Agent, Model, Reply, tool
from nanoagent.channels import voice as voice_module
from nanoagent.channels.voice import files, speakable, voice
from nanoagent.testing import FakeModel, call, reply, running, text

FIXTURE = Path(__file__).parent / "data" / "silence.wav"


class Ears:
    """A fake recognizer: each utterance is heard as the next scripted text."""

    def __init__(self, *heard):
        self.heard, self.audio = list(heard), []

    def __call__(self, audio):
        self.audio.append(audio)
        return self.heard.pop(0)


class Mouth:
    def __init__(self):
        self.said, self.times = [], []

    def __call__(self, text):
        self.said.append(text)
        self.times.append(time.monotonic())


async def until(condition, timeout=3):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("timed out")
        await asyncio.sleep(0.01)


def run_voice(model, heard, utterances=1, tools=(), wait=lambda mouth: mouth.said, **options):
    ears, mouth = Ears(*heard), Mouth()

    async def main():
        agent = Agent(tools=list(tools), model=model)
        agent.use(voice(listen=files([FIXTURE] * utterances), hear=ears, speak=mouth, **options))
        async with running(agent):
            await until(lambda: wait(mouth))
            await asyncio.sleep(0.05)

    asyncio.run(main())
    return ears, mouth


def test_audio_in_speech_out():
    ears, mouth = run_voice(FakeModel(reply(text("It is sunny."))), ["What's the weather?"])
    assert ears.audio == [FIXTURE.read_bytes()]
    assert mouth.said == ["It is sunny."]


def test_speaks_before_the_answer_is_finished():
    finished = []

    class Slow(Model):
        async def areply(self, system, tools, messages, on_text=None):
            on_text("First sentence. ")
            await asyncio.sleep(0.3)
            on_text("Second sentence.")
            finished.append(time.monotonic())
            return Reply(text="First sentence. Second sentence.")

    ears, mouth = run_voice(Slow(), ["hi"], wait=lambda mouth: len(mouth.said) == 2)
    assert mouth.said == ["First sentence.", "Second sentence."]
    assert mouth.times[0] < finished[0]


def test_code_and_urls_are_not_read_aloud():
    said = speakable("Run this:\n```python\nprint('x')\n```\nSee https://docs.python.org/3/library/wave.html and `ls -la`. **Done**")
    assert said == "Run this: a code block. See a link to docs.python.org and ls -la. Done"


def test_silence_is_ignored():
    model = FakeModel(reply(text("Hello.")))
    ears, mouth = run_voice(model, ["", "  ", "hello"], utterances=3)
    assert len(model.requests) == 1
    assert mouth.said == ["Hello."]


def test_missing_extra_error_names_the_install_command(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def without_whisper(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_whisper)
    with pytest.raises(RuntimeError, match=r"pip install 'nanoagent\[voice\]'"):
        voice(listen=files([]), speak=print)


def test_tool_activity_is_announced():
    @tool
    def look_up(topic: str) -> str:
        """Look something up."""
        return "found"

    model = FakeModel(reply(call("look_up", topic="tides")), reply(text("High tide at noon.")))
    ears, mouth = run_voice(model, ["tides?"], tools=[look_up], announce_after=0,
                            wait=lambda mouth: len(mouth.said) == 2)
    assert mouth.said == ["Using look up.", "High tide at noon."]


def test_error_answers_are_spoken():
    class Broken(Model):
        def reply(self, *args):
            raise RuntimeError("no model")

    ears, mouth = run_voice(Broken(), ["hi"])
    assert mouth.said == ["Sorry, something went wrong: RuntimeError: no model"]


@pytest.mark.voice
def test_real_whisper_transcribes_fixture(tmp_path):
    pytest.importorskip("faster_whisper")
    synthesizer = shutil.which("espeak-ng") or shutil.which("espeak")
    if not synthesizer:
        pytest.skip("needs espeak to make a spoken fixture")
    path = tmp_path / "spoken.wav"
    subprocess.run([synthesizer, "-w", str(path), "the weather is sunny today"], check=True)
    heard = voice_module.whisper("tiny.en")(path.read_bytes()).lower()
    assert "weather" in heard and "sunny" in heard
