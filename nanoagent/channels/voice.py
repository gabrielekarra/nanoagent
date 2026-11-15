"""Voice: the agent listens and speaks, on your machine.

    agent.use(voice())          # microphone in, speakers out
    agent.live()

Three parts, each replaceable:

    listen   returns the next utterance as WAV bytes (None to stop). Default:
             push to talk on the microphone: press Enter, speak, press Enter.
    hear     turns WAV bytes into text. Default: Whisper, run locally
             through faster-whisper.
    speak    says a piece of text. Default: Piper if installed, else the
             system voice (`say` on macOS, `espeak` on Linux).

Install the defaults with: pip install 'nanoagent[voice]'

The answer is spoken a sentence at a time while the model is still writing,
code blocks and long links are summarized instead of read aloud, and if the
agent works for a while it says which tool it is using.

    agent.use(voice(listen=files(["question.wav"])))   # from audio files instead
"""

import io
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

from ..runtime import Channel

INSTALL = "pip install 'nanoagent[voice]'"
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")


def speakable(text):
    """Text as it should be said: no code, no long links, no Markdown symbols."""
    text = re.sub(r"```.*?```", " a code block. ", text, flags=re.DOTALL)
    text = re.sub(r"https?://(?:www\.)?([^/\s]+)\S*", r"a link to \1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"[*_#>|]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


class Voice(Channel):
    name = "voice"

    def __init__(self, listen=None, hear=None, speak=None, announce_after=4.0, peer="you"):
        self.listen = listen or push_to_talk()
        self.hear = hear or whisper()
        self.speak = speak or system_voice()
        self.announce_after = announce_after
        self.peer = peer
        self.default_peer = peer
        self.speech = queue.Queue()  # sentences waiting to be said, in order
        self.buffer = ""  # streamed text not yet spoken
        self.spoken_any = False
        self.started_at = None
        self.running = False

    def attach(self, agent, **options):
        super().attach(agent, **options)
        agent.hooks.append(self._hook)

    async def start(self, deliver):
        self.running = True
        threading.Thread(target=self._speaker, name="nanoagent-speaker", daemon=True).start()
        threading.Thread(target=self._listener, name="nanoagent-listener", daemon=True).start()

    async def stop(self):
        self.running = False
        self.speech.put(None)

    # --- listening -------------------------------------------------------------------

    def _listener(self):
        while self.running:
            audio = self.listen()
            if audio is None:
                return
            heard = (self.hear(audio) or "").strip()
            if not heard:
                continue  # silence or noise: nothing to answer
            self.started_at, self.spoken_any, self.buffer = time.monotonic(), False, ""
            self.post(self.peer, heard)

    # --- speaking --------------------------------------------------------------------

    def on_text(self, peer, piece):
        """Speak each sentence as soon as it is complete."""
        self.buffer += piece
        *done, self.buffer = SENTENCE_END.split(self.buffer)
        for sentence in done:
            self._say(sentence)

    async def send(self, peer, text):
        if self.spoken_any or self.buffer:
            self._say(self.buffer)  # whatever is left of the streamed answer
        else:
            self._say(text)  # nothing was streamed (an error, a guided answer)
        self.buffer = ""

    def _say(self, text):
        text = speakable(text)
        if text:
            self.spoken_any = True
            self.speech.put(text)

    def _speaker(self):
        while True:
            text = self.speech.get()
            if text is None:
                return
            try:
                self.speak(text)
            except Exception as error:  # a voice failure must not kill the agent
                print(f"nanoagent: could not speak: {error}", file=sys.stderr)

    def _hook(self, event, data):
        if event == "tool_start" and self.started_at and time.monotonic() - self.started_at >= self.announce_after:
            self.speech.put(f"Using {data['call']['name'].replace('_', ' ')}.")


# --- default parts ------------------------------------------------------------------


def files(paths):
    """`listen` from audio files, one utterance each, then stop."""
    pending = [Path(path) for path in paths]

    def listen():
        return pending.pop(0).read_bytes() if pending else None

    return listen


def push_to_talk(rate=16_000):
    """`listen` on the microphone: Enter to start, Enter to stop."""
    try:
        import numpy  # noqa: F401
        import sounddevice
    except ImportError:
        return _missing("the microphone needs sounddevice and numpy")

    def listen():
        try:
            input("Press Enter and speak; press Enter again when done. ")
        except EOFError:
            return None
        frames, done = [], threading.Event()

        def record(data, *_):
            frames.append(bytes(data))

        with sounddevice.RawInputStream(samplerate=rate, channels=1, dtype="int16", callback=record):
            threading.Thread(target=lambda: (input(), done.set()), daemon=True).start()
            done.wait()
        return to_wav(b"".join(frames), rate)

    return listen


def to_wav(pcm, rate=16_000):
    output = io.BytesIO()
    with wave.open(output, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(pcm)
    return output.getvalue()


def whisper(size="base", language=None):
    """`hear` with Whisper, locally, through faster-whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return _missing("speech recognition needs faster-whisper")
    model = None

    def hear(audio):
        nonlocal model
        model = model or WhisperModel(size, compute_type="int8")
        with tempfile.NamedTemporaryFile(suffix=".wav") as file:
            file.write(audio)
            file.flush()
            segments, _ = model.transcribe(file.name, language=language, vad_filter=True)
            return " ".join(segment.text.strip() for segment in segments)

    return hear


def piper(voice_model):
    """`speak` with Piper, given the path of a voice model (.onnx)."""
    command = shutil.which("piper")
    if not command:
        raise RuntimeError(f"Piper is not installed: {INSTALL}, or install piper and put it on PATH.")

    def speak(text):
        with tempfile.NamedTemporaryFile(suffix=".wav") as file:
            subprocess.run([command, "--model", str(voice_model), "--output_file", file.name],
                           input=text.encode(), check=True, capture_output=True)
            _play(file.name)

    return speak


def system_voice():
    """`speak` with the operating system's own voice: no install needed on macOS."""
    for command in (["say"], ["espeak-ng"], ["espeak"], ["spd-say", "--wait"]):
        if shutil.which(command[0]):
            return lambda text, command=command: subprocess.run([*command, text], check=False, capture_output=True)
    return _missing("no voice found: install espeak-ng, or pass speak=piper('voice.onnx')")


def _play(path):
    for command in (["afplay"], ["aplay", "-q"], ["paplay"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if shutil.which(command[0]):
            subprocess.run([*command, path], check=False)
            return
    raise RuntimeError("no audio player found (afplay, aplay, paplay or ffplay)")


def _missing(what):
    raise RuntimeError(f"Voice: {what}. Install the defaults with: {INSTALL}")


def voice(listen=None, hear=None, speak=None, **options):
    """A voice channel. See the module docstring."""
    return Voice(listen, hear, speak, **options)
