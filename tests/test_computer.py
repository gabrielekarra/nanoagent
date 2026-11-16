"""Issue #24: computer control."""

import pytest

from nanoagent import Agent
from nanoagent.computer import computer
from nanoagent.testing import FakeModel, call, reply, text


class FakeScreen:
    def __init__(self):
        self.actions = []

    def screenshot(self):
        return b"\x89PNG screen " + str(len(self.actions)).encode()

    def click(self, x, y, button="left"):
        self.actions.append(("click", x, y, button))

    def type(self, text):
        self.actions.append(("type", text))

    def press(self, keys):
        self.actions.append(("press", keys))

    def scroll(self, amount):
        self.actions.append(("scroll", amount))


def run(script, approve=lambda name, arguments: True, **options):
    screen, model = FakeScreen(), FakeModel(*script, images=True)
    agent = Agent(model=model, approve=approve)
    agent.use(computer(screen, **options))
    agent.run("use the computer")
    return screen, model, agent


def test_screenshot_is_sent_as_image():
    screen, model, _ = run([reply(call("screenshot")), reply(text("ok"))])
    content = model.tool_results()[0]["content"]
    assert content[0] == "The screen now:"
    assert content[1]["type"] == "image" and content[1]["media_type"] == "image/png"


def test_click_and_type_reach_the_backend(capsys):
    screen, model, _ = run([
        reply(call("click", x=120, y=48)),
        reply(call("type_text", text="Hello, world")),
        reply(call("press", keys="Ctrl+S")),
        reply(call("scroll", amount=-3)),
        reply(text("ok")),
    ])
    assert screen.actions == [("click", 120, 48, "left"), ("type", "Hello, world"), ("press", ["ctrl", "s"]), ("scroll", -3)]
    assert "controlling this computer" in capsys.readouterr().err
    assert model.tool_results(1)[0]["content"][0] == "Clicked left at (120, 48). The screen now:"


def test_every_action_needs_approval_by_default():
    asked = []
    screen, model, agent = run(
        [reply(call("screenshot")), reply(call("click", x=1, y=1)), reply(call("type_text", text="x")), reply(text("ok"))],
        approve=lambda name, arguments: asked.append(name) or False,
    )
    assert asked == ["click", "type_text"]
    assert screen.actions == []
    assert all(agent.tools[name].approve for name in ("click", "type_text", "press", "scroll"))


def test_approval_can_be_turned_off():
    agent = Agent(model=FakeModel(images=True))
    agent.use(computer(FakeScreen(), approve=False))
    assert not agent.tools["click"].approve


def test_text_only_model_gets_a_clear_error():
    with pytest.raises(ValueError, match="needs a model that accepts images"):
        Agent(model=FakeModel(images=False)).use(computer(FakeScreen()))


def test_missing_extra_error_names_the_install_command(monkeypatch):
    import builtins

    real = builtins.__import__

    def without(name, *args, **kwargs):
        if name == "pyautogui":
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without)
    with pytest.raises(RuntimeError, match=r"pip install 'nanoagent\[computer\]'"):
        computer()
