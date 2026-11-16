"""Computer control: the agent sees the screen and uses the mouse and keyboard. Experimental.

    agent.use(computer())

For apps with neither an API nor a web page. The model must accept images
(`model.images`); adding this to a text-only model fails right away. Every
action asks for approval unless `approve=False`, and a notice is printed
when the agent takes control. Small local vision models are not reliable
at this yet: run `nanoagent bench` with your model before trusting it.

The default backend uses pyautogui: pip install 'nanoagent[computer]'.
Anything with the same five methods can replace it (see `Screen`).
"""

import io
import sys

from .tools import Image, tool

INSTALL = "pip install 'nanoagent[computer]'"


class Screen:
    """What computer() needs from the machine. The default uses pyautogui."""

    def __init__(self):
        try:
            import pyautogui
        except ImportError:
            raise RuntimeError(f"Computer control needs pyautogui: {INSTALL}") from None
        self.gui = pyautogui

    def screenshot(self):
        output = io.BytesIO()
        self.gui.screenshot().save(output, format="PNG")
        return output.getvalue()

    def click(self, x, y, button="left"):
        self.gui.click(x, y, button=button)

    def type(self, text):
        self.gui.write(text, interval=0.02)

    def press(self, keys):
        self.gui.hotkey(*keys)

    def scroll(self, amount):
        self.gui.scroll(amount)


class Computer:
    def __init__(self, screen=None, approve=True):
        self.screen = screen or Screen()
        self.approve = approve
        self.announced = False

    def attach(self, agent):
        if not getattr(agent.model, "images", False):
            raise ValueError(
                "computer() needs a model that accepts images: the agent works from screenshots. "
                "Use a vision model (and images=True for OpenAI-compatible and Ollama models)."
            )
        agent._add_tools(self._tools())

    def _look(self, what):
        return [what, Image(self.screen.screenshot())]

    def _take_control(self):
        if not self.announced:
            print("nanoagent: the agent is now controlling this computer. Press Ctrl-C to stop it.", file=sys.stderr)
            self.announced = True

    def _tools(self):
        computer = self

        def screenshot() -> list:
            """See the screen."""
            return computer._look("The screen now:")

        def click(x: int, y: int, button: str = "left") -> list:
            """Click at a point of the screen, in screenshot pixels. button: left, right or middle."""
            if button not in ("left", "right", "middle"):
                raise ValueError("button must be left, right or middle")
            computer._take_control()
            computer.screen.click(x, y, button)
            return computer._look(f"Clicked {button} at ({x}, {y}). The screen now:")

        def type_text(text: str) -> list:
            """Type text where the keyboard focus is."""
            computer._take_control()
            computer.screen.type(text)
            return computer._look("Typed. The screen now:")

        def press(keys: str) -> list:
            """Press a key or a combination, e.g. "enter", "ctrl+s", "cmd+tab"."""
            computer._take_control()
            computer.screen.press([key.strip() for key in keys.lower().split("+") if key.strip()])
            return computer._look(f"Pressed {keys}. The screen now:")

        def scroll(amount: int) -> list:
            """Scroll: positive up, negative down."""
            computer._take_control()
            computer.screen.scroll(amount)
            return computer._look(f"Scrolled {amount}. The screen now:")

        return [tool(screenshot, parallel=False)] + [
            tool(action, approve=self.approve, parallel=False) for action in (click, type_text, press, scroll)
        ]


def computer(screen=None, approve=True):
    """Screen, mouse and keyboard control for vision models. See the module docstring."""
    return Computer(screen, approve)
