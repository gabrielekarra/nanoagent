"""Issue #23: the browser."""

import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("playwright")

from nanoagent import Agent  # noqa: E402
from nanoagent.browser import UNTRUSTED, browser  # noqa: E402
from nanoagent.testing import FakeModel, call, reply, text  # noqa: E402

CHROMIUM = "/opt/pw-browsers/chromium"
pytestmark = pytest.mark.browser

PAGES = {
    "/": """<html><head><title>Home</title><style>.x{color:red}</style><script>var secret = 1;</script></head>
            <body><h1>Welcome</h1><p>Hello there.</p><a href="/about">About us</a>
            <div style="display:none">invisible text</div></body></html>""",
    "/about": "<html><head><title>About</title></head><body><h2>About</h2><p>We make tea.</p></body></html>",
    "/form": """<html><head><title>Sign up</title></head><body><form method="post" action="/thanks">
            <input name="email" placeholder="Email"><button type="submit">Sign up</button></form></body></html>""",
    "/long": "<html><head><title>Long</title></head><body>" + "".join(f"<p>Paragraph {n} of a long page.</p>" for n in range(800)) + "</body></html>",
}


@pytest.fixture
def site():
    posted = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self._page(PAGES.get(self.path, "<html><body>missing</body></html>"))

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"])).decode()
            posted.append(dict(urllib.parse.parse_qsl(body)))
            self._page("<html><head><title>Thanks</title></head><body><p>Thanks for signing up.</p></body></html>")

        def _page(self, html):
            data = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", posted
    server.shutdown()


def run(script, approve=None, images=False):
    model = FakeModel(*script, images=images)
    agent = Agent(model=model, approve=approve)
    executable = CHROMIUM if Path(CHROMIUM).exists() else None
    agent.use(browser(executable=executable))
    try:
        agent.run("use the browser")
    finally:
        agent.close()
    return model, agent


def test_read_page_as_text(site):
    url, _ = site
    model, _ = run([reply(call("open_page", url=url + "/")), reply(text("ok"))])
    page = model.tool_results()[0]["content"]
    assert '<page title="Home"' in page
    assert "# Welcome" in page and "Hello there." in page
    assert "[1] link: About us" in page
    assert "secret" not in page and "color:red" not in page and "invisible" not in page


def test_click_follows_link(site):
    url, _ = site
    model, _ = run([reply(call("open_page", url=url + "/")), reply(call("click", element=1)), reply(text("ok"))])
    page = model.tool_results(2)[0]["content"]
    assert 'title="About"' in page and "We make tea." in page


def test_fill_and_submit_form(site):
    url, posted = site
    model, _ = run([
        reply(call("open_page", url=url + "/form")),
        reply(call("type_text", element=1, text="ada@example.com")),
        reply(call("submit", element=2)),
        reply(text("ok")),
    ], approve=lambda name, arguments: True)
    assert posted == [{"email": "ada@example.com"}]
    assert "Thanks for signing up." in model.tool_results(3)[0]["content"]


def test_submit_needs_approval(site):
    url, posted = site
    asked = []
    model, agent = run([
        reply(call("open_page", url=url + "/form")),
        reply(call("click", element=2)),
        reply(call("submit", element=2)),
        reply(text("ok")),
    ], approve=lambda name, arguments: asked.append(name) or False)
    assert agent.tools["submit"].approve
    assert "use submit(2)" in model.tool_results(2)[0]["content"]
    assert asked == ["submit"] and posted == []


def test_page_text_fits_budget(site):
    url, _ = site
    model, _ = run([reply(call("open_page", url=url + "/long")), reply(call("read_page", start=8000)), reply(text("ok"))])
    first, second = model.tool_results(1)[0]["content"], model.tool_results(2)[0]["content"]
    assert len(first) < 9000 and "read_page(start=8000)" in first
    assert "Paragraph 0 " in first and "Paragraph 0 " not in second


def test_page_content_is_marked_untrusted(site):
    url, _ = site
    model, _ = run([reply(call("open_page", url=url + "/about")), reply(text("ok"))])
    page = model.tool_results()[0]["content"]
    assert page.rstrip().endswith(UNTRUSTED) and "</page>" in page


def test_screenshot_only_for_models_that_see(site):
    url, _ = site
    model, agent = run([reply(call("open_page", url=url + "/")), reply(call("screenshot")), reply(text("ok"))], images=True)
    content = model.tool_results(2)[0]["content"]
    assert content[1]["type"] == "image" and content[1]["media_type"] == "image/png"
    blind = Agent(model=FakeModel(images=False))
    blind.use(browser())
    assert "screenshot" not in blind.tools


def test_missing_extra_error_names_the_install_command(monkeypatch):
    import builtins

    real = builtins.__import__

    def without(name, *args, **kwargs):
        if name == "playwright":
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without)
    with pytest.raises(RuntimeError, match=r"pip install 'nanoagent\[browser\]'"):
        browser()
