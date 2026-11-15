"""Triggers: reasons for the agent to act on its own.

    agent.use(every("1h", "Check the build and tell me if it is red"))
    agent.use(at("09:00", "Summarize my calendar for today"))
    agent.use(at("mon 09:00", "Plan the week"))

The prompt runs as an event in the runtime and the answer goes to the
channel added with `notify=True`. With a `state` directory, the last run is
saved: a restart does not run a trigger twice, and a run missed while the
agent was down happens once when it comes back (`catch_up=False` skips it).
"""

import hashlib
import json
import re
from datetime import datetime, timedelta
from datetime import time as clock_time

UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
GROUPS = {"weekdays": DAYS[:5], "weekends": DAYS[5:], "daily": DAYS}


def parse_period(text):
    """"15m", "1h", "1h30m", "2d" -> timedelta."""
    parts = re.findall(r"(\d+)\s*([smhd])", text.strip().lower())
    if not parts or re.sub(r"(\d+)\s*([smhd])", "", text.strip().lower()).strip():
        raise ValueError(f"Cannot read the period {text!r}: use e.g. '15m', '1h', '1h30m' or '2d'.")
    seconds = sum(int(number) * UNITS[unit] for number, unit in parts)
    if seconds < 1:
        raise ValueError("The period must be at least one second.")
    return timedelta(seconds=seconds)


def parse_time(text):
    """"09:00", "mon 09:00", "mon,thu 18:30", "weekdays 08:00" -> (days, time)."""
    words = text.strip().lower().split()
    if len(words) not in (1, 2):
        raise ValueError(f"Cannot read the time {text!r}: use e.g. '09:00' or 'mon 09:00'.")
    days = DAYS
    if len(words) == 2:
        days = []
        for name in words[0].split(","):
            if name in GROUPS:
                days += GROUPS[name]
            elif name[:3] in DAYS:
                days.append(name[:3])
            else:
                raise ValueError(f"Unknown day {name!r}: use mon..sun, weekdays, weekends or daily.")
    try:
        moment = clock_time.fromisoformat(words[-1])
    except ValueError:
        raise ValueError(f"Cannot read the time {words[-1]!r}: use HH:MM.") from None
    return [DAYS.index(day) for day in days], moment


class Trigger:
    grace = timedelta(minutes=2)  # how late a run may be and still count as on time
    kind = "trigger"

    def __init__(self, spec, prompt, catch_up=True, name=None):
        self.spec = spec
        self.prompt = prompt
        self.catch_up = catch_up
        digest = hashlib.sha1(f"{self.kind} {spec} {prompt}".encode()).hexdigest()[:8]
        self.name = name or f"{self.kind}-{digest}"
        self.anchor = None  # when the trigger first started (for periods)
        self.last = None  # the last scheduled time that was handled
        self.agent = None

    def attach(self, agent):
        self.agent = agent
        agent.triggers.append(self)
        saved = self._read().get(self.name, {})
        self.anchor = datetime.fromisoformat(saved["anchor"]) if saved.get("anchor") else None
        self.last = datetime.fromisoformat(saved["last"]) if saved.get("last") else None

    def check(self, now):
        """The prompts to run at `now`: none, or this trigger's prompt once."""
        if self.last is None:  # first start ever: begin counting from now
            self.anchor = self.anchor or now
            self.last = now
            self._write()
            return []
        due = self.previous(now)
        if due is None or due <= self.last:
            return []
        self.last = due
        self._write()
        return [self.prompt] if self.catch_up or now - due <= self.grace else []

    def previous(self, now):
        raise NotImplementedError

    # --- persistence ------------------------------------------------------------

    def _file(self):
        state = getattr(self.agent, "state", None)
        return state / "triggers.json" if state else None

    def _read(self):
        path = self._file()
        if not path or not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except ValueError:
            return {}

    def _write(self):
        path = self._file()
        if not path:
            return
        saved = self._read()
        saved[self.name] = {"anchor": self.anchor and self.anchor.isoformat(), "last": self.last.isoformat()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(saved, indent=1))


class Every(Trigger):
    kind = "every"

    def __init__(self, period, prompt, **options):
        self.period = parse_period(period)
        super().__init__(period, prompt, **options)

    def previous(self, now):
        steps = (now - self.anchor) // self.period
        return self.anchor + steps * self.period if steps >= 1 else None


class At(Trigger):
    kind = "at"

    def __init__(self, when, prompt, **options):
        self.days, self.time = parse_time(when)
        super().__init__(when, prompt, **options)

    def previous(self, now):
        for back in range(8):
            day = now.date() - timedelta(days=back)
            moment = datetime.combine(day, self.time)
            if day.weekday() in self.days and moment <= now:
                return moment
        return None


def every(period, prompt, *, catch_up=True, name=None):
    """Run `prompt` every `period`: "15m", "1h", "1h30m", "2d"."""
    return Every(period, prompt, catch_up=catch_up, name=name)


def at(when, prompt, *, catch_up=True, name=None):
    """Run `prompt` at a time of day: "09:00", "mon 09:00", "weekdays 08:30"."""
    return At(when, prompt, catch_up=catch_up, name=name)
