"""The smallest agent: a model and one tool."""

import os

from nanoagent import Agent, tool


@tool
def get_weather(city: str) -> str:
    """Return today's weather in a city."""
    return "sunny, 24°C"


agent = Agent("You are a travel assistant.", tools=[get_weather], model=os.environ.get("NANOAGENT_MODEL", "ollama:qwen3:4b"))
print(agent.run("Do I need an umbrella in Rome?"))
