"""A 1.7B model with no tool support, working with files through guided tool calling.

    ollama pull smollm2:1.7b && python examples/tiny_model.py
"""

from nanoagent import Agent, read_file, write_file
from nanoagent.providers.ollama import Ollama

agent = Agent(
    "You are a careful assistant. Use the tools to work with files.",
    tools=[read_file, write_file],
    model=Ollama("smollm2:1.7b", tools="guided", context=4096),
)
print(agent.run("Write a haiku about autumn to haiku.txt."))
