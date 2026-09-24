from dataclasses import dataclass
from types import SimpleNamespace

@dataclass
class FakeFunction:
    name: str
    arguments: str

class FakeToolCall:
    def __init__(self, call_id, name, arguments="{}"):
        self.id = call_id
        self.function = FakeFunction(name, arguments)
    def model_dump(self):
        return {"id": self.id, "type": "function", "function": {"name": self.function.name, "arguments": self.function.arguments}}

def assistant(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])

def tool_call(call_id, name, arguments="{}"):
    return FakeToolCall(call_id, name, arguments)
