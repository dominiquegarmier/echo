from __future__ import annotations

from collections.abc import Generator
from contextlib import redirect_stdout
from io import StringIO
from typing import NamedTuple
from typing import NoReturn

from litellm import completion
from rich.console import Console


console = Console()


MODEL = "ollama/mistral"
API_BASE = "http://localhost:11434"

PROMPT = """\
Your Name is Echo. You are a helpful assistant that knows how to
programm in python. In order to give better answers you are equipped
with the ability to generate and execute python code. The code that
is executed is contained in markdown code blocks of the form

```python
# code goes here
```

The User will then be able to authorize the execution of the code.

If the user does not authorize the execution of the code you will simply
get "ABORTED" as a result. If the code is executed you will get the
stdout of the above code.

Dont add python code blocks to your response unless you want to execute them.
"""
PYTHON_CODE_BLOCK = "```python\n"
CODE_BLOCK = "```\n"


class Message(NamedTuple):
    content: str
    role: str = "user"


def generate(chat: list[Message]) -> Generator[str, None, None]:
    messages = [{"content": m.content, "role": m.role} for m in chat]
    response = completion(
        model="ollama/llama2",
        messages=messages,
        api_base="http://localhost:11434",
        stream=True,
    )
    for chunk in response:
        yield chunk["choices"][0]["delta"].content  # type: ignore


def initialize_chat() -> list[Message]:
    prompt = Message(
        content=PROMPT,
        role="system",
    )
    return [prompt]


def parse_code(message: Message) -> str | None:
    content = message.content
    try:
        content = content.split(PYTHON_CODE_BLOCK)[1]
        return content.split(CODE_BLOCK)[0]
    except IndexError:
        return None


def run_code(code: str) -> str:
    stdout = StringIO()
    with redirect_stdout(stdout):
        try:
            exec(code)
        except Exception as e:
            print(e)
    return stdout.getvalue()


def handle_code(code: str) -> Message:
    styled("\nPYTHON CODE DETECTED\n", style="bold red")
    styled(f"{PYTHON_CODE_BLOCK}{code}{CODE_BLOCK}\n", style="red")
    styled("EXECUTE CODE? (y/n)", style="bold red")
    answer = input()
    if answer == "y":
        styled("EXECUTING CODE\n", style="bold red")
        result = run_code(code)
    else:
        result = "ABORTED"

    wrapped = f"{CODE_BLOCK}{result}{CODE_BLOCK}\n"
    styled(wrapped, style="red")

    return Message(content=wrapped, role="system")


def styled(contents: str, style: str) -> None:
    console.print(contents, style=style, end="")
    return None


def main() -> NoReturn:
    styled("Welcome to Echo!\n\n", style="bold blue")

    chat = initialize_chat()
    while True:
        styled("> ", style="bold green")
        promp = input()

        message = Message(content=promp, role="user")
        chat.append(message)

        styled("Echo: ", style="bold blue")
        response = ""
        for text in generate(chat):
            styled(text, style="blue")
            response += text
        styled("\n", style="blue")

        message = Message(content=response, role="assistant")
        chat.append(message)

        # find python code
        code = parse_code(message)
        if code is not None:
            message = handle_code(code)
            chat.append(message)


if __name__ == "__main__":
    raise SystemExit(main())
