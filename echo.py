from __future__ import annotations

from collections.abc import Generator
from collections.abc import MutableSequence
from contextlib import redirect_stdout
from io import StringIO
from typing import NamedTuple

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

Dont add python code blocks to your response unless you want to execute them.
If it makes sense you should use code to find the answer to the question.
Make Your answer as brief as possible.
If not absolutely necessary say nothing else than the code.
"""

RESPONSE_PROMPT = """\
the stdout of the code above was:

```
****
```

I should quickly explain the answer I got...
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


def _parse_code(message: Message) -> str | None:
    content = message.content + "\n"
    try:
        content = content.split(PYTHON_CODE_BLOCK)[1]
        return content.split(CODE_BLOCK)[0]
    except IndexError:
        return None


def _exec_code(code: str) -> str:
    stdout = StringIO()
    with redirect_stdout(stdout):
        try:
            exec(code)
        except Exception as e:
            print(e)
    return stdout.getvalue()


def handle_code(code: str) -> Message | None:
    styled("\nPYTHON CODE DETECTED\n", style="bold red")
    styled(f"{PYTHON_CODE_BLOCK}{code}{CODE_BLOCK}\n", style="red")
    styled("EXECUTE CODE? (y/n): ", style="bold red")

    if input() in ["y", "Y"]:
        styled("EXECUTING CODE...\n", style="bold red")
        result = _exec_code(code)
        wrapped = RESPONSE_PROMPT.replace("****", result)
        styled(wrapped, style="red")
        return Message(content=wrapped, role="assistant")
    else:
        return None


def reply_loop(chat: MutableSequence[Message]) -> None:
    while True:
        styled("Echo: ", style="bold blue")
        response = ""
        for text in generate(list(chat)):
            styled(text, style="blue")
            response += text
        styled("\n", style="blue")

        message = Message(content=response, role="assistant")
        chat.append(message)

        code = _parse_code(message)
        if code is not None:
            message = handle_code(code)
            if message is None:
                break
            chat.append(message)
        else:
            break


def styled(contents: str, style: str) -> None:
    console.print(contents, style=style, end="")
    return None


def main() -> int:
    styled("Welcome to Echo!\n\n", style="bold blue")

    chat = initialize_chat()
    try:
        while True:
            styled("> ", style="bold green")
            promp = input()

            message = Message(content=promp, role="user")
            chat.append(message)
            reply_loop(chat)

    except KeyboardInterrupt:
        styled("\n\nGoodbye!\n", style="bold blue")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
