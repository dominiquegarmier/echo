from __future__ import annotations

from collections.abc import Generator
from collections.abc import MutableSequence
from contextlib import redirect_stdout
from io import StringIO
from textwrap import dedent
from typing import NamedTuple

from litellm import completion
from rich.console import Console


console = Console()


MODEL = "ollama/codellama"
API_BASE = "http://localhost:11434"

PROMPT = """\
Your Name is Echo.

To better answer questions you are equipped with the ability to generate
and execute python code. The user will decide whether to execute the code.

The code has to be given in markdown code blocks of the form:

```python
# code goes here
```

Only write python code if you want to execute it.
Make your answer as brief as possible.
Unless absolutely necessary say nothing else than the code.
"""

PYTHON_CODE_BLOCK = "```python"
CODE_BLOCK = "```"


class Message(NamedTuple):
    content: str
    role: str = "user"


def styled(contents: str, style: str) -> None:
    console.print(contents, style=style, end="")
    return None


def format_exec_response(result) -> Message:
    ret = f"""\
    the code returned:
    {CODE_BLOCK}
    {result}
    {CODE_BLOCK}
    what does this mean with respect to the question?
    """
    return Message(content=dedent(ret), role="user")


def parse_code(message: Message) -> str | None:
    content = message.content + "\n"
    try:
        content = content.split(PYTHON_CODE_BLOCK + "\n")[1]
        return content.split(CODE_BLOCK + "\n")[0]
    except IndexError:
        return None


def exec_code(code: str) -> str:
    stdout = StringIO()
    with redirect_stdout(stdout):
        try:
            exec(code)
        except Exception as e:
            print(e)
    return stdout.getvalue()


def handle_code(code: str) -> tuple[Message, bool]:
    styled("\nPYTHON CODE DETECTED\n", style="bold red")
    styled(f"{PYTHON_CODE_BLOCK}\n{code}{CODE_BLOCK}\n\n", style="red")
    styled("EXECUTE CODE? (y/n): ", style="bold red")

    if input() in ["y", "Y"]:
        styled("EXECUTING CODE...\n", style="bold red")
        return format_exec_response(exec_code(code)), False
    else:
        err = RuntimeError("User chose not to execute code.")
        return format_exec_response(err), True


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


def reply(chat: MutableSequence[Message]) -> None:
    styled("echo: ", style="bold green")

    agg = ""
    for text in generate(list(chat)):
        styled(text, style="blue")
        agg += text

    styled("\n", style="blue")
    message = Message(content=agg, role="assistant")
    chat.append(message)

    code = parse_code(message)
    if code is not None:
        message, _ = handle_code(code)
        styled(message.content, style="bold red")
        chat.append(message)


def main() -> int:
    styled("Welcome to Echo!\n\n", style="bold blue")
    chat = initialize_chat()
    try:
        while True:
            styled("> ", style="bold green")
            message = Message(content=input(), role="user")
            chat.append(message)
            reply(chat)
    except KeyboardInterrupt:
        styled("\n\nGoodbye!\n", style="bold blue")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
