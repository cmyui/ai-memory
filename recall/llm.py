"""LLM backend — supports OpenAI API and Claude CLI subprocess."""

import asyncio

import openai

DEFAULT_OPENAI_MODEL = "gpt-4.1-nano"
CLAUDE_CLI_MODEL = "opus"

_use_claude_cli = False
_model_override: str | None = None


def set_use_claude_cli(enabled: bool) -> None:
    global _use_claude_cli
    _use_claude_cli = enabled


def set_model(model: str) -> None:
    global _model_override
    _model_override = model


async def call(
    system_prompt: str,
    content: str,
    *,
    default_model: str = DEFAULT_OPENAI_MODEL,
) -> str:
    """Call an LLM with the configured backend.

    When --use-claude-cli is set, shells out to the `claude` CLI (uses
    the user's subscription). Otherwise, calls the OpenAI API with
    the resolved model (--model override > caller default > global default).
    """
    if _use_claude_cli:
        return await _call_claude_cli(system_prompt, content)
    model = _model_override or default_model
    return await _call_openai(system_prompt, content, model=model)


async def _call_openai(system_prompt: str, content: str, *, model: str) -> str:
    client = openai.AsyncOpenAI()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""
    except openai.OpenAIError as e:
        print(f"  \u26a0 LLM error ({model}): {e}", flush=True)
        return ""


async def _call_claude_cli(system_prompt: str, content: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "claude",
        "-p",
        "--model",
        CLAUDE_CLI_MODEL,
        "--effort",
        "max",
        "--setting-sources",
        "",
        "--tools",
        "",
        "--strict-mcp-config",
        "--system-prompt",
        system_prompt,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=content.encode())
    if proc.returncode != 0:
        err = stderr.decode().strip()
        print(f"  \u26a0 Claude CLI error: {err}", flush=True)
        return ""
    return stdout.decode().strip()
