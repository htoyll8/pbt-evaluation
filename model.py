import os
import re
import anthropic
from anthropic import AnthropicVertex
from openai import OpenAI

# Provider selection. Each vendor is called directly by default: Anthropic SDK
# for Claude (Vertex when ANTHROPIC_VERTEX_PROJECT is set), else the OpenAI
# Responses API. Set OPENROUTER_API_KEY to route every model through OpenRouter
# instead; PBT_PROVIDER=openrouter|direct overrides the key-based auto-detect.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter namespaces model IDs by vendor. Map this project's bare names to
# their slugs; a name already containing "/" passes through, so callers can
# always supply an exact slug.
OPENROUTER_MODEL_IDS = {
    "claude-sonnet-4-5": "anthropic/claude-sonnet-4.5",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
    "gpt-5.1": "openai/gpt-5.1",
    "gpt-5.5": "openai/gpt-5.5",
    "gpt-5.3-codex": "openai/gpt-5.3-codex",  # latest codex; no 5.5-codex exists
    "gpt-4": "openai/gpt-4",
    "gpt-4o-mini": "openai/gpt-4o-mini",
}


def _resolve_provider(model_name: str) -> str:
    """Return the provider to use: 'openrouter', 'vertex', 'anthropic', or 'openai'."""
    choice = os.environ.get("PBT_PROVIDER", "auto").lower()
    if choice == "openrouter" or (choice == "auto" and os.environ.get("OPENROUTER_API_KEY")):
        return "openrouter"
    if model_name.lower().startswith("claude"):
        return "vertex" if os.environ.get("ANTHROPIC_VERTEX_PROJECT") else "anthropic"
    return "openai"


def _openrouter_model_id(model_name: str) -> str:
    """Map a bare model name to its OpenRouter slug (pass-through if already namespaced)."""
    if "/" in model_name:
        return model_name
    if model_name in OPENROUTER_MODEL_IDS:
        return OPENROUTER_MODEL_IDS[model_name]
    if model_name.lower().startswith("claude"):
        return f"anthropic/{model_name}"
    if re.match(r"^(gpt|o[1-9])", model_name.lower()):
        return f"openai/{model_name}"
    return model_name


class Model:
    def __init__(self, model_name="gpt-4o-mini", temperature=0):
        self.model_name = model_name
        self.temperature = temperature
        self.provider = _resolve_provider(model_name)

        if self.provider == "openrouter":
            self.client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
            print(f"[INFO] Using OpenRouter for {model_name} -> {_openrouter_model_id(model_name)}")
        elif self.provider == "vertex":
            self.client = AnthropicVertex(
                project_id=os.environ["ANTHROPIC_VERTEX_PROJECT"],
                region=os.environ.get("ANTHROPIC_VERTEX_REGION", "us-east5"),
            )
            print(f"[INFO] Using Vertex AI for Claude (project={os.environ['ANTHROPIC_VERTEX_PROJECT']})")
        elif self.provider == "anthropic":
            self.client = anthropic.Anthropic()
            print("[INFO] Using direct Anthropic API for Claude")
        else:
            self.client = OpenAI()
            print("[INFO] Using direct OpenAI Responses API")

    # -- provider-specific request paths ------------------------------------

    def _claude_generate(self, prompt, max_tokens=2048, temperature=None):
        """Anthropic Messages API -> (text, usage)."""
        kwargs = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        message = self.client.messages.create(**kwargs)
        usage = {
            "input_tokens":  message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        }
        return message.content[0].text, usage

    def _responses_generate(self, prompt, max_tokens=2048, temperature=None):
        """OpenAI Responses API -> (text, usage)."""
        request = {"model": self.model_name, "input": prompt, "max_output_tokens": max_tokens}
        if temperature is not None and "gpt-5" not in self.model_name.lower():
            request["temperature"] = temperature
        response = self.client.responses.create(**request)
        usage = {
            "input_tokens":  response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return response.output_text, usage

    def _chat_generate(self, prompt, max_tokens=2048, temperature=None):
        """OpenAI-compatible Chat Completions API (OpenRouter) -> (text, usage)."""
        kwargs = {
            "model": _openrouter_model_id(self.model_name),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if temperature is not None and "gpt-5" not in self.model_name.lower():
            kwargs["temperature"] = temperature
        response = self.client.chat.completions.create(**kwargs)
        usage = {
            "input_tokens":  response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
        return response.choices[0].message.content, usage

    def complete(self, prompt, max_tokens=2048, temperature=None):
        """Single completion for the active provider -> (text, usage)."""
        temperature = temperature if temperature is not None else self.temperature
        if self.provider == "openrouter":
            return self._chat_generate(prompt, max_tokens, temperature)
        if self.provider in ("anthropic", "vertex"):
            return self._claude_generate(prompt, max_tokens, temperature)
        return self._responses_generate(prompt, max_tokens, temperature)

    def generate(self, task_description, n=1, max_tokens=2048, temperature=None):
        """Generate n candidate Python programs for a task -> (programs, summed usage)."""
        # APPS tasks read stdin and write stdout; say so explicitly.
        io_instruction = (
            "Write a complete Python program that reads from stdin and writes to stdout.\n"
            if any(kw in task_description for kw in ["-----Input-----", "-----Output-----", "Input\n", "Output\n", "stdin", "stdout"])
            else ""
        )
        prompt = (
            task_description
            + io_instruction
            + "Please provide Python code wrapped in triple backticks like:\n"
            "```python\n# your code here\n```\n\n"
        )

        programs = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        for _ in range(n):
            code, call_usage = self.complete(prompt, max_tokens, temperature)
            programs.append(code)
            usage["input_tokens"] += call_usage["input_tokens"]
            usage["output_tokens"] += call_usage["output_tokens"]
        return programs, usage
