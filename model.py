import os
import re
import anthropic
from openai import OpenAI

# Provider selection
# -------------------
# By default the wrapper talks to each vendor directly: Anthropic SDK for
# Claude, the OpenAI Responses API for everything else (Vertex for Claude when
# ANTHROPIC_VERTEX_PROJECT is set).
#
# Set OPENROUTER_API_KEY to route *every* model through OpenRouter instead — one
# OpenAI-compatible endpoint with instant cost reporting. Force the choice with
# PBT_PROVIDER=openrouter|direct (default: auto-detect from the key).
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter namespaces model IDs by vendor. Map the bare names this project
# uses to their OpenRouter slugs; any name already containing "/" is passed
# through unchanged, so callers can always supply an exact slug.
OPENROUTER_MODEL_IDS = {
    "claude-sonnet-4-5": "anthropic/claude-sonnet-4.5",
    "gpt-5.1": "openai/gpt-5.1",
    "gpt-4": "openai/gpt-4",
    "gpt-4o-mini": "openai/gpt-4o-mini",
}


def _zero_usage():
    return {"input_tokens": 0, "output_tokens": 0}


def _add_usage(a, b):
    return {
        "input_tokens":  a["input_tokens"]  + b["input_tokens"],
        "output_tokens": a["output_tokens"] + b["output_tokens"],
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
            from anthropic import AnthropicVertex
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

    def _is_claude(self):
        return self.model_name.lower().startswith("claude")

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
        request = {"model": self.model_name, "input": prompt}
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

    # -- output cleanup -----------------------------------------------------

    def _sanitize_java_output(self, code: str) -> str:
        # Remove triple backticks and language tags
        code = re.sub(r"```(?:java)?", "", code, flags=re.IGNORECASE).strip()
        # Remove a lone leading "java" token
        code = re.sub(r"^\s*java\s*\n", "", code, flags=re.IGNORECASE)
        # Drop anything before the first import or class declaration
        match = re.search(r"(import\s+|class\s+)", code)
        if match:
            code = code[match.start():]
        return code.strip()

    # -- public generation helpers ------------------------------------------

    def generate(self, task_description, n=1, temperature=None, language="python"):
        """
        Generate n initial programs (seeds) given a natural language description.
        Returns (responses, usage) where usage sums tokens across all n calls.
        """
        responses = []
        total_usage = _zero_usage()

        for _ in range(n):
            if language == "java":
                prompt = (
                    task_description
                    + "\n\n"
                    + "Write a complete Java class named `Solution` that implements the method above.\n"
                    + "Requirements:\n"
                    + "• Include `import java.util.*;` and `import java.lang.*;` at the top.\n"
                    + "• The class must be named exactly `Solution`.\n"
                    + "• Do NOT include a `main` method or any other class.\n"
                    + "\n"
                    + "Provide the code in a single ```java block:\n"
                    + "```java\n"
                    + "import java.util.*;\n"
                    + "import java.lang.*;\n"
                    + "\n"
                    + "class Solution {\n"
                    + "    // your method here\n"
                    + "}\n"
                    + "```\n\n"
                )
            elif language == "python":
                # APPS problems include stdin/stdout I/O — add explicit instruction
                # so all models (especially GPT-5.1) generate correct I/O handling.
                io_instruction = (
                    "Write a complete Python program that reads from stdin and writes to stdout.\n"
                    if any(kw in task_description for kw in ["-----Input-----", "-----Output-----", "Input\n", "Output\n", "stdin", "stdout"])
                    else ""
                )
                prompt = (
                    task_description +
                    io_instruction +
                    "Please provide Python code wrapped in triple backticks like:\n"
                    "```python\n"
                    "# your code here\n"
                    "```\n\n"
                )
            else:
                lang_map = {
                    "cpp": ("C++",          "cpp",        "// your code here"),
                    "go":  ("Go",           "go",         "// your code here"),
                    "js":  ("JavaScript",   "javascript", "// your code here"),
                }
                lang_name, lang_tag, lang_comment = lang_map.get(language, ("Python", "python", "# your code here"))
                prompt = (
                    task_description +
                    f"Please provide {lang_name} code wrapped in triple backticks like:\n"
                    f"```{lang_tag}\n"
                    f"{lang_comment}\n"
                    "```\n\n"
                )

            code, usage = self.complete(prompt, temperature=temperature)

            if language == "java":
                code = self._sanitize_java_output(code)

            responses.append(code)
            total_usage = _add_usage(total_usage, usage)

        return responses, total_usage

    def generate_feedback(self, task_description, program_or_context, temperature=None):
        """
        Ask the model to critique a failed attempt (program only or with history context).
        Returns (feedback_text, usage).
        """
        has_history = "Summary of previous attempts:" in program_or_context
        section_label = (
            "Context (previous attempts and current program)"
            if has_history else
            "Program to Critique"
        )

        prompt = (
            f"The following attempt did not pass all of its tests.\n\n"
            f"Please explain what might be wrong.\n\n"
            f"Task:\n{task_description}\n\n"
            f"{section_label}:\n{program_or_context}\n\n"
        )

        print(f"DEBUG [model.py]: Feedback prompt: {prompt}", flush=True)
        return self.complete(prompt, temperature=temperature)

    def generate_antiunified_history(self, trajectory, temperature=None):
        """
        Ask the model to perform *anti-unification* over all previous program attempts.
        Returns (abstraction_text, usage).
        """
        attempts = trajectory.get("refinement_attempts", [])
        if not attempts:
            return "", _zero_usage()

        program_snippets = [
            f"### Attempt {r['attempt']} (pass rate: {r['pass_fraction']*100:.1f}%)\n{r['program']}\n"
            for r in attempts
        ]
        programs_text = "\n\n".join(program_snippets)

        prompt = (
            "Given several program variants that attempt to solve the same problem, "
            "derive a single generalized form that captures their shared structure.\n\n"
            "Rules for generalization:\n"
            "- Preserve common syntax and control flow.\n"
            "- Replace differing expressions, constants, or statements with placeholders "
            "such as <EXPR>, <VAR>, or <COND>.\n"
            "- Do not paraphrase or summarize the code — output a concrete unified program skeleton.\n\n"
            "Here are the previous attempts:\n\n"
            f"{programs_text}\n\n"
            "Produce the anti-unified abstraction below:\n"
        )
        return self.complete(prompt, temperature=temperature)

    def refine(self, task_description, program, feedback=None, temperature=None):
        """
        Ask the model to revise its program, either directly or using critique feedback.
        Returns (revised_code, usage).
        """
        if feedback:
            prompt = (
                f"Task:\n{task_description}\n\n"
                f"Current Program:\n{program}\n\n"
                f"Feedback:\n{feedback}\n\n"
                f"Revise the program to address the feedback. "
                f"Only return the corrected code."
            )
        else:
            prompt = (
                f"Task:\n{task_description}\n\n"
                f"Current Program:\n{program}\n\n"
                f"Revise and improve the program to make it pass all tests."
            )
        return self.complete(prompt, temperature=temperature)
