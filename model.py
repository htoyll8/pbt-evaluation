import os
import re
import textwrap
import anthropic
from openai import OpenAI

# Set ANTHROPIC_VERTEX_PROJECT to use Vertex AI instead of direct Anthropic API.
# e.g. export ANTHROPIC_VERTEX_PROJECT=your-gcp-project
_VERTEX_PROJECT = os.environ.get("ANTHROPIC_VERTEX_PROJECT")
_VERTEX_REGION  = os.environ.get("ANTHROPIC_VERTEX_REGION", "us-east5")


def _zero_usage():
    return {"input_tokens": 0, "output_tokens": 0}


def _add_usage(a, b):
    return {
        "input_tokens":  a["input_tokens"]  + b["input_tokens"],
        "output_tokens": a["output_tokens"] + b["output_tokens"],
    }


class Model:
    def __init__(self,
                 model_name="gpt-4o-mini",
                 temperature=0):
        self.model_name = model_name
        self.temperature = temperature

        if self._is_claude():
            vertex_project = os.environ.get("ANTHROPIC_VERTEX_PROJECT")
            vertex_region = os.environ.get("ANTHROPIC_VERTEX_REGION", "us-east5")
            if vertex_project:
                from anthropic import AnthropicVertex
                self.client = AnthropicVertex(
                    project_id=vertex_project,
                    region=vertex_region,
                )
                print(f"[INFO] Using Vertex AI for Claude (project={vertex_project}, region={vertex_region})")
            else:
                self.client = anthropic.Anthropic()
                print("[INFO] Using direct Anthropic API for Claude")
        else:
            self.client = OpenAI()
            print(hasattr(self.client, "responses"))

    def _is_claude(self):
        return self.model_name.lower().startswith("claude")

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

    def _claude_generate(self, prompt, max_tokens=2048, temperature=None):
        """Call Claude and return (text, usage_dict)."""
        kwargs = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        if temperature is not None:
            kwargs["temperature"] = temperature

        message = self.client.messages.create(**kwargs)
        usage = {
            "input_tokens":  message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        }
        return message.content[0].text, usage

    def _openai_generate(self, request):
        """Call OpenAI Responses API and return (text, usage_dict)."""
        response = self.client.responses.create(**request)
        usage = {
            "input_tokens":  response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return response.output_text, usage

    def generate(
            self,
            task_description,
            n=1,
            temperature=None,
            language="python"):
        """
        Generate n initial programs (seeds) given a natural language description.
        Returns (responses, usage) where usage sums tokens across all n calls.
        """
        temperature = temperature if temperature is not None else getattr(self, "temperature", None)

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

            if self._is_claude():
                code, usage = self._claude_generate(prompt, temperature=temperature)
            else:
                request = {"model": self.model_name, "input": prompt}
                if temperature is not None and "gpt-5" not in self.model_name.lower():
                    request["temperature"] = temperature
                code, usage = self._openai_generate(request)

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
        temperature = temperature if temperature is not None else getattr(self, "temperature", None)

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

        if self._is_claude():
            return self._claude_generate(prompt, temperature=temperature)

        request = {
            "model": self.model_name,
            "input": [{"role": "user", "content": prompt}],
        }
        if temperature is not None and "gpt-5" not in self.model_name.lower():
            request["temperature"] = temperature
        return self._openai_generate(request)

    def generate_antiunified_history(self, trajectory, temperature=None):
        """
        Ask the model to perform *anti-unification* over all previous program attempts.
        Returns (abstraction_text, usage).
        """
        temperature = temperature if temperature is not None else getattr(self, "temperature", None)

        attempts = trajectory.get("refinement_attempts", [])
        if not attempts:
            return "", _zero_usage()

        program_snippets = []
        for r in attempts:
            program_snippets.append(
                f"### Attempt {r['attempt']} (pass rate: {r['pass_fraction']*100:.1f}%)\n"
                f"{r['program']}\n"
            )

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

        request = {
            "model": self.model_name,
            "input": [{"role": "user", "content": prompt}],
        }
        if temperature is not None and "gpt-5" not in self.model_name.lower():
            request["temperature"] = temperature
        return self._openai_generate(request)

    def refine(self,
               task_description,
               program,
               feedback=None,
               temperature=None):
        """
        Ask the model to revise its program, either directly or using critique feedback.
        Returns (revised_code, usage).
        """
        temperature = temperature if temperature is not None else getattr(self, "temperature", None)

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

        if self._is_claude():
            return self._claude_generate(prompt, temperature=temperature)

        request = {
            "model": self.model_name,
            "input": [{"role": "user", "content": prompt}],
        }
        if temperature is not None and "gpt-5" not in self.model_name.lower():
            request["temperature"] = temperature
        return self._openai_generate(request)
