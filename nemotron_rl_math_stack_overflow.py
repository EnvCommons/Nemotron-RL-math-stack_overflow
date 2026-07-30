"""Nemotron-RL-Math-Stack-Overflow Environment for math problem solving.

This environment evaluates agents on 436,307+ math problems from Stack Overflow
using LLM-based grading for flexible mathematical equivalence checking.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import openai
from pydantic import BaseModel, Field

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool, Split

from constants import DATA_PATH
from prompts import MATH_GRADER_TEMPLATE


# ---------------------------------------------------------------------------
# Dataset loading from parquet (lazy, O(1) per-row access)
# ---------------------------------------------------------------------------

PARQUET_FILE = DATA_PATH / "nemotron_math_consolidated.parquet"
INDEX_PATH = DATA_PATH / "task_index.json"


class _TaskIndex:
    """Precomputed task index for O(1) lookups by split and index.

    Built once by build_index.py, loaded lazily on first access.
    Individual rows are fetched from parquet by targeting the exact row group.
    """

    def __init__(self, index_path: Path, parquet_path: Path):
        raw = json.loads(index_path.read_text())
        self._splits: dict[str, list[int]] = raw["splits"]
        self._parquet_path = parquet_path

    def num_tasks(self, split: str) -> int:
        if split not in self._splits:
            raise ValueError(f"Unknown split: {split!r}")
        return len(self._splits[split])

    def get_row(self, split: str, index: int) -> dict:
        """Read a single row by split and index from the correct row group."""
        if split not in self._splits:
            raise ValueError(f"Unknown split: {split!r}")
        indices = self._splits[split]
        if index < 0 or index >= len(indices):
            raise IndexError(
                f"index {index} out of range (0..{len(indices) - 1})"
            )
        raw_idx = indices[index]
        return self._read_row(self._parquet_path, raw_idx)

    @staticmethod
    def _read_row(path: Path, local_idx: int) -> dict:
        """Read one row from a parquet file via its row group."""
        pf = pq.ParquetFile(path)
        offset = 0
        for rg_idx in range(pf.metadata.num_row_groups):
            rg_rows = pf.metadata.row_group(rg_idx).num_rows
            if local_idx < offset + rg_rows:
                rg_table = pf.read_row_group(rg_idx)
                row = rg_table.slice(local_idx - offset, 1).to_pydict()
                return {k: v[0] for k, v in row.items()}
            offset += rg_rows
        raise IndexError(f"row index {local_idx} not found in {path}")


_dataset: _TaskIndex | None = None


def _get_dataset() -> _TaskIndex:
    """Lazy singleton for the task index."""
    global _dataset
    if _dataset is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Task index not found: {INDEX_PATH}\n\n"
                f"Run `python build_index.py` to generate it from the parquet file."
            )
        if not PARQUET_FILE.exists():
            raise FileNotFoundError(
                f"Data file not found: {PARQUET_FILE}\n\n"
                f"For local dev: Run `python download_data.py` to generate the dataset\n"
                f"For production: Upload nemotron_math_consolidated.parquet to /orwd_data/"
            )
        _dataset = _TaskIndex(INDEX_PATH, PARQUET_FILE)
    return _dataset


class TaskSpec(BaseModel):
    """Task specification for Nemotron math tasks."""
    task_id: str
    question: str
    expected_answer: str
    split: str
    row_idx: int  # For O(1) DataFrame lookup


class AnswerInput(BaseModel, extra="forbid"):
    """Input schema for answer tool."""
    answer: str = Field(
        ...,
        description="Your final answer to the math problem. Can be a number, expression, fraction, etc."
    )


class NemotronRLMathStackOverflow(Environment):
    """Nemotron-RL-Math-Stack-Overflow environment for math problem solving.

    This environment presents math problems from Stack Overflow and evaluates
    answers using LLM-based grading (gpt-5-mini) for flexible mathematical
    equivalence checking. The grader handles different formats: fractions,
    decimals, expressions, boxed notation, etc.

    Dataset: nvidia/Nemotron-RL-math-stack_overflow
    Total examples: 436,307+
    Splits: train, validation
    Grading: LLM-based via gpt-5-mini
    Reward: Binary (1.0 for correct, 0.0 for incorrect)
    """

    @classmethod
    def list_splits(cls) -> list[str]:
        return [Split(name="train", type="train"), Split(name="validation", type="validation")]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        raise NotImplementedError(
            "Dataset has 436K+ tasks — use num_tasks/get_task instead"
        )

    @classmethod
    async def num_tasks(cls, split: str) -> int:
        return _get_dataset().num_tasks(split)

    @classmethod
    async def get_task(cls, split: str, index: int) -> JSONObject:
        row = _get_dataset().get_row(split, index)
        return {
            "task_id": row["task_id"],
            "split": row["split"],
            "question": row["question"],
            "expected_answer": row["expected_answer"],
            "row_idx": int(row["row_idx"]),
        }

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        """Initialize the environment for a specific task.

        Args:
            task_spec: Task specification with task_id, question, expected_answer, etc.
            secrets: Dictionary containing API keys (must include "openai_api_key")

        Raises:
            ValueError: If openai_api_key is not provided in secrets
        """
        super().__init__(task_spec)
        self.validated = TaskSpec.model_validate(task_spec)

        # CRITICAL: Use secrets parameter for API key (per CLAUDE.md)
        api_key = secrets.get("openai_api_key")
        if not api_key:
            raise ValueError("OpenAI API key must be provided via secrets parameter")

        # Bounded per-call timeout so a degraded endpoint fails fast; retries are
        # handled by the loop in _grade_answer, not by the SDK.
        self.client = openai.AsyncClient(api_key=api_key, timeout=120.0, max_retries=0)

    async def get_prompt(self) -> list[TextBlock]:
        """Generate the prompt for this math problem.

        Returns:
            List containing formatted prompt with the math question and instructions
        """
        prompt_text = f"""# Math Problem

{self.validated.question}

## Instructions:
- Solve the math problem step by step
- Provide your final answer using the `answer` tool
- Your answer will be evaluated for mathematical equivalence (different formats like fractions, decimals, or expressions are acceptable)
- You have one attempt to submit your answer

## Examples of Equivalent Formats:
- `5/9` = `0.555...` = `\\boxed{{5/9}}`
- `sqrt(2)` = `1.414...`
- `42` = `\\boxed{{42}}`
"""

        return [TextBlock(text=prompt_text)]

    def _parse_grading_response(self, response: str) -> tuple[str, str]:
        """Parse grading response with reasoning and answer tags.

        Args:
            response: Raw response from LLM with <reasoning> and <answer> tags

        Returns:
            tuple[str, str]: (reasoning, grade) where grade is "CORRECT" or "INCORRECT"

        Raises:
            ValueError: If the response carries no valid CORRECT/INCORRECT grade, so
                the caller can retry rather than treat it as a verdict.
        """
        # Extract reasoning
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL | re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Extract answer
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        answer = answer_match.group(1).strip().upper() if answer_match else ""

        # Validate answer
        if answer not in ["CORRECT", "INCORRECT"]:
            raise ValueError(f"No valid <answer> grade in grading response: {response!r:.500}")

        return reasoning, answer

    async def _grade_answer(self, student_answer: str) -> tuple[str, str]:
        """Grade student answer using LLM grader.

        Args:
            student_answer: The answer provided by the agent

        Returns:
            tuple[str, str]: (reasoning, grade) where grade is "CORRECT" or "INCORRECT"

        Raises:
            RuntimeError: If grading fails on every attempt. A grader outage is not a
                verdict on the answer, so it propagates and lets the platform retry
                the tool call instead of scoring the rollout 0.0.

        Note:
            Uses gpt-5-mini with NO temperature parameter (per CLAUDE.md)
            Retries both API errors and unparseable grader responses.
        """
        # Format grading prompt
        grader_prompt = MATH_GRADER_TEMPLATE.format(
            reference_answer=self.validated.expected_answer,
            student_answer=student_answer
        )

        max_retries = 3
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                # Use gpt-5-mini with NO temperature parameter (per CLAUDE.md)
                res = await self.client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{"role": "user", "content": grader_prompt}]
                )

                grading_response = res.choices[0].message.content or ""

                # Parse response with reasoning and answer tags
                return self._parse_grading_response(grading_response)

            except Exception as e:
                last_error = e
                print(f"Grading error (attempt {attempt + 1}/{max_retries}): {e}")

        raise RuntimeError(
            f"Grader failed after {max_retries} attempts: {last_error}"
        ) from last_error

    @tool
    async def answer(self, params: AnswerInput) -> ToolOutput:
        """Submit your answer for the math problem.

        Your answer will be checked for mathematical equivalence with the expected
        answer using LLM-based grading (gpt-5-mini). This handles various formats:
        - Fractions: 5/9, 1/2
        - Decimals: 0.555..., 1.414
        - Expressions: sqrt(2), 2*pi
        - Boxed notation: \\boxed{42}

        Args:
            params: AnswerInput containing your answer

        Returns:
            ToolOutput with correctness feedback, metadata, and binary reward
            (1.0 for correct, 0.0 for incorrect)
        """
        # Grade answer using LLM
        reasoning, grade = await self._grade_answer(params.answer)
        reward = 1.0 if grade == "CORRECT" else 0.0

        # Format result message
        result_text = f"""# Grading Results

**Grade**: {grade}
**Reward**: {reward}

**Your Answer**: {params.answer}
**Expected Answer**: {self.validated.expected_answer}

**Reasoning**: {reasoning}
"""

        return ToolOutput(
            blocks=[TextBlock(text=result_text)],
            metadata={
                "task_id": self.validated.task_id,
                "split": self.validated.split,
                "submitted": params.answer,
                "expected": self.validated.expected_answer,
                "grade": grade,
                "reasoning": reasoning,
                "correct": grade == "CORRECT"
            },
            reward=reward,
            finished=True
        )
