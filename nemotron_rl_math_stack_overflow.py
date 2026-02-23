"""Nemotron-RL-Math-Stack-Overflow Environment for math problem solving.

This environment evaluates agents on 436,307+ math problems from Stack Overflow
using LLM-based grading for flexible mathematical equivalence checking.
"""

from __future__ import annotations

import re
import pyarrow.parquet as pq
import openai
from pydantic import BaseModel, Field

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool

from constants import DATA_PATH
from prompts import MATH_GRADER_TEMPLATE


# Load consolidated dataset at module import time (like mmlu_prox pattern)
PARQUET_FILE = DATA_PATH / "nemotron_math_consolidated.parquet"

if PARQUET_FILE.exists():
    TASKS_DF = pq.read_table(str(PARQUET_FILE)).to_pandas()
    print(f"Loaded {len(TASKS_DF):,} tasks from {PARQUET_FILE}")
else:
    raise FileNotFoundError(
        f"Data file not found: {PARQUET_FILE}\n\n"
        f"For local dev: Run `python download_data.py` to generate the dataset\n"
        f"For production: Upload nemotron_math_consolidated.parquet to /orwd_data/"
    )


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
        """Return available splits.

        Returns:
            ["train", "validation"]
        """
        return ["train", "validation"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """Return task specifications for the given split.

        Args:
            split: Either "train" or "validation"

        Returns:
            List of task specifications as dictionaries

        Raises:
            ValueError: If split is not "train" or "validation"
        """
        if split not in ["train", "validation"]:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train' or 'validation'"
            )

        # Filter dataframe by split
        filtered = TASKS_DF[TASKS_DF["split"] == split]

        # Return task specs as dictionaries
        return [
            {
                "task_id": row["task_id"],
                "split": row["split"],
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "row_idx": int(row["row_idx"])
            }
            for _, row in filtered.iterrows()
        ]

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

        self.client = openai.AsyncClient(api_key=api_key)

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
        """
        # Extract reasoning
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL | re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Extract answer
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        answer = answer_match.group(1).strip().upper() if answer_match else ""

        # Validate answer
        if answer not in ["CORRECT", "INCORRECT"]:
            print(f"Invalid grade in response: {answer}")
            return reasoning or "Failed to parse grading response", "INCORRECT"

        return reasoning, answer

    async def _grade_answer(self, student_answer: str) -> tuple[str, str]:
        """Grade student answer using LLM grader.

        Args:
            student_answer: The answer provided by the agent

        Returns:
            tuple[str, str]: (reasoning, grade) where grade is "CORRECT" or "INCORRECT"

        Note:
            Uses gpt-5-mini with NO temperature parameter (per CLAUDE.md)
            Includes retry loop for parsing failures
        """
        # Format grading prompt
        grader_prompt = MATH_GRADER_TEMPLATE.format(
            reference_answer=self.validated.expected_answer,
            student_answer=student_answer
        )

        # Retry loop for parsing failures (like medrb)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use gpt-5-mini with NO temperature parameter (per CLAUDE.md)
                res = await self.client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{"role": "user", "content": grader_prompt}]
                )

                grading_response = res.choices[0].message.content or ""

                # Parse response with reasoning and answer tags
                reasoning, grade = self._parse_grading_response(grading_response)

                if grade in ["CORRECT", "INCORRECT"]:
                    return reasoning, grade

            except Exception as e:
                print(f"Grading error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return "Grading failed after retries", "INCORRECT"

        return "Grading failed", "INCORRECT"

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
