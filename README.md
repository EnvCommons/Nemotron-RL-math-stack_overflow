# Nemotron-RL-Math-Stack-Overflow

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/nemotron_rl_math_stack_overflow) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/nvidia/Nemotron-RL-math-stack_overflow)

## Description

Nemotron-RL-Math-Stack-Overflow is an environment for evaluating agents on mathematical problem-solving using problems extracted from Stack Overflow. Part of NVIDIA's NeMo Gym framework for reinforcement learning from verifiable reward (RLVR), it provides a large-scale collection of math problems with LLM-based grading for flexible mathematical equivalence checking.

## Capabilities

- Solving mathematical problems across a wide range of topics and difficulty levels
- Handling various answer formats: fractions, decimals, expressions, boxed notation
- Step-by-step mathematical reasoning

## Compute Requirements

This is a single-turn environment with no sandbox. No special compute resources are required.

## License

[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Tasks

There are two splits in this environment:

- **Train**: 436,307 tasks
- **Validation**: 30 tasks

Each task presents a math problem sourced from Stack Overflow and requires the agent to provide a final answer.

## Reward Structure

This is a single-turn environment with binary reward:

- **1.0** — Correct answer (mathematically equivalent to the reference)
- **0.0** — Incorrect answer

Grading is performed by gpt-5-mini, which evaluates mathematical equivalence across different representations (e.g., `5/9` = `0.555...` = `\boxed{5/9}`). Includes a retry loop (3 attempts) for robust evaluation.

## Data

The dataset contains math problems and solutions sourced from Stack Overflow forums, extracted using methods similar to the OpenMathReasoning dataset. Only problems with extracted answers are included. Data is stored as a consolidated Parquet file on the OpenReward platform.

Source: [nvidia/Nemotron-RL-math-stack_overflow](https://huggingface.co/datasets/nvidia/Nemotron-RL-math-stack_overflow)

## Tools

| Tool | Description |
|------|-------------|
| `answer` | Submit your final answer to the math problem. Accepts numbers, expressions, fractions, boxed notation, etc. Returns binary reward with grading explanation. |

## Time Horizon

Nemotron-RL-Math-Stack-Overflow is a single-turn environment. The agent receives a math problem and submits one answer for a total of one tool call.

## Environment Difficulty

The dataset covers a wide range of mathematical difficulty, from basic arithmetic to advanced topics found on Stack Overflow.

## Other Environment Requirements

- **OpenAI API key**: Required for LLM-based grading of mathematical equivalence. Pass via `secrets={"openai_api_key": "..."}`.

## Safety

This environment evaluates mathematical reasoning and does not present direct safety risks. Agents interact only with math problems and a grading system.

## Citations

```bibtex
@dataset{nvidia_nemotron_rl_math,
  author    = {NVIDIA},
  title     = {Nemotron-RL-math-stack\_overflow},
  year      = {2025},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/nvidia/Nemotron-RL-math-stack_overflow}
}
```
