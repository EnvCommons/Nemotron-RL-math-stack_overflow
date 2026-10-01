"""Offline tests for the answer/grading path, with a scripted grader client.

Run: uv run --no-project --with-requirements requirements.txt --with pytest python -m pytest tests.py -q
"""
import asyncio
from types import SimpleNamespace

import pytest

import nemotron_rl_math_stack_overflow as mod
from nemotron_rl_math_stack_overflow import AnswerInput, NemotronRLMathStackOverflow

TASK = {"task_id": "t0", "question": "What is 1+1?", "expected_answer": "2", "split": "train", "row_idx": 0}


class ScriptedClient:
    """Stands in for openai.AsyncClient; each scripted item is a reply or an exception."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def _env(replies: list) -> tuple[NemotronRLMathStackOverflow, ScriptedClient]:
    env = NemotronRLMathStackOverflow(task_spec=TASK, secrets={"openai_api_key": "test"})
    env.client = ScriptedClient(replies)
    return env, env.client


def _answer(env, answer: str = "2"):
    return asyncio.run(env.answer(AnswerInput(answer=answer)))


def test_client_timeout_leaves_room_for_a_capped_reply():
    env = NemotronRLMathStackOverflow(task_spec=TASK, secrets={"openai_api_key": "test"})
    assert env.client.timeout == mod.GRADER_TIMEOUT_S
    assert env.client.max_retries == 0


def test_grader_request_is_capped():
    env, client = _env(["<reasoning>ok</reasoning><answer>CORRECT</answer>"])
    out = _answer(env)
    assert out.reward == 1.0 and out.finished is True
    assert client.requests[0]["max_completion_tokens"] == mod.GRADER_MAX_TOKENS


def test_truncated_and_timed_out_replies_are_retried():
    env, client = _env([TimeoutError("Request timed out."), "", "<answer>INCORRECT</answer>"])
    out = _answer(env)
    assert out.reward == 0.0 and out.finished is True and len(client.requests) == 3


def test_grader_failure_raises_without_reference_and_keeps_attempt():
    env, client = _env(["", "", ""])
    with pytest.raises(RuntimeError) as exc:
        _answer(env)
    assert "Grader failed" in str(exc.value)
    assert env.submitted == 0
