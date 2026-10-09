"""Offline tests for the answer/grading path, with a scripted grader client.

Run: uv run --no-project --with-requirements requirements.txt --with pytest python -m pytest tests.py -q
"""
import asyncio
import json
from types import SimpleNamespace

import httpx
import openai
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

    async def _create(self, *, model, messages, max_completion_tokens, reasoning_effort=None):
        # Keyword-only, like the SDK's create(); an unexpected argument is a TypeError.
        assert reasoning_effort in (None, "minimal", "low", "medium", "high")
        self.requests.append({"model": model, "messages": messages,
                              "max_completion_tokens": max_completion_tokens,
                              "reasoning_effort": reasoning_effort})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        # An empty reply is what the API returns when reasoning uses up the token cap.
        finish_reason = "stop" if reply else "length"
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=reply), finish_reason=finish_reason)])


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


def _timeout() -> Exception:
    return openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))


def test_grader_outage_raises_without_reference_and_keeps_attempt():
    env, client = _env([_timeout(), _timeout(), _timeout()])
    with pytest.raises(RuntimeError) as exc:
        _answer(env)
    assert "Grader failed" in str(exc.value)
    assert TASK["expected_answer"] not in str(exc.value)
    assert env.submitted == 0


def test_client_timeout_is_bounded():
    # Three attempts must fit well inside a tool call's budget.
    assert mod.GRADER_TIMEOUT_S * len(mod.GRADER_ATTEMPTS) <= 600


def test_verdictless_replies_are_not_graded_and_keep_attempt():
    env, client = _env(["", "", "", "<reasoning>ok</reasoning><answer>CORRECT</answer>"])
    out = _answer(env, "the agent's answer")
    assert out.finished is False and out.reward == 0.0
    assert out.metadata["graded"] is False
    text = out.blocks[0].text + str(out.metadata)
    assert "could not be graded" in text and "does not count" in text
    assert TASK["expected_answer"] not in out.blocks[0].text
    assert env.submitted == 0
    # The resubmission is graded normally.
    out = _answer(env)
    assert out.finished is True and out.reward == 1.0 and env.submitted == 1


def test_retry_after_a_verdictless_reply_uses_low_effort():
    env, client = _env(["", "<reasoning>ok</reasoning><answer>INCORRECT</answer>"])
    out = _answer(env)
    assert out.finished is True and out.reward == 0.0
    assert client.requests[0]["reasoning_effort"] is None
    assert client.requests[0]["max_completion_tokens"] == mod.GRADER_MAX_TOKENS
    assert client.requests[1]["reasoning_effort"] == "low"


def test_verdictless_reply_then_outage_is_not_graded():
    env, client = _env(["", _timeout(), _timeout()])
    out = _answer(env)
    assert out.finished is False and env.submitted == 0


POKER = ("What is the probability that a 5-card poker hand contains no pairs, no runs of 5 "
         "consecutive values, and not all 5 cards of the same suit?")


class _Rows:
    def __init__(self, row: dict) -> None:
        self.row = row

    def get_row(self, split: str, index: int) -> dict:
        return dict(self.row)


def _get_task(monkeypatch, row: dict) -> dict:
    monkeypatch.setattr(mod, "_get_dataset", lambda: _Rows(row))
    return asyncio.run(NemotronRLMathStackOverflow.get_task("train", 0))


def test_wrong_poker_reference_is_corrected(monkeypatch):
    from fractions import Fraction
    from math import comb

    # High-card hands: distinct ranks, not one of the 10 straights, not one of 4 flushes.
    assert Fraction((comb(13, 5) - 10) * (4**5 - 4), comb(52, 5)) == Fraction(1277, 2548)
    row = {"task_id": "train_21938", "split": "train", "question": POKER, "expected_answer": "0.507", "row_idx": 21938}
    assert _get_task(monkeypatch, row)["expected_answer"] == "1277/2548"


def test_correction_needs_the_matching_question(monkeypatch):
    row = {"task_id": "train_21938", "split": "train", "question": "Another question?", "expected_answer": "0.507", "row_idx": 21938}
    assert _get_task(monkeypatch, row)["expected_answer"] == "0.507"
    row = {"task_id": "train_1", "split": "train", "question": "What is 1+1?", "expected_answer": "2", "row_idx": 1}
    assert _get_task(monkeypatch, row)["expected_answer"] == "2"


COUNTEREXAMPLE = {
    "task_id": "t1",
    "question": "Provide a counterexample to the conjecture that if \\( a^2 > b^2 \\) for real numbers "
                "\\( a \\) and \\( b \\), then \\( a > b \\). Use the set {a, b}.",
    "expected_answer": "\\( a = -3 \\), \\( b = 2 \\)",
    "split": "train",
    "row_idx": 1,
}


def test_grader_sees_the_question_and_accepts_other_valid_answers():
    # Without the question the grader can only compare to the one reference
    # counterexample, so a different valid counterexample is graded by chance.
    env = NemotronRLMathStackOverflow(task_spec=COUNTEREXAMPLE, secrets={"openai_api_key": "test"})
    env.client = client = ScriptedClient(["<reasoning>ok</reasoning><answer>CORRECT</answer>"])
    out = _answer(env, "a = -1, b = 0")
    assert out.reward == 1.0
    prompt = client.requests[0]["messages"][0]["content"]
    assert COUNTEREXAMPLE["question"] in prompt
    assert prompt.index(COUNTEREXAMPLE["question"]) < prompt.index(COUNTEREXAMPLE["expected_answer"])
    assert "more than one correct answer" in prompt and "does not need to match the reference" in prompt
    assert "a = -1, b = 0" in prompt


COMBINATIONS = ("How many possible combinations are there if you have 95 possibilities and 63 slots to fill, "
                "where the choices can repeat and order matters?")
BEAM = ("Solve the partial differential equation \\( u_{tt} = u_{xxxx} \\) for \\( t > 0 \\) with the initial "
        "conditions \\( u(0,x) = p(x) \\) and \\( u_t(0,x) = 0 \\) using the Fourier Transform.")


def test_wrong_combinations_reference_is_corrected(monkeypatch):
    # The dataset's reference sums every length from 1 to 63 slots.
    assert (95**64 - 95) // 94 == sum(95**k for k in range(1, 64)) != 95**63
    row = {"task_id": "train_188208", "split": "train", "question": COMBINATIONS,
           "expected_answer": "\\(\\frac{95^{64}-95}{94}\\)", "row_idx": 188208}
    assert _get_task(monkeypatch, row)["expected_answer"] == "\\(95^{63}\\)"


def test_wrong_fourier_pde_reference_is_corrected(monkeypatch):
    sp = pytest.importorskip("sympy")
    t, x, k = sp.symbols("t x k", real=True)
    mode = lambda f: f(k**2 * t) * sp.exp(sp.I * k * x)
    pde = lambda u: sp.simplify(sp.diff(u, t, 2) - sp.diff(u, x, 4))
    assert pde(mode(sp.cosh)) == 0 and pde(mode(sp.cos)) != 0
    row = {"task_id": "train_80384", "split": "train", "question": BEAM,
           "expected_answer": "\\( u(t,x) = \\mathcal{F}^{-1}\\left[F(p(x))\\cos(k^2t)\\right] \\)", "row_idx": 80384}
    ref = _get_task(monkeypatch, row)["expected_answer"]
    assert "\\cosh(k^2t)" in ref and "\\cos(" not in ref


def test_excluded_task_is_dropped_and_later_tasks_shift(tmp_path):
    rows = list(range(377_715))
    index = tmp_path / "task_index.json"
    index.write_text(json.dumps({"splits": {"train": rows, "validation": [377_715, 377_716]}}))
    tasks = mod._TaskIndex(index, tmp_path / "unused.parquet")
    assert tasks.num_tasks("train") == len(rows) - 1
    assert tasks.num_tasks("validation") == 2
    train = tasks._splits["train"]
    assert 377_710 not in train
    assert train[377_709] == 377_709 and train[377_710] == 377_711
