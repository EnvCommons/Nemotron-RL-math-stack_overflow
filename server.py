"""Minimal server wrapper for Nemotron-RL-Math-Stack-Overflow environment."""

from openreward.environments import Server

from nemotron_rl_math_stack_overflow import NemotronRLMathStackOverflow

if __name__ == "__main__":
    server = Server([NemotronRLMathStackOverflow])
    server.run()
