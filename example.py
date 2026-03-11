"""Example agent trajectory for the Nemotron-RL-Math-Stack-Overflow environment."""

import argparse
import json
import textwrap
import os

from openai import OpenAI
from openreward import OpenReward

or_client = OpenReward()
oai_client = OpenAI()
MODEL_NAME = "gpt-5.4"

BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_separator(char="─", width=80):
    print(f"{DIM}{char * width}{RESET}")


def print_model_output(response):
    print()
    print_separator()
    print(f"{BOLD}{BLUE}🤖 Model Response{RESET}  {DIM}(id: {response.id}){RESET}")
    print_separator()
    for item in response.output:
        if item.type == "reasoning":
            summary = getattr(item, "summary", None)
            if summary:
                for s in summary:
                    print(f"\n  {DIM}💭 {s.text}{RESET}")
        elif item.type == "message":
            for part in item.content:
                if hasattr(part, "text"):
                    print(f"\n{part.text}")
        elif item.type == "function_call":
            try:
                args = json.loads(item.arguments)
                args_str = json.dumps(args, indent=2)
            except (json.JSONDecodeError, TypeError):
                args_str = item.arguments
            print(f"\n  {YELLOW}⚡ Tool Call:{RESET} {BOLD}{item.name}{RESET}")
            print(f"  {DIM}call_id: {item.call_id}{RESET}")
            for line in args_str.splitlines():
                print(f"  {DIM}│{RESET} {line}")
    print()


def print_tool_result(tool_name, tool_result):
    print_separator("·")
    status = f"{GREEN}✓ done" if tool_result.finished else f"{CYAN}… continuing"
    reward_str = f"  reward={tool_result.reward}" if tool_result.reward is not None else ""
    print(f"  {GREEN}↩ Result:{RESET} {BOLD}{tool_name}{RESET}  {DIM}[{status}{RESET}{DIM}{reward_str}]{RESET}")
    for block in tool_result.blocks:
        if block.type == "text":
            text = block.text
            if len(text) > 2000:
                text = text[:2000] + f"\n{DIM}... ({len(block.text)} chars total){RESET}"
            for line in text.splitlines():
                print(f"  {DIM}│{RESET} {line}")
        elif block.type == "image":
            print(f"  {DIM}│ [image: {block.mimeType}]{RESET}")
    print()


parser = argparse.ArgumentParser(description="Run an example agent trajectory")
parser.add_argument("--index", type=int, default=0, help="Task index to run (default: 0)")
args = parser.parse_args()

ENV_NAME = "GeneralReasoning/Nemotron-RL-math-stack_overflow"
TASK_INDEX = args.index

environment = or_client.environments.get(name=ENV_NAME)
tools = environment.list_tools(format="openai")

with environment.session(split="train", index=TASK_INDEX, secrets={"openai_api_key": os.environ.get("OPENAI_API_KEY", "")}) as session:

    rollout = or_client.rollout.create(
        run_name="nemotron-math-so-train-quickstart",
        rollout_name="example_task",
        environment=ENV_NAME,
        split="train",
    )

    prompt = session.get_prompt()
    print(f"\n{BOLD}📋 Prompt:{RESET}")
    print_separator()
    for block in prompt:
        if block.type == "text":
            print(textwrap.shorten(block.text, width=200, placeholder="..."))
    print_separator()

    input_list = [{"role": "user", "content": prompt[0].text}]
    rollout.log_openai_response(input_list[0])
    finished = False
    step = 0

    while not finished:
        step += 1
        print(f"\n{BOLD}━━━ Step {step} ━━━{RESET}")
        response = oai_client.responses.create(
            model=MODEL_NAME,
            tools=tools,
            input=input_list,
        )
        print_model_output(response)
        rollout.log_openai_response(response)

        input_list += response.output

        for item in response.output:
            if item.type == "function_call":
                tool_result = session.call_tool(item.name, json.loads(str(item.arguments)))
                print_tool_result(item.name, tool_result)

                reward = tool_result.reward
                finished = tool_result.finished

                tool_result_item = {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps({
                        "result": tool_result.blocks[0].text
                    }),
                }
                input_list.append(tool_result_item)
                rollout.log_openai_response(tool_result_item, reward=reward, is_finished=finished)

                if tool_result.finished:
                    finished = True
                    break

    print_separator("═")
    print(f"{BOLD}{GREEN}✅ Finished!{RESET}")
    print_separator("═")
    or_client.rollout.close()
