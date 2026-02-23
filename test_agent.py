"""Agent integration testing script for Nemotron-RL-Math-Stack-Overflow environment.

This script tests the environment by running an agent on a few validation tasks.
"""

import json
import asyncio
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward


async def main():
    """Test Nemotron-RL-Math-Stack-Overflow environment with an agent."""

    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI()

    MODEL_NAME = "gpt-5.2"
    ENV_NAME = "YourOrg/nemotron_rl_math_stack_overflow"
    SPLIT = "validation"  # Start with validation (smaller)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return

    print(f"Connecting to environment: {ENV_NAME}")
    environment = or_client.environments.get(
        name=ENV_NAME,
        base_url="http://localhost:8080"
    )

    print(f"Listing tasks for split: {SPLIT}")
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks in {SPLIT} split")
    print(f"Testing first 3 tasks...\n")

    # Test first 3 tasks
    for task in tasks[:3]:
        print(f"\n{'=' * 70}")
        print(f"Testing task: {task.task_spec['task_id']}")
        print(f"Question: {task.task_spec['question'][:100]}...")
        print(f"{'=' * 70}\n")

        rollout = or_client.rollout.create(
            run_name="nemotron_math_test",
            rollout_name=f"test_{task.task_spec['task_id']}",
            environment=ENV_NAME,
            split=SPLIT,
            task_spec=task.task_spec
        )

        # Pass OpenAI API key for LLM grader
        async with environment.session(task=task, secrets={"openai_api_key": OPENAI_API_KEY}) as session:
            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]
            finished = False

            rollout.log_openai_response(message=input_list[0], is_finished=finished)

            while not finished:
                # Use responses.create() NOT chat.completions.create()
                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    reasoning={"effort": "high"},
                    tools=tools,
                    input=input_list  # Note: 'input' not 'messages'
                )

                rollout.log_openai_response(response.output[-1])
                input_list += response.output

                for item in response.output:
                    if item.type == "function_call":
                        tool_result = await session.call_tool(
                            item.name,
                            json.loads(str(item.arguments))
                        )

                        reward = tool_result.reward
                        finished = tool_result.finished

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text
                        })
                        rollout.log_openai_response(
                            input_list[-1],
                            reward=reward,
                            is_finished=finished
                        )

                        print(f"Tool: {item.name}")
                        print(f"Arguments: {item.arguments}")
                        print(f"Reward: {reward:.3f}")

                        if tool_result.finished:
                            finished = True
                            print('FINISHED!')
                            break

    print(f"\n{'=' * 70}")
    print("Testing complete!")


if __name__ == "__main__":
    asyncio.run(main())
