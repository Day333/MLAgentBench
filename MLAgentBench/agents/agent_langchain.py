""" This file contains the agent class for the LangChain agent, which adapts LangChain agents to the MLAgentBench framework."""

import os

from pydantic import create_model, Field
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from MLAgentBench.schema import Action
from .agent import Agent


class EnvTool:
    """ A wrapper class to wrap actions as tools for the LangChain agent. """
    def __init__(self, action_info, env):
        self.action_info = action_info
        self.env = env

    def run(self, **action_input) -> str:
        """Run the wrapped action with the arguments supplied by the LLM's tool call."""
        try:
            return self.env.execute(Action(self.action_info.name, action_input))
        except Exception as e:
            return f"EnvException: {e}"


def build_args_schema(action_info):
    """ Build a pydantic schema (one string field per usage entry) so the LLM can call the
    action as a proper function-calling tool instead of packing everything into one string. """
    fields = {k: (str, Field(description=v)) for k, v in action_info.usage.items()}
    model_name = "".join(w.capitalize() for w in action_info.name.split()) + "Input"
    return create_model(model_name, **fields)


class LangChainAgent(Agent):
    """ A wrapper class to run a LangChain (>=1.0) tool-calling agent inside the MLAgentBench framework. """

    def run(self, env):

        # init chat model
        if self.args.llm_name.startswith("claude"):
            llm = ChatAnthropic(
                model=self.args.llm_name,
                anthropic_api_key=open("claude_api_key.txt").read().strip(),
                temperature=0.5,
                max_tokens=2000,
            )
        elif self.args.llm_name.startswith("qwen"):
            llm = ChatOpenAI(
                model=self.args.llm_name,
                api_key=open("qwen_api_key.txt").read().strip(),
                base_url="https://ws-r1de3xs467h5tod7.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                temperature=0.5,
                max_tokens=2000,
            )
        else:
            # TODO: add support for other providers
            raise NotImplementedError

        tools = [
            StructuredTool.from_function(
                func=EnvTool(self.action_infos[tool_name], env).run,
                name=tool_name,
                description=self.action_infos[tool_name].description,
                args_schema=build_args_schema(self.action_infos[tool_name]),
            )
            for tool_name in self.prompt_tool_names
        ]

        system_prompt = (
            f"Research Problem: {env.research_problem}\n\n"
            "You have access to the tools below. Call them as needed to make progress on the "
            "research problem, then give your final answer via the Final Answer tool."
        )

        with open(os.path.join(self.log_dir, "main_log"), "a", 1) as f:
            f.write(system_prompt + "\n")

        checkpointer = InMemorySaver()
        agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt, checkpointer=checkpointer)
        config = {"recursion_limit": self.args.agent_max_steps * 2 + 5, "configurable": {"thread_id": "main"}}

        try:
            result = agent.invoke({"messages": [("user", env.research_problem)]}, config=config)
            messages = result["messages"]
        except GraphRecursionError:
            # ran out of steps without calling Final Answer; salvage whatever progress was made
            messages = agent.get_state(config).values["messages"]

        with open(os.path.join(self.log_dir, "step_log.log"), "a", 1) as f:
            for message in messages:
                f.write(f"{type(message).__name__}: {message.content}\n\n")

        self.save(os.path.join(self.log_dir, "agent_final.json"))

        final_answers = [m.content for m in messages if getattr(m, "type", None) == "ai" and m.content]
        return final_answers[-1] if final_answers else "Agent ran out of steps without submitting a final answer."
