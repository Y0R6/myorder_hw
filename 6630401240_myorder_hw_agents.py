import os
import logging
import google.cloud.logging

from dotenv import load_dotenv

from google.adk import Agent
from google.adk.models import Gemini
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool
from google.genai import types

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()

model_name = os.getenv("MODEL")

# --- Tools ---

def append_to_state(
    tool_context: ToolContext, field: str, response: str
) -> dict[str, str]:
    """Append new output to an existing state key."""
    existing_state = tool_context.state.get(field, [])
    tool_context.state[field] = existing_state + [response]
    logging.info(f"[Added to {field}] {response}")
    return {"status": "success"}

def write_file(
    tool_context: ToolContext,
    directory: str,
    filename: str,
    content: str
) -> dict[str, str]:
    """Writes content to a file in a specified directory."""
    target_path = os.path.join(directory, filename)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w") as f:
        f.write(content)
    logging.info(f"File written to {target_path}")
    return {"status": "success"}

def exit_loop(tool_context: ToolContext) -> dict:
    """Exits the current loop."""
    tool_context.exit_loop = True
    return {"status": "exited loop"}

# --- Agents ---

# Step 2: The Investigation (Parallel)
admirer_researcher = Agent(
    name="admirer_researcher",
    model=Gemini(model=model_name, retry_options={"initial_delay": 1, "attempts": 2}),
    description="Researches the positive aspects and achievements of a historical topic.",
    instruction="""
    You are The Admirer. Your task is to research the positive aspects, successes, and achievements of the topic: { TOPIC? }.
    - Use your Wikipedia tool to search for information. Append terms like 'achievements', 'successes', 'positive impact' to your search query to get relevant results.
    - Use the 'append_to_state' tool to save your findings to the 'pos_data' field.
    - Summarize the positive findings.
    """,
    tools=[
        LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
        append_to_state,
    ],
)

critic_researcher = Agent(
    name="critic_researcher",
    model=Gemini(model=model_name, retry_options={"initial_delay": 1, "attempts": 2}),
    description="Researches the negative aspects and controversies of a historical topic.",
    instruction="""
    You are The Critic. Your task is to research the failures, controversies, and negative aspects of the topic: { TOPIC? }.
    - Use your Wikipedia tool to search for information. Append terms like 'controversy', 'failures', 'criticism', 'negative impact' to your search query to get relevant results.
    - Use the 'append_to_state' tool to save your findings to the 'neg_data' field.
    - Summarize the negative findings.
    """,
    tools=[
        LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
        append_to_state,
    ],
)

investigation_team = ParallelAgent(
    name="investigation_team",
    description="Conducts parallel research on positive and negative aspects of a topic.",
    sub_agents=[admirer_researcher, critic_researcher],
)

# Step 3: The Trial & Review (Loop)
judge = Agent(
    name="judge",
    model=Gemini(model=model_name, retry_options={"initial_delay": 1, "attempts": 2}),
    description="Judges the collected data and decides if more research is needed or to conclude.",
    instruction="""
    You are The Judge. Review the evidence presented.
    Positive Evidence: { pos_data? }
    Negative Evidence: { neg_data? }

    - Your task is to check if the information for both sides is sufficient and balanced.
    - If either 'pos_data' or 'neg_data' is empty or significantly less detailed than the other, you must order more research.
    - To order more research, provide a summary of what is missing and what needs to be found. The loop will then restart the investigation.
    - If you find the information on both sides to be sufficient and balanced, you MUST use the 'exit_loop' tool to conclude the trial.
    """,
    tools=[exit_loop],
)

trial_and_review = LoopAgent(
    name="trial_and_review",
    description="Repeatedly investigates and judges until the data is balanced.",
    sub_agents=[investigation_team, judge],
    max_iterations=3,
)

# Step 4: The Verdict (Output)
verdict_writer = Agent(
    name="verdict_writer",
    model=Gemini(model=model_name, retry_options={"initial_delay": 1, "attempts": 2}),
    description="Summarizes the findings and writes the final verdict to a file.",
    instruction="""
    You are the Court Scribe. Your final duty is to write the verdict report.
    Topic: { TOPIC? }
    Positive Evidence: { pos_data? }
    Negative Evidence: { neg_data? }

    - Create a balanced report summarizing and comparing the facts from both the positive and negative evidence.
    - Use the 'write_file' tool to save your report.
    - The filename should be the topic name with '.txt' extension.
    - The directory should be 'verdicts'.
    - The content should be your full, balanced report.
    """,
    tools=[write_file],
)

# Step 1 & Orchestration
historical_court = SequentialAgent(
    name="historical_court",
    description="Orchestrates the entire historical court process from inquiry to verdict.",
    sub_agents=[
        trial_and_review,
        verdict_writer,
    ],
)

root_agent = Agent(
    name="inquiry_clerk",
    model=Gemini(model=model_name, retry_options={"initial_delay": 1, "attempts": 2}),
    description="Starts the historical court process by getting a topic from the user.",
    instruction="""
    - You are the Inquiry Clerk of the Historical Court.
    - Greet the user and ask them to name a historical figure or event to be analyzed.
    - When they respond, use the 'append_to_state' tool to store their response in the 'TOPIC' state key.
    - Then, transfer control to the 'historical_court' agent to begin the proceedings.
    """,
    tools=[append_to_state],
    sub_agents=[historical_court],
)