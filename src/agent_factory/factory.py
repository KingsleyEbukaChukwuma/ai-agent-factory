import os
from google import genai
def get_client():
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        raise RuntimeError("Set GOOGLE_API_KEY in your environment before running.")
    return genai.Client(api_key=API_KEY)


import json
import ast
from json import JSONDecodeError, JSONDecoder
from typing import Dict, Any, List
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.tools.function_tool import FunctionTool
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio
import warnings
MODEL_ID = "gemini-2.5-flash"
warnings.filterwarnings("ignore")
GLOBAL_SESSION_SERVICE = InMemorySessionService()
APP_NAME = "agents"
USER_ID = "kingsley"
SESSION_ID = "agent_builder_session"
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

llm_adapter = Gemini(
    model=MODEL_ID,
    retry_options=retry_config,
)
async def ensure_session(app_name: str, user_id: str, session_id: str):
    session = await GLOBAL_SESSION_SERVICE.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    if session is None:
        session = await GLOBAL_SESSION_SERVICE.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )

    return session
async def _run_agent_once(agent, app_name, user_id, session_id, prompt_text: str) -> str:
    """
    Executes a single agent call using the shared GLOBAL_SESSION_SERVICE.
    All sub-agent calls in the factory use the same (app_name, user_id, session_id),
    which allows the ADK runtime to preserve conversational history and state.
    """

    await ensure_session(app_name=app_name, user_id=user_id, session_id=session_id)
    # Reuse the global session service instead of creating a new one
    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=GLOBAL_SESSION_SERVICE,
    )

    content = types.Content(
        role="user",
        parts=[types.Part(text=prompt_text)],
    )

    final_text = None  # last piece of text we saw from any event

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        text_chunks = []

        agent_output = getattr(event, "agent_output", None)
        if agent_output is not None:
            ao_text = getattr(agent_output, "text", None)
            if ao_text:
                text_chunks.append(ao_text)

        content_obj = getattr(event, "content", None)
        if content_obj is not None and getattr(content_obj, "parts", None):
            for part in content_obj.parts:
                p_text = getattr(part, "text", None)
                if p_text:
                    text_chunks.append(p_text)

        if text_chunks:
            final_text = "\n".join(text_chunks)

    if final_text is None:
        raise RuntimeError(
            "No text found in any response event. "
            "Model may have only produced tool calls / function_call parts."
        )

    return final_text

def parse_first_json_object(text: str, source: str) -> Dict[str, Any]:
    """
    Extract and parse the FIRST JSON object from a model response.

    - Handles ```json fences
    - Handles extra garbage before/after the JSON
    - Raises a clear RuntimeError if parsing fails
    """
    raw = text.strip()

    # If fenced in ```...```, strip to the JSON block
    if raw.startswith("```"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end + 1]

    # Fallback: clip from first '{' to last '}' in case of leading/trailing noise
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]

    decoder = JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(raw)
        return obj
    except JSONDecodeError as e:
        raise RuntimeError(
            f"{source} did not return valid JSON: {e}\n\nGot:\n{text}"
        )
def python_syntax_checker_tool(code: str) -> Dict[str, Any]:
    """
    Validate Python code syntax.

    Args:
        code: Python source as a string.

    Returns:
        {
            "is_valid": bool,
            "error": str | None
        }
    """
    try:
        ast.parse(code)
        return {"is_valid": True, "error": None}
    except SyntaxError as e:
        return {
            "is_valid": False,
            "error": f"{e.__class__.__name__}: {e.msg} at line {e.lineno}, column {e.offset}",
        }
AGENT_BUILDER_INSTRUCTION = """
You are 'Agent Builder', a solution architect for AI agents that will be implemented using Google's Agent Development Kit (ADK) in Python.

INPUT
You receive a JSON-like object called requirements:

requirements = {
  "description": string,        # what the agent should do in plain language
  "memory_strategy": string,    # "in-session", "long-term", or "both"
  "data_sources": [string],     # e.g. ["SQL database", "API", "CSV files"]
  "access_mode": string,        # "read" or "read-write"
  "constraints": [string]       # safety / behaviour constraints
}

TASK
Design a clear agent specification (agent_spec) that will be used to build a Google ADK agent.

Your output MUST have this exact structure:

agent_spec = {
  "name": string,               # short, human-readable name
  "goal": string,               # one sentence describing what the agent does
  "inputs": [string],           # what the user / system provides
  "outputs": [string],          # what the agent returns
  "constraints": [string],      # operational / safety constraints
  "memory_strategy": string,    # "in-session", "long-term", or "both"
  "access_mode": string,        # "read" or "read-write"
  "requires_tools": boolean,    # true if it must call tools / functions
  "tool_intents": [
    {
      "purpose": string,        # what this tool is for (business description)
      "data_source": string,    # e.g. "SQL database", "Sales API", "File system"
      "mode": "read" | "read-write"
    },
    ...
  ]
}

RULES
- Be concrete. Tie the goal, inputs, outputs, and tool_intents to the requirements you are given.
- If requirements.data_sources is empty, infer reasonable sources from the description but stay realistic.
- If requirements.access_mode is "read", do NOT invent any write/update behaviour.
- Constraints should be concrete and enforceable, for example:
  - "Never perform destructive SQL operations (no INSERT, UPDATE, DELETE, DROP, ALTER)."
  - "Explain your reasoning briefly in plain English."
  - "Use only the configured data sources."

OUTPUT
Return ONLY the agent_spec object as valid JSON (no prose, no code, no comments).
"""


agent_builder = Agent(
    model=llm_adapter,
    name="agent_builder",
    description="Designs the high-level agent spec from requirements.",
    instruction=AGENT_BUILDER_INSTRUCTION,
)
agent_builder_tool = AgentTool(agent=agent_builder)
TOOL_BUILDER_INSTRUCTION = """
You are 'Tool Builder', responsible for designing the concrete tools an ADK agent will use.

INPUT
You receive:

agent_spec = {
  "name": string,
  "goal": string,
  "inputs": [string, ...],
  "outputs": [string, ...],
  "constraints": [string, ...],
  "memory_strategy": string,
  "access_mode": string,
  "requires_tools": bool,
  "tool_intents": [
    {
      "purpose": string,
      "data_source": string,
      "mode": string
    },
    ...
  ]
}

TASK

1) Design a minimal but sufficient list of tools that:
   - Directly support the agent's goal and tool_intents.
   - Respect the data sources and access_mode ("read" vs "read-write").
   - Avoid redundancy.

2) SPECIAL RULE FOR VISUALIZATION:
   - If agent_spec.outputs mention charts, plots, visualizations, or graphs,
     and there is no existing tool_intent for visualization, you MUST add
     a visualization tool.
   - Example:
     {
       "name": "generate_sales_chart",
       "description": "Generates a simple chart (e.g., bar or line) from aggregated sales data.",
       "inputs": {
         "data": "list[json]",
         "chart_type": "string",
         "x_axis": "string",
         "y_axis": "string",
         "title": "string"
       },
       "outputs": {
         "chart_object": "json"
       },
       "mode": "read"
     }

3) For each tool, define:
   - name: snake_case function name.
   - description: 1–2 sentences, concrete and specific.
   - inputs: a JSON object where each key is a parameter name and each value is a simple type string
     like "string", "int", "float", "bool", "list[json]", "json".
   - outputs: a JSON object with one or two named outputs and simple type strings.
   - mode: "read" or "read-write" consistent with agent_spec.access_mode and tool_intents.

4) Build a Python code skeleton string (tool_code_skeleton) that contains ONLY:
   - Top-level Python function stubs for each tool.
   - Each function’s signature must match the tools[].inputs.
   - Each function must return a sensible placeholder (or raise NotImplementedError).
   - No ADK Agent classes, no LLM configuration, no Runner, no sessions here.
   - This module is ONLY for tool function definitions.

OUTPUT

Return ONLY this JSON object:

{
  "tools": [
    {
      "name": "execute_sales_query",
      "description": "...",
      "inputs": { "sql_query": "string" },
      "outputs": { "rows": "list[json]" },
      "mode": "read"
    },
    ...
  ],
  "tool_code_skeleton": "..."
}

RULES
- Do NOT include markdown or ``` fences in tool_code_skeleton.
- The first character of tool_code_skeleton must be a Python keyword (e.g. 'd' from 'def').
- The last character of tool_code_skeleton must be part of valid Python code.
- Do NOT include any imports unless they are clearly needed for type hints or standard library use.
"""



tool_builder = Agent(
    model=llm_adapter,
    name="tool_builder",
    description="Turns tool intents into concrete tools and Python stubs.",
    instruction=TOOL_BUILDER_INSTRUCTION,
)
tool_builder_tool = AgentTool(agent=tool_builder)

EVALUATE_AGENT_INSTRUCTION = """
You are 'Evaluate Agent', a strict reviewer of AI agent designs.

All agents are assumed to be implemented in Python using Google's Agent Development Kit (ADK).

INPUT
You receive a design_bundle object:

design_bundle = {
  "requirements": {...},
  "agent_spec": {...},
  "tools": [...],
  "system_prompt": string,
  "code_skeleton": string    # Python module as text
}

TASK
Evaluate this design on 5 dimensions, each scored 1–5:

1) clarity_score
   - Is agent_spec understandable and consistent with requirements?
   - Are inputs / outputs / constraints well defined and non-ambiguous?

2) tool_design_score
   - Do tools align with tool_intents and data_sources?
   - Are tools minimal but sufficient? Any redundancy or obvious gaps?

3) prompt_quality_score
   - Does system_prompt clearly set the role, goal, inputs, outputs, constraints, and how to use tools?
   - Is it simple, direct, and not overloaded with fluff?

4) feasibility_score
   - Can this design be implemented realistically in ADK?
   - Does code_skeleton look like valid Python?
   - FRAMEWORK CHECK: if code_skeleton uses a non-ADK agent framework (e.g. Agency Swarm, LangChain) or obviously conflicts with ADK usage, this MUST reduce the feasibility_score.

5) safety_score
   - Are constraints sufficient to avoid dangerous or destructive behaviour?
   - For "read" agents, code_skeleton must not perform destructive operations (no INSERT/UPDATE/DELETE, no DROP/ALTER, etc.).
   - Is user data handled cautiously (no unnecessary exposure)?

Do NOT compute any aggregate, total, or percentage scores. 
Only output the five individual scores (1–5). The backend Python code will handle any further calculations.

TOOLS

You have access to a tool named python_syntax_checker.

When design_bundle.code_skeleton is present, you MUST call python_syntax_checker
with the full code_skeleton string as the "code" argument. Use the tool result to
inform the feasibility_score and safety_score:

- If is_valid is false, reduce feasibility_score and safety_score.
- Mention the syntax error in summary and improvement_suggestions.

Do not describe the tool call itself in your JSON. Just use its result.



OUTPUT
Return ONLY this JSON object:

{
  "clarity_score": int,          # 1–5
  "tool_design_score": int,      # 1–5
  "prompt_quality_score": int,   # 1–5
  "feasibility_score": int,      # 1–5
  "safety_score": int,           # 1–5
  "summary": string,             # 2–4 sentences, direct and honest
  "improvement_suggestions": [   # each item is a concrete change
    "Rewrite the goal to be more specific about inputs and outputs.",
    "Tighten SQL safety constraints and remove any write operations."
  ]
}
- The FIRST character of your output MUST be '{' and the LAST character MUST be '}'.
- Do NOT wrap it in ```json or ``` fences.
- Do NOT add any prose, markdown, or explanation.
- Do NOT prefix with labels like "Here is the JSON:".
Just output the JSON object itself.
"""

evaluate_agent = Agent(
    model=llm_adapter,
    name="evaluate_agent",
    description="Evaluates agent designs for clarity, tools, prompt, feasibility, and safety.",
    instruction=EVALUATE_AGENT_INSTRUCTION,
    tools=[python_syntax_checker_tool],
)
evaluate_agent_tool = AgentTool(agent=evaluate_agent)

REFINER_AGENT_INSTRUCTION = """
You are 'Refiner Agent', responsible for improving an AI agent design based on review feedback.

All agents are implemented in Python using Google's Agent Development Kit (ADK).

INPUT
You receive:
- design_bundle:

  {
    "requirements": {...},
    "agent_spec": {...},
    "tools": [...],
    "system_prompt": string,
    "code_skeleton": string    # Python module as text
  }

- eval_result:

  {
    "clarity_score": int,
    "tool_design_score": int,
    "prompt_quality_score": int,
    "feasibility_score": int,
    "safety_score": int,
    "summary": string,
    "improvement_suggestions": [string, ...]
  }

TASK
Produce an improved design_bundle that addresses the weaknesses in eval_result.

REFINEMENT RULES

- Preserve the overall intent and requirements. Do NOT change what the user wants.
- agent_spec:
  - Clarify name, goal, inputs, outputs, and constraints if they are vague.
  - Keep fields aligned with requirements.

- tools:
  - Align with tool_intents and requirements.data_sources.
  - Reduce redundancy and unnecessary complexity.
  - Preserve the contract used by Tool Builder (names and parameter shapes).

- system_prompt:
  - Make it simpler, more direct, and clearer about:
    - Role / goal
    - Inputs / outputs
    - How and when to use each tool
    - Safety / constraints

- code_skeleton:
  - TREAT AS PYTHON CODE.
  - Fix obvious Python syntax errors if any.
  - Ensure it is consistent with ADK usage (import from google.adk.agents, etc., or use simple Python functions that will be wired as tools).
  - DO NOT introduce non-ADK agent frameworks (no Agency Swarm, no LangChain). If such frameworks are present, refactor them back to an ADK-compatible structure or plain Python functions.

OUTPUT
- Return ONLY this updated design_bundle as JSON:
- The FIRST character of your output MUST be '{' and the LAST character MUST be '}'.
- Do NOT wrap it in ```json or ``` fences.
- Do NOT add any prose, markdown, or explanation.
- Do NOT prefix with labels like "Here is the JSON:".
Just output the JSON object itself.

{
  "requirements": {...},
  "agent_spec": {...},
  "tools": [...],
  "system_prompt": string,
  "code_skeleton": string
}
"""




refiner_agent = Agent(
    model=llm_adapter,
    name="refiner_agent",
    description="Refines an agent design using evaluation feedback.",
    instruction=REFINER_AGENT_INSTRUCTION,
)
refiner_agent_tool = AgentTool(agent=refiner_agent)

CODE_BUILDER_INSTRUCTION = """
You are 'Code Builder', responsible for generating a COMPLETE Python module
for a single ADK agent, using Google’s Agent Development Kit (ADK).

INPUT
You receive:

{
  "agent_spec": {...},
  "tools": [...],
  "tool_code_skeleton": "string",   # Python functions only
  "system_prompt": "string"
}

- agent_spec: final agent specification.
- tools: final tool definitions (from Tool Builder, possibly refined).
- tool_code_skeleton: Python code containing ONLY top-level function stubs for each tool.
- system_prompt: the detailed instruction text for the agent.

GOAL
Produce ONE complete Python module that:

1) Uses ADK's Agent class properly.
2) Uses ADK's FunctionTool for tools.
3) Does NOT hard-code any specific GenAI model or call genai.configure().
   The model will be provided externally by the app / runner.
4) Exposes a small helper function to run the child agent once.

STRUCTURE

Your output code must follow this structure:

1) Imports

   - Standard library imports as needed (typing, asyncio, etc.).
   - ADK imports:
       from typing import Any, Dict, List
       from google.adk.agents import Agent
       from google.adk.tools import FunctionTool
       from google.adk.runners import Runner
       from google.adk.sessions import InMemorySessionService

   - Do NOT import google.genai or call genai.configure here.
     LLM configuration is handled by the outer application.

2) Tool function definitions

   - Paste tool_code_skeleton as-is, but you may improve docstrings and comments.
   - Do NOT change the function names or parameters.
   - No ADK logic inside these functions. They are plain Python.

   IMPLEMENTATION RULES:
   - For generic tools, the default is:
       - raise NotImplementedError("Tool '<name>' not implemented.").
   - SPECIAL CASE: chart / visualization tools.
     If there is a tool whose name is exactly "generate_chart"
     (or very close, e.g. "generate_sales_chart") and whose description
     mentions "chart", "visualization", or "plot":

       * Implement the function body so that it does NOT draw the chart.
       * Instead, it MUST return a JSON-serializable dict with:
           {
             "chart_type": <string>,
             "title": <string>,
             "x_axis_field": <string>,
             "y_axis_field": <string>,
             "data_points": [ {"x": ..., "y": ...}, ... ],
             "code": <string>  # ready-to-run Python matplotlib code
           }

       * The "code" string MUST:
           - import matplotlib.pyplot as plt
           - define x and y as Python lists (taken from the input "data")
           - call plt.figure(), the correct plt.<type>() (e.g. bar/plot/pie),
             set title, xlabel, ylabel, rotate ticks, tight_layout, and show().
           - be fully runnable by the user as-is.

       * The function MUST extract x/y from the "data" input using the
         x_axis_field and y_axis_field parameters.

   This means: generate_chart tools are REAL implementations that output
   Python plotting code; other tools can remain NotImplementedError placeholders.


3) Tools list (FunctionTool)

   - Build a list of FunctionTool objects from the Python functions.
   - Example:

       sales_query_tool = FunctionTool(execute_sales_query)
       generate_chart_tool = FunctionTool(generate_sales_chart)

       TOOL_LIST = [sales_query_tool, generate_chart_tool]

   - Every tool in the 'tools' JSON must correspond to a Python function.
   - The FunctionTool names must match the function names.

4) Agent class

   - Define ONE Agent subclass, named from agent_spec.name, made safe as a Python identifier.
     For example, "Sales Analyzer Agent" -> class SalesAnalyzerAgent(Agent):
   - In __init__, call super().__init__ with:

       super().__init__(
           name=agent_spec.name,
           goal=agent_spec.goal,
           instruction=system_prompt,
           constraints=agent_spec.constraints,
           memory_strategy=agent_spec.memory_strategy,
           access_mode=agent_spec.access_mode,
           tools=TOOL_LIST,
           **kwargs
       )

   - Do NOT set self.model or call any GenAI client here.
   - Do NOT import or configure google.genai here.
   - Any additional comments should be minimal and practical.

5) Helper function: run_child_agent_once

   - Define:

       async def run_child_agent_once(message: str) -> str:
           \"\"\"Run the child agent once and return its final text reply.\"\"\"
           session_service = InMemorySessionService()
           agent = SalesAnalyzerAgent()
           runner = Runner(agent=agent, session_service=session_service)

           session_id = await runner.new_session()
           response = await runner.send_message(session_id=session_id, message=message)

           # Collect the final text from the response
           if hasattr(response, "text") and response.text:
               return response.text
           if hasattr(response, "agent_output") and response.agent_output and response.agent_output.text:
               return response.agent_output.text
           return "No response received."

   - You may adjust Runner usage to match the current ADK API,
     but keep it simple: new session, send one message, return final text.

OUTPUT

Return ONLY the code_skeleton string (the full Python module) as plain text.
Do NOT wrap it in ``` fences.
Do NOT include any JSON, markdown, or explanation.

RULES & CHECKS

- The module must be valid Python syntax (best effort).
- No calls to genai.configure or GenerativeModel.
- No direct HTTP/LLM calls; all LLM logic is through ADK.
- Tools must be wired with FunctionTool; do not manually construct types.Tool or FunctionDeclaration.
- Use the given system_prompt verbatim as the instruction.
- DO NOT import matplotlib or any plotting library at the module level.
- The agent must NEVER include `import matplotlib.pyplot as plt` anywhere in the module.
- If the agent needs chart code, it MUST be inside the chart tool’s returned string ONLY.
- The module must NOT import or reference matplotlib anywhere outside the returned code string.

"""



code_builder = Agent(
    model=llm_adapter,
    name="code_builder",
    description="Builds a full ADK Python module from agent_spec, tools and system_prompt.",
    instruction=CODE_BUILDER_INSTRUCTION,
)
code_builder_tool = AgentTool(agent=code_builder)

FINALIZER_AGENT_INSTRUCTION = """
You are 'Finalizer Agent'. Your job is to turn the final agent design into a clear, user-facing explanation.

INPUT
You receive a JSON object:

{
  "design_bundle": {
    "requirements": {...},
    "agent_spec": {...},
    "tools": [...],
    "system_prompt": "string",
    "code_skeleton": "string"
  },
  "eval_result": {
    "clarity_score": int,
    "tool_design_score": int,
    "prompt_quality_score": int,
    "feasibility_score": int,
    "safety_score": int,
    "summary": "string",
    "improvement_suggestions": [string, ...]
  },
  "percent": float,
  "accepted": bool
}

TASK
Produce ONE final answer for the end user that includes:

1) A short plain-language summary of what the agent does and how it behaves.
2) The final agent_spec as formatted JSON.
3) The final tools list as formatted JSON.
4) The FULL code_skeleton inside a ```python fenced code block.
5) The five evaluation scores (clarity, tool_design, prompt_quality, feasibility, safety).
6) The overall percent score (from the Python backend).
7) A short comment:
   - If accepted is true: "This design meets the quality bar."
   - If accepted is false: "This design did not fully meet the quality bar, but this is the best version after refinement."

RULES
- Do NOT recompute any scores or percentages. Use the numbers you are given.
- Do NOT modify scores or percent.
- Be simple, direct, and structured.

OUTPUT
Return ONLY the final user-facing answer as plain text (with JSON and code fenced where needed).
"""

finalizer_agent = Agent(
    model=llm_adapter,
    name="finalizer_agent",
    description="Turns the final design bundle and evaluation result into a clean user-facing answer.",
    instruction=FINALIZER_AGENT_INSTRUCTION,
)
finalizer_agent_tool = AgentTool(agent=finalizer_agent)

MAIN_ORCHESTRATOR_INSTRUCTION = """
You are 'Main Orchestrator', the coordinator of an AI agent factory.

You are NOT responsible for evaluation, refinement, or final user-facing formatting.
Your ONLY job is to:

- Understand the user's description of the agent they want.
- Turn that into a structured set of requirements.
- Use child tools to build an initial design_bundle.
- Return ONLY that initial design_bundle as JSON.

You have access to these nested agent tools:
- agent_builder_tool    -> designs agent_spec from requirements
- tool_builder_tool     -> designs tools + tool_code_skeleton from agent_spec
- code_builder_tool     -> builds full ADK Python module from agent_spec, tools, tool_code_skeleton, system_prompt

All resulting agents are implemented in Python using Google's Agent Development Kit (ADK).

========================
PHASE 1 – REQUIREMENTS
========================

You receive a single natural-language message from the user describing the agent they want.

From that message, construct an internal object:

requirements = {
  "description": string,              # what the agent should do
  "memory_strategy": string,          # "in-session", "long-term", or "both"
  "data_sources": [string],           # e.g. ["SQL database", "API", "CSV files"]
  "access_mode": string,              # "read" or "read-write"
  "constraints": [string]             # safety / behaviour constraints
}

Inference rules:
- If memory is not stated → assume "in-session".
- If data sources are not explicit → infer them from context (e.g. "weekly sales" → "SQL database").
- If write access is not explicit → assume "read".
- Always include basic safety constraints, such as:
  - "No destructive database operations (no INSERT, UPDATE, DELETE, DROP, ALTER)."
  - "Explain reasoning briefly in plain English."

You DO NOT output the requirements object. It is internal for you.

=====================
PHASE 2 – INITIAL DESIGN
=====================

Using the internal requirements:

1) Call agent_builder_tool with the requirements to get agent_spec.

2) Call tool_builder_tool with agent_spec to get:

   {
     "tools": [...],
     "tool_code_skeleton": "..."
   }

3) Construct a system_prompt (in your reasoning) that clearly explains:
   - The agent's role and goal.
   - Expected inputs and outputs.
   - How and when to use each tool.
   - Safety / constraints.
   - Memory behaviour (e.g. "You only remember within this session").

4) Call code_builder_tool with:

   {
     "agent_spec": agent_spec,
     "tools": tools,
     "tool_code_skeleton": tool_code_skeleton,
     "system_prompt": system_prompt
   }

   to produce code_skeleton (a full ADK Python module).

5) Build a design_bundle:

design_bundle = {
  "requirements": requirements,
  "agent_spec": agent_spec,
  "tools": tools,
  "system_prompt": system_prompt,
  "code_skeleton": code_skeleton
}

OUTPUT

- Return ONLY the design_bundle as raw JSON text.
- The FIRST character of your output MUST be '{' and the LAST character MUST be '}'.
- Do NOT wrap it in ```json or ``` fences.
- Do NOT add any prose, markdown, or explanation.
- Do NOT prefix with labels like "Here is the JSON:".
Just output the JSON object itself.
"""

root_agent = Agent(
    model=llm_adapter,
    name="agent_factory_orchestrator",
    description="Collects requirements and builds an initial ADK agent design bundle.",
    instruction=MAIN_ORCHESTRATOR_INSTRUCTION,
    tools=[
        agent_builder_tool,
        tool_builder_tool,
        code_builder_tool, 
    ],
)
def extract_json_block(text: str) -> str:
    """
    Best-effort extraction of the first JSON-like object from a model response.
    DOES NOT PARSE, just returns the substring from first '{' to last '}'.
    """
    raw = text.strip()

    # If fenced, strip fences first
    if raw.startswith("```"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return raw[start:end + 1]

    # Generic: take from first '{' to last '}'
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]

    # Fallback: just return original text (finalizer/evaluator can cope)
    return raw
MAX_REFINEMENTS = 5
TARGET_PERCENT = 85.0


async def run_factory_once(user_message: str) -> str:
    """

    1) Call root_agent (Main Orchestrator) to get initial design_bundle (as text).
    2) Run evaluation + refinement loop in Python:
       - Call evaluate_agent
       - Parse eval_result JSON
       - Use compute_score_percent() to get percent
       - If percent < TARGET_PERCENT, call refiner_agent with the design text
    3) Call finalizer_agent with the final design text, eval result, percent, and accepted flag.
    4) Return the final user-facing text from finalizer_agent.
    """

    # Initial design from root_agent
    design_bundle_text = await _run_agent_once(
        agent=root_agent,
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        prompt_text=user_message.strip(),
    )

    # keep it as text.
    best_design_text = extract_json_block(design_bundle_text)

    best_eval: Dict[str, Any] = {}
    best_percent: float = 0.0
    accepted: bool = False

    # Evaluation + refinement loop in Python
    for iteration in range(MAX_REFINEMENTS + 1):
        # 2a. Evaluate current design
        
        eval_prompt = json.dumps({"design_bundle": best_design_text})
        eval_text = await _run_agent_once(
            agent=evaluate_agent,
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
            prompt_text=eval_prompt,
        )

        # HERE we parse JSON, but this is small and should be valid
        eval_result = parse_first_json_object(
            eval_text,
            source="evaluate_agent",
        )

        # Compute percent using Python function
        best_percent = compute_score_percent(eval_result)
        best_eval = eval_result

        # Acceptance check
        if best_percent >= TARGET_PERCENT:
            accepted = True
            break

        # If this was the last allowed iteration, stop (not accepted)
        if iteration == MAX_REFINEMENTS:
            accepted = False
            break

        # 2d. Refine the design
        refine_prompt = json.dumps({
            "design_bundle": best_design_text,
            "eval_result": eval_result,
        })
        refine_text = await _run_agent_once(
            agent=refiner_agent,
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
            prompt_text=refine_prompt,
        )

        best_design_text = extract_json_block(refine_text)

    # Finalize for the user
    finalizer_prompt = json.dumps({
        "design_bundle": best_design_text,  # TEXT, not dict
        "eval_result": best_eval,           # dict
        "percent": best_percent,
        "accepted": accepted,
    })
    final_text = await _run_agent_once(
        agent=finalizer_agent,
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        prompt_text=finalizer_prompt,
    )

    return final_text
def _to_str(result: Any) -> str:
    """Normalize ADK run() result to string."""
    if isinstance(result, str):
        return result
    # Some setups might return dict-like with 'output'
    if isinstance(result, dict) and "output" in result:
        return str(result["output"])
    return str(result)


def _parse_json_or_raise(raw: Any) -> Dict[str, Any]:
    text = _to_str(raw).strip()
    # Sometimes model wraps JSON in ```json ... ```; strip that if needed
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)
async def call_agent_builder(requirements: dict) -> dict:
    prompt = json.dumps({"requirements": requirements}, indent=2)
    raw = await _run_agent_once(
        agent=agent_builder,
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        prompt_text=prompt,
    )
    agent_spec = _parse_json_or_raise(raw)
    return agent_spec

def call_tool_builder(agent_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send agent_spec to Tool Builder and return:
    {
      "tools": [...],
      "tool_code_skeleton": "..."
    }
    """
    prompt = json.dumps({"agent_spec": agent_spec}, indent=2)
    raw = tool_builder.run(prompt)
    tool_bundle = _parse_json_or_raise(raw)
    # Ensure keys exist
    tool_bundle.setdefault("tools", [])
    tool_bundle.setdefault("tool_code_skeleton", "")
    return tool_bundle
def compute_score_percent(eval_result: Dict[str, Any]) -> float:
    scores = [
        eval_result.get("clarity_score", 0),
        eval_result.get("tool_design_score", 0),
        eval_result.get("prompt_quality_score", 0),
        eval_result.get("feasibility_score", 0),
        eval_result.get("safety_score", 0),
    ]
    total = sum(scores)
    max_total = 25
    return (total / max_total) * 100.0
def generate_system_prompt(agent_spec: Dict[str, Any], tools: Any) -> str:
    """
    Build a system prompt for the child agent based on the spec and tools.
    """
    tool_lines = []
    for t in tools or []:
        tool_lines.append(
            f"- {t.get('name')}: {t.get('description')} "
            f"(inputs: {t.get('inputs')}, outputs: {t.get('outputs')})"
        )
    tools_text = "\n".join(tool_lines) if tool_lines else "None (no external tools)."

    constraints = agent_spec.get("constraints", [])
    constraints_text = "\n".join(f"- {c}" for c in constraints) if constraints else "None."

    prompt = f"""
You are an AI agent named "{agent_spec.get('name')}".
Goal: {agent_spec.get('goal')}

Inputs you expect:
- {", ".join(agent_spec.get("inputs", [])) or "None specified"}

Outputs you must produce:
- {", ".join(agent_spec.get("outputs", [])) or "None specified"}

Memory strategy: {agent_spec.get("memory_strategy")}
Access mode: {agent_spec.get("access_mode")}

Tools available:
{tools_text}

Constraints:
{constraints_text}

RULES:
- Follow the constraints strictly.
- Use tools only when they help you achieve the goal.
- When you use a tool, explain briefly why.
- Be clear and structured in your outputs.
    """.strip()
    return prompt
def build_code_skeleton(
    agent_spec: Dict[str, Any],
    tools_bundle: Dict[str, Any],
    system_prompt: str,
) -> str:
    """
    Generate a Python module string that:
    - Contains the tool stubs from Tool Builder
    - Defines a child ADK Agent using those tools
    - Provides an async helper to run the agent once
    """
    tools = tools_bundle.get("tools", [])
    tool_code = tools_bundle.get("tool_code_skeleton", "").rstrip()

    tool_names = ", ".join(t["name"] for t in tools) if tools else ""
    child_agent_name = agent_spec.get("name", "child_agent").lower().replace(" ", "_")

    # If there are no tools, tool list should be empty
    tools_list_expr = f"[{tool_names}]" if tool_names else "[]"

    skeleton = f'''"""
Auto-generated ADK agent module for: {agent_spec.get("name")}
"""

import os
import asyncio
from google import genai
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Configure your API key outside this file, for example:
#   export GOOGLE_API_KEY="your-key"
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY environment variable before running this module.")

client = get_client()

# ========== Tool functions ==========
{tool_code or "# No tools required for this agent."}

# ========== Agent definition ==========
{child_agent_name} = Agent(
    model="{MODEL_ID}",
    name="{child_agent_name}",
    description={json.dumps(agent_spec.get("goal") or "")},
    instruction={json.dumps(system_prompt)},
    tools={tools_list_expr},
)

async def run_child_agent_once(message: str) -> str:
    \"\"\"Run the child agent once and return its final text reply.\"\"\"
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="child_agent_app",
        user_id="local_user",
        session_id="local_session",
    )

    runner = Runner(
        agent={child_agent_name},
        app_name="child_agent_app",
        session_service=session_service,
    )

    content = types.Content(
        role="user",
        parts=[types.Part(text=message)],
    )

    final_text = None
    async for event in runner.run_async(
        user_id="local_user",
        session_id="local_session",
        new_message=content,
    ):
        if getattr(event, "is_final_response", None) and event.is_final_response():
            if event.content and event.content.parts:
                part_text = getattr(event.content.parts[0], "text", None)
                if part_text:
                    final_text = part_text

    return final_text or ""

if __name__ == "__main__":
    async def _demo():
        reply = await run_child_agent_once("Replace this with a real request.")
        print(reply)

    asyncio.run(_demo())
'''
    return skeleton
def build_design_bundle(
    requirements: Dict[str, Any],
    agent_spec: Dict[str, Any],
    tools_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    tools = tools_bundle.get("tools", [])
    system_prompt = generate_system_prompt(agent_spec, tools)
    code_skeleton = build_code_skeleton(agent_spec, tools_bundle, system_prompt)

    design_bundle = {
        "requirements": requirements,
        "agent_spec": agent_spec,
        "tools": tools,
        "system_prompt": system_prompt,
        "code_skeleton": code_skeleton,
    }
    return design_bundle
def evaluate_design(design_bundle: Dict[str, Any]) -> Dict[str, Any]:
    prompt = json.dumps({"design_bundle": design_bundle}, indent=2)
    raw = evaluate_agent.run(prompt)
    eval_result = _parse_json_or_raise(raw)
    eval_result["percent"] = compute_score_percent(eval_result)
    return eval_result
def refine_design(design_bundle: Dict[str, Any], eval_result: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "design_bundle": design_bundle,
        "eval_result": eval_result,
    }
    prompt = json.dumps(payload, indent=2)
    raw = refiner_agent.run(prompt)
    improved_bundle = _parse_json_or_raise(raw)
    return improved_bundle
def refine_until_approved(
    design_bundle: Dict[str, Any],
    max_iterations: int = 5,
    target_percent: float = 85.0,
):
    current_bundle = design_bundle
    history = []

    for i in range(1, max_iterations + 1):
        eval_result = evaluate_design(current_bundle)
        history.append({"iteration": i, "eval_result": eval_result})

        if eval_result["percent"] >= target_percent:
            return {
                "design_bundle": current_bundle,
                "history": history,
                "status": "approved",
            }

        # refine and loop
        current_bundle = refine_design(current_bundle, eval_result)

    # If we reach here, we never hit the target
    return {
        "design_bundle": current_bundle,
        "history": history,
        "status": "max_iterations_reached",
    }



