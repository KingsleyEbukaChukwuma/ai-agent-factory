# **Agent Factory: Automated Multi-Agent System for ADK Agent Generation**

**Track: Enterprise Agents**

## **1. Problem Statement**

Designing ADK agents manually is slow, error-prone, and inconsistent.
Each agent requires specification design, tool definitions, safe prompts, code scaffolding, evaluation, and validation.
This leads to high variability in quality and unnecessary engineering overhead.

Organizations building many agents need a **standardized, deterministic pipeline** that converts natural-language requirements into fully structured, validated, and executable ADK agents.


## **2. Solution Overview**

**Agent Factory** is a multi-agent system that automates the entire ADK agent-creation workflow.
Given a single natural-language requirement, the system produces:

1. Formal requirements
2. JSON-structured agent specification
3. Tool definitions with typed schemas
4. System prompt aligned to constraints
5. Complete ADK Python module (agent + tools)
6. Evaluation across clarity, feasibility, safety, and prompt quality
7. Iterative refinement until a quality threshold is met
8. A final user-facing deliverable

The output is deterministic, validated, and immediately executable.

This automates what previously required multiple engineers and several hours.

## **3. Value**

* **Consistency:** All agents meet a minimum design quality threshold.
* **Speed:** Converts plain English to functional ADK agent modules in minutes.
* **Safety:** Enforces structured constraints and static code validation through custom tools.
* **Scalability:** Architecture supports many agent designs with predictable results.

For enterprises developing internal agent ecosystems, this reduces engineering load and eliminates design drift.


## **4. Architecture**

The factory is implemented as a multi-agent pipeline using Google’s ADK.

### **Components**

1. **Orchestrator (root agent)**

   * Extracts requirements
   * Calls all sub-agents
   * Produces the initial design bundle

2. **Agent Builder**

   * Converts natural language into strict JSON `agent_spec`
   * Defines goal, inputs, outputs, memory strategy, access mode, constraints

3. **Tool Builder**

   * Designs tool schemas
   * Generates Python stubs
   * Ensures typed inputs/outputs

4. **Code Builder**

   * Produces a complete ADK module
   * Configures tools with `FunctionTool`
   * Builds a session-aware helper runner

5. **Evaluator**

   * Scores the design across 5 dimensions:

     * clarity
     * tool design
     * prompt quality
     * feasibility
     * safety
   * Uses a custom **python_syntax_checker** tool
   * Rejects invalid code or unsafe operations

6. **Refiner**

   * Applies evaluator feedback
   * Improves spec, tools, prompt, and code

7. **Finalizer**

   * Generates a complete user-facing summary
   * Includes JSON + code skeleton + scores

### **Session & State Management**

The entire pipeline uses a **shared `InMemorySessionService`**.
All sub-agents share the same:

* `app_name`
* `user_id`
* `session_id`

This provides consistent conversational state across the pipeline.


## **5. Workflow**

1. User provides natural-language description
2. Orchestrator → structured `requirements`
3. Agent Builder → structured `agent_spec`
4. Tool Builder → tools + code skeleton
5. Code Builder → ADK module
6. Evaluator → scores design
7. Refiner → improves design
8. Loop continues until threshold met
9. Finalizer → complete output package

This process ensures that every generated agent meets a minimum quality bar.


## **6. Example Usage**

```python
from agent_factory import run_factory_once

result = await run_factory_once(
    "Build me an agent that analyzes weekly sales from SQL and proposes actions."
)

print(result)
```

This returns:

* agent_spec
* tools
* system prompt
* complete executable ADK module
* evaluation scores
* acceptance status


## **7. Repository Structure**

```
ai-agent-factory/
│
├── README.md
├── pyproject.toml
├── requirements.txt  
└── src/
    └── agent_factory/
        ├── __init__.py
        └── factory.py
```

## **8. Setup**

### **1. Install dependencies**

```bash
pip install -e .
```

### **2. Set environment variable**

```bash
export GOOGLE_API_KEY="your-key"
# or Windows:
setx GOOGLE_API_KEY "your-key"
```

### **3. Run the factory**

```python
from agent_factory import run_factory_once
```

## **9. Future Improvements**

* Agent deployment via Cloud Run / Agent Engine
* Static linting and type-checking of generated modules
* Visualization of agent pipelines
* Vector-store–augmented long-term memory
* Regression suite for agent correctness



