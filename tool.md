# Tools

Tools in LangChain are **wrappers around functions** that a **llm can call** through agents.

Tools extend LLMs’ capabilities beyond pure text generation:

- Call real-time APIs (weather, finance, etc.)
- Perform math or logic
- Interact with databases
- Use code execution environments
- Combine with retrieval (RAG) for document Q&A

Tools in LangChain are **how LLMs access external functionality**. Combine tools with agents to create intelligent workflows.

## Components

- `name`: Unique identifier.
- `func`: The actual function the tool wraps.
- `description`: Text that helps the agent know when to use this tool.
- `args_schema`: (Optional) Pydantic schema to validate structured input.

## Example

Scenario: Perform calculation(addition) on two number.

**Define the function:**

```py
def add_numbers(inputs: str) -> str:
    a, b = map(int, inputs.split(","))
    return str(a + b)
```

**Wrap it as a tool:**

```py
add_tool = Tool(
    name="AddTwoNumbers",
    func=add_numbers,
    description="Use this tool when you need to add two integers. Input format: '3,5'"
)
```

**Latest Way** to create a tool is using the `@tool` decorator

```py
@tool
def add_numbers(inputs: str) -> str:
    a, b = map(int, inputs.split(","))
    return str(a + b)
```

Once you have defined a tool, you can use it direcly: `add_numbers.invoke({"a": 2, "b": 3})`.

You can also inspect the tools property such `add_numbers.name`, `add_numbers.description`.

# Agent

LLM can't take actions - they just output text. Agents are systems that **take a high-level task** and **use an LLM as a reasoning engine** to **decide what actions to take** and execute those actions.

LangGraph is an extension of LangChain specifically aimed at creating highly controllable and customizable agents. It is recommended to use LangGraph for building agents.

**Agents can:**

- Parse complex user queries.
- Choose tools dynamically.
- Handle multi-step reasoning.
- Perform intermediate reasoning steps with context.

## Components

1. **LLM**: The brain of the agent.
2. **Tools**: External functions/APIs the agent can call.
3. **Agent Type**: Strategy that defines how the agent thinks and acts.
4. **Memory (optional)**: Keeps history of past interactions.

## Types of Agent

| Agent Type                              | Description                                                         |
| --------------------------------------- | ------------------------------------------------------------------- |
| `ZERO_SHOT_REACT_DESCRIPTION`           | Simple, effective agent that chooses tools using tool descriptions. |
| `OPENAI_FUNCTIONS`                      | Leverages OpenAI’s native function-calling interface.               |
| `STRUCTURED_CHAT_ZERO_SHOT_REACT`       | Similar to ZERO_SHOT but with structured input/output.              |
| `CHAT_CONVERSATIONAL_REACT_DESCRIPTION` | For multi-turn conversations with memory.                           |

## Example

**Initialize an agent with the tool:**

```py
agent = initialize_agent(
    tools=[add_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

**Run the Agent:**

```py
result = agent.run("What is the result of adding 10 and 22?")
print(result)
```

Agents can also maintain context across turns. Use `agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION` and now it can remember previous messages — useful for chatbots and assistants.

## Summary
| Concept          | Description                                                                  |
| ---------------- | ---------------------------------------------------------------------------- |
| **Tool**         | A callable function with a name and description                              |
| **Agent**        | The brain that decides *when* and *how* to call a tool                       |
| **Tool Calling** | The process of the agent invoking external functions                         |
| **LLM Role**     | Parses, plans, and calls tools as needed based on prompt + tool descriptions |
