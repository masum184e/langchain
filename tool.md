- [Tools](#tools)
  - [Components](#components)
  - [Example](#example)
- [Agent](#agent)
  - [Components](#components-1)
  - [Types of Agent](#types-of-agent)
  - [Example](#example-1)
  - [Running Agents](#running-agents)

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
    """Add two numbers."""
    a, b = map(int, inputs.split(","))
    return str(a + b)
```

Once you have defined a tool, you can use it direcly: `add_numbers.invoke({"a": 2, "b": 3})`.

You can also inspect the tools property such `add_numbers.name`, `add_numbers.description`.

# Agent

LLM can't take actions - they just output text. Agents are systems that **take a high-level task** and **use an LLM as a reasoning engine** to **decide what actions to take** and execute those actions.

An agent consists of three components: a large language model (LLM), a set of tools it can use, and a prompt that provides instructions.

The LLM operates in a loop. In each iteration, it selects a tool to invoke, provides input, receives the result (an observation), and uses that observation to inform the next action. The loop continues until a stopping condition is met — typically when the agent has gathered enough information to respond to the user.

**Agents can:**

- Parse complex user queries.
- Choose tools dynamically.
- Handle multi-step reasoning.
- Perform intermediate reasoning steps with context.
- Perform just the specific job either define in the `tool` or in `prompt`.

## Components

1. **LLM**: The brain of the agent.
2. **Tools**: External functions/APIs the agent can call.
3. **Agent Type**: Strategy that defines how the agent thinks and acts.
4. **Memory (optional)**: Keeps history of past interactions.

## Example

**Initialize an agent with the tool:**

```py
agent = create_react_agent(
    model=llm,
    tools=[add_numbers],
    prompt="You are a helpful assistant"
)
```

`initialize_agent` was designed for sequential tasks and straightforward workflows. The LLM would decide on a tool, use it, observe the output, and then decide on the next step, all within a relatively rigid, linear "chain" of operations. While it could iterate, it wasn't designed for arbitrary, dynamic graph structures.

`create_react_agent` designed for non-linear workflows that involve loops and human-in-the-loop interactions. The ReAct pattern (Reason, Act, Observe, and then loop) is naturally represented as a cycle within a LangGraph.

## Running Agents

### Basic Usage

Agents can be executed in two primary modes:

- Synchronous using `.invoke()` or `.stream()`
- Asynchronous using `await .ainvoke()` or `async for` with `.astream()`

```py
response = agent.invoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]})
```

### Inputs and Output

Agents use a language model that expects a list of `messages` as an input. Therefore, agent inputs and outputs are stored as a list of `messages` under the `messages` key in the agent state.

#### Input Format

Agent input must be a dictionary with a `messages` key. Supported formats are:
| Format | Example |
|---------------------|-------------------------------------------------------------------------------------------|
| String | `{"messages": "Hello"}` — Interpreted as a `HumanMessage` |
| Message dictionary | `{"messages": {"role": "user", "content": "Hello"}}` |
| List of messages | `{"messages": [{"role": "user", "content": "Hello"}]}` |
| With custom state | `{"messages": [{"role": "user", "content": "Hello"}], "user_name": "Alice"}` — If using a `custom state_schema` |

#### Output Format

Agent output is a dictionary containing:

- `messages`: A list of all messages exchanged during execution (user input, assistant replies, tool invocations).
- Optionally, `structured_response` if structured output is configured.
- If using a custom `state_schema`, additional keys corresponding to your defined fields may also be present in the output. These can hold updated state values from tool execution or prompt logic.

#### Streaming

```py
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    stream_mode="updates"
):
    print(chunk)
```
