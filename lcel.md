# LangChain Expression Language

LangChain Expression Language (LCEL) is a declarative, composable, and chainable syntax introduced by LangChain to make it easier to construct chains of components like prompt templates, models, memory, output parsers, and tools — all with a functional programming flavor.

## Key Concepts of LCEL

LCEL provides a unified way to compose chains using a `|` (pipe) syntax, similar to Unix pipes or functional programming. It abstracts away boilerplate and focuses on how components connect, rather than how to instantiate each class manually.

Think of it as:

```
Input --> PromptTemplate --> LLM --> OutputParser --> Final Result
```

In LCEL, this is written:

```
prompt | llm | output_parser
```

## Advantages of LCEL

- **Declarative**: Clearly states how data flows.
- **Composable**: Reuse and combine components easily.
- **Readable**: Simple syntax, less verbose.
- **Asynchronous Support**: Built-in `ainvoke` support.
- **Streaming Support**: Easily add `.stream()` or `.astream()`.

## Core LCEL Components

- **PromptTemplate** — Format user input.
- **LLM / ChatModel** — Run the language model.
- **Output Parsers** — Convert raw LLM output to structured data.
- **Runnable Interface** — All components support `.invoke`, `.ainvoke`, `.batch`, `.stream`, etc.

## Example

**Advanced Example With Memory:**

```py
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage

memory_chain = RunnableLambda(lambda x: {"question": x["question"], "chat_history": x.get("chat_history", [])})

full_chain = memory_chain | prompt | llm | output_parser

# Run the chain
result = full_chain.invoke({"question": "What is the capital of France?"})
print(result)
```

## Guidelines
- If you are making a single LLM call, you don't need LCEL; instead call the underlying chat model directly.
- If you have a simple chain (e.g., prompt + llm + parser, simple retrieval set up etc.), LCEL is a reasonable fit, if you're taking advantage of the LCEL benefits.
- If you're building a complex chain (e.g., with branching, cycles, multiple agents, etc.) use LangGraph instead. Remember that you can always use LCEL within individual nodes in LangGraph.