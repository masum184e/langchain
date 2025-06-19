# Runnable
The `Runnable` interface in LangChain is a core abstraction that represents any component in a chain (e.g., LLMs, tools, prompts, retrievers) that can be executed with input and produce output. This allows for flexible composition of complex chains or pipelines by connecting multiple `Runnable` objects together.

Any process that takes input and produces output—whether it's a prompt template, a language model, a retriever, or a custom function—can be wrapped in a `Runnable`.

This provides:

- Standardization of input/output across components
- Composable chains (via `pipe`, `map`, `batch`, etc.)
- Streaming support
- Parallelism and branching

## Key Methods of Runnable
| Method                       | Description                                                                |
| ---------------------------- | -------------------------------------------------------------------------- |
| `invoke(input)`              | Runs the Runnable with a single input.                                     |
| `batch(inputs)`              | Runs the Runnable on a batch of inputs.                                    |
| `stream(input)`              | Streams output from the Runnable (useful for LLMs).                        |
| `transform(input_generator)` | Accepts an input stream and yields output stream (generator-to-generator). |

## Runnable Types
- `RunnableLambda`: Wraps a Python function
- `RunnableMap`: Runs multiple runnables in parallel
- `RunnableSequence`: Chains runnables in a sequence
- `RunnableParallel`: Runs runnables concurrently (returns dict)

You can also create your own class that implements the Runnable protocol.

## Example
**Scenario:**
You want to build a small chain where:
1. The input is a user's name.
2. A template generates a greeting prompt.
3. An LLM completes the greeting.

### Create a simple function to wrap with RunnableLambda
```py
def name_to_greeting(name: str) -> dict:
    return {"name": name}
greeting_func = RunnableLambda(name_to_greeting)
```
### Create the sequence
```py
chain = greeting_func | prompt | llm
```
- `RunnableLambda`: wraps a function that prepares the input.
- `RunnableSequence` (the `|` pipe operator): chains them into a flow.

## Advance Example
### Branching
```py
# Run two chains in parallel
parallel = RunnableMap({
    "greeting": chain,
    "raw_input": RunnableLambda(lambda x: x)
})

output = parallel.invoke("Bob")
```
### Batching
```py
names = ["Alice", "Bob", "Charlie"]
results = chain.batch(names)
for res in results:
    print(res.content)
```
### Streaming
```py
for chunk in chain.stream("Eve"):
    print(chunk.content, end="")
```
