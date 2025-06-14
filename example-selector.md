# ExampleSelector
An `ExampleSelector` chooses examples from a list or database based on some logic (e.g., similarity to input) to be inserted into a `FewShotPromptTemplate`. This is useful when:

- You have a large pool of training examples.
- Including all examples is impractical due to token limits.
- You want examples most relevant to the user’s input.

An ExampleSelector is used primarily in prompt templating, especially with few-shot prompting. It helps dynamically select the most relevant examples to include in a prompt based on the current input.
## Common Types of Example Selectors
LangChain provides several built-in example selectors:

| Selector                              | Description                                                         |
| ------------------------------------- | ------------------------------------------------------------------- |
| `LengthBasedExampleSelector`          | Selects examples that fit within a token limit.                     |
| `SemanticSimilarityExampleSelector`   | Selects examples most similar to the input using embeddings.        |
| `MaxMarginalRelevanceExampleSelector` | A variation of semantic similarity that encourages diversity.       |
| `MultiPromptSelector`                 | Chooses between entirely different prompt templates based on input. |

## Example
**Goal:** You're building a model that reformulates questions. Based on the input, you want to provide similar question-reformulation examples.

**Create SemanticSimilarityExampleSelector**
```py
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),              # Embedding class
    FAISS,                           # Vector store
    k=2                              # Number of examples to retrieve
)
```

**Create the few-shot prompt template**
```py
prompt_template = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}"
    ),
    prefix="You are a helpful assistant that reformulates questions.\nHere are some examples:",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
)
```
## When to Use Which Selector?
| Selector                              | Use Case                                                                       |
| ------------------------------------- | ------------------------------------------------------------------------------ |
| `LengthBasedExampleSelector`          | Token constraints matter (e.g., GPT-3.5)                                       |
| `SemanticSimilarityExampleSelector`   | Personalized prompts with semantically relevant examples                       |
| `MaxMarginalRelevanceExampleSelector` | Balance relevance and diversity                                                |
| `MultiPromptSelector`                 | Inputs need drastically different prompts (e.g., summarization vs translation) |
