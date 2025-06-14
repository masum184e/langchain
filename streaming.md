# Streaming
Streaming in LangChain refers to the ability to receive partial outputs from a language model in real time, as they're generated. This is particularly useful for building responsive applications like chatbots, voice interfaces, or collaborative tools, where users benefit from seeing content appear progressively rather than waiting for the entire response.

## Why Use Streaming
- **Better UX:** Improves user experience by displaying tokens as they’re generated.
- **Real-Time Interaction:** Helps in building assistants or apps where real-time responses matter.
- **Efficiency:** Allows processing or reacting to parts of a response without waiting for completion.

## Components Supporting Streaming
LangChain integrates streaming with several components:

- `ChatOpenAI` (or other chat models like Anthropic or Azure OpenAI)
- `Callbacks` (`StreamingStdOutCallbackHandler`, custom handlers)
- `Chain` classes (`LLMChain`, `ConversationChain`, etc.)

## Example
```py
# Instantiate the model with streaming enabled
chat = ChatOpenAI(
    streaming=True, 
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0.7
)
```
- `StreamingStdOutCallbackHandler` prints each token to stdout as it is generated.
- `streaming=True` tells the `ChatOpenAI` model to stream output instead of waiting for full generation.