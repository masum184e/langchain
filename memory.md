# Memory
Memory refers to the ability to retain information across interactions in a conversation with a language model (LLM). It enables stateful conversations, where the context from previous user inputs and model outputs can be remembered and used in future steps.

By default, LLMs (like GPT) are stateless – they only respond based on the current input. However, in real-world applications (like chatbots, agents, or support assistants), it’s useful to retain past information. LangChain’s Memory modules help manage this contextual information automatically.

## Types of Memory
| Memory Type                      | Description                                                      |
| -------------------------------- | ---------------------------------------------------------------- |
| `ConversationBufferMemory`       | Stores the entire conversation as a string.                      |
| `ConversationBufferWindowMemory` | Stores the last *k* messages.                                    |
| `ConversationTokenBufferMemory`  | Stores messages until a token limit is reached.                  |
| `ConversationSummaryMemory`      | Uses LLM to summarize old messages and retains only the summary. |
| `VectorStoreRetrieverMemory`     | Stores facts in a vector store and retrieves relevant ones.      |
| `ZepMemory` (external)           | Plug-in for long-term memory support using Zep DB.               |

## Example
```py
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

# 1. Create memory instance
memory = ConversationBufferMemory()

# 2. Create LLM chain with memory
llm = ChatOpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 3. Interact with the conversation
print(conversation.predict(input="Hi, my name is Masum."))
print(conversation.predict(input="What is my name?"))
```
You can see the raw history like this:
```py
print(memory.buffer)
```
## Internals
- Stores the conversation as a list of messages (user/AI)
- Formats this into a prompt (e.g., "Human: ...\nAI: ...")
- Passes it to the LLM on each predict() call