# Contents

- []()
  - []()
  - []()
  - []()
  - []()
- [Large Language Models](#large-language-models)
  - [Models](#models)
    - [OpenAI GPT-4 / GPT-4o](#1-openai-gpt-4--gpt-4o)
    - [Anthropic Claude (Claude 3 series)](#2-anthropic-claude-claude-3-series)
    - [Google Gemini (Gemini 1.5)](#3-google-gemini-gemini-15)
    - [Meta LLaMA (LLaMA 3)](#4-meta-llama-llama-3)
    - [Mistral (Mistral 7B, Mixtral)](#5-mistral-mistral-7b-mixtral)
    - [Cohere Command R+](#6-cohere-command-r)
    - [xAI Grok](#7-xai-grok)
    - [TII Falcon LLM](#8-tii-falcon-llm)
    - [Aleph Alpha Luminous](#9-aleph-alpha-luminous)
    - [Command GPT (by Replit)](#10-command-gpt-by-replit)
  - [Overview](#summary-of-overview)
  - [LangChain Access](#summary-of-langchain-access-overview)
  - [Free Access](#summary-of-free-access)
- [LLM vs Chat Model](#llm-vs-chat-model-in-langchain)
  - [When to Use What?](#when-to-use-what)
  - [LLM](#using-llm-plain-completion-model)
  - [ChatModel](#using-chatmodel)
  - [Differences](#differences)

# Discriminative Model vs Generative Model

| Feature                  | Discriminative Model     | Generative Model           |                |         |
| ------------------------ | ------------------------ | -------------------------- | -------------- | ------- |
| Learns                   | (P(Y                     | X))                        | (P(X, Y) = P(X | Y)P(Y)) |
| Focus                    | Decision boundary        | Data generation process    |                |         |
| Classification accuracy  | Usually higher           | Often lower                |                |         |
| Ability to generate data | ❌ No                    | ✅ Yes                     |                |         |
| Examples                 | Logistic Regression, SVM | Naive Bayes, GAN, HMM, VAE |                |         |
| Training complexity      | Generally simpler        | More complex               |                |         |

# Large Language Models

## Models

### 1. OpenAI GPT-4 / GPT-4o

- **Developer:** OpenAI
- **Type:** General-purpose autoregressive language model
- **Key Use Cases:**
  - Conversational agents (e.g., ChatGPT)
  - Code generation and debugging
  - Creative writing (poetry, stories, scripts)
  - Translation and summarization
  - Tutoring and education support
  - Multimodal tasks (GPT-4o can process text, images, audio)
- **JSON Support:**
  - ✅ Yes
  - **How:** Strong support via prompting, and via the function calling / tool calling feature in the API.
  - **Special Features:** Native structured output with schemas.

### 2. Anthropic Claude (Claude 3 series)

- **Developer:** Anthropic
- **Key Focus:** Alignment with human intent and safety
- **Key Use Cases:**
  - Safe and ethical AI applications
  - Complex reasoning and summarization
  - Corporate documentation and Q&A
  - Enterprise tools (Claude for Slack, etc.)
- **JSON Support:**
  - ✅ Yes
  - **How:** Excellent at producing valid JSON using natural prompts (e.g., "respond in this format: {...}").
  - **Use in APIs:** Can be structured with output parsers, though not as formalized as OpenAI's function calling.

### 3. Google Gemini (Gemini 1.5)

- **Developer:** Google DeepMind
- **Former Name:** Bard
- **Key Use Cases:**
  - Integration with Google Workspace (Docs, Gmail, etc.)
  - Answering questions with Google Search access
  - Multimodal understanding (text, images, code)
  - Real-time assistant for browsing and research
- **JSON Support:**
  - ✅ Yes
  - **How:** Good JSON output on prompt. Limited formal schema support.
  - **Notes:** API does not currently support function calling like OpenAI but works well for general JSON outputs.

### 4. Meta LLaMA (LLaMA 3)

- **Developer:** Meta (Facebook AI Research)
- **Type:** Open-weight models (used for fine-tuning and research)
- **Key Use Cases:**
  - Research and experimentation
  - Chatbots and assistants
  - Embedding into applications (e.g., customer support tools)
  - Model customization (via fine-tuning)
- **JSON Support:**
  - ⚠️ Yes, with prompt engineering
  - **How:** Outputs valid JSON if prompted carefully, but not inherently robust (no schema enforcement).
  - **Used in:** Open-source apps with custom wrappers for JSON handling.

### 5. Mistral (Mistral 7B, Mixtral)

- **Developer:** Mistral.ai (French startup)
- **Type:** Open-weight, efficient transformer models
- **Key Use Cases:**
  - Lightweight deployments on local machines
  - Low-latency applications (due to size-efficiency)
  - Fine-tuning for niche use cases
  - Code and text generation
- **JSON Support:**
  - ⚠️ Yes, with good prompting
  - **How:** Outputs JSON well for simpler tasks, but prone to hallucinations in long structures.
  - **Best For:** Light JSON needs or embedded apps with post-processing.

### 6. Cohere Command R+

- **Developer:** Cohere
- **Focus:** Retrieval-Augmented Generation (RAG)
- **Key Use Cases:**
  - Document summarization
  - Enterprise knowledge management
  - QA systems with vector search
  - Chatbots with up-to-date company data
- **JSON Support:**
  - ✅ Yes (strong support)
  - **How:** Specifically designed for RAG and structured outputs.
  - **Notes:** Good at extracting information into predefined JSON templates.

### 7. xAI Grok

- **Developer:** xAI (Elon Musk’s AI company)
- **Available via:** X (formerly Twitter)
- **Key Use Cases:**
  - Social media integration
  - Real-time commentary and question answering
  - Content recommendation
- **JSON Support:**
  - ❌ Not officially documented
  - **How:** May support JSON with prompting, but limited tooling/API information available.

### 8. TII Falcon LLM

- **Developer:** Technology Innovation Institute (UAE)
- **Type:** Open-weight model
- **Key Use Cases:**
  - Academic research
  - Open-source AI development
  - Multilingual tasks
- **JSON Support:**
  - ⚠️ Yes, with structured prompts
  - **How:** Outputs simple JSON reliably with prompt guidance.
  - **Limitations:** Not reliable for complex nested structures.

### 9. Aleph Alpha Luminous

- **Developer:** Aleph Alpha (Germany)
- **Focus:** Explainability and transparency
- **Key Use Cases:**
  - Government and legal sector applications
  - Use in highly regulated industries
  - Reasoning and document analysis with traceability
- **JSON Support:**
  - ❌ Not a primary feature
  - **How:** Possible via prompting, but not designed for structured output tasks.
  - **Focus:** Traceable reasoning over structured data.

### 10. Command GPT (by Replit)

- **Developer:** Replit (based on open models, tuned for coding)
- **Key Use Cases:**
  - Code completion and debugging
  - In-browser development assistance
  - Coding education and tutorials
- **JSON Support:**
  - ✅ Yes
  - **How:** Designed for code generation, outputs JSON well when asked.
  - **Use Case:** Dev environments, CLI tools, local apps.

## Summary of Overview

| Model      | Creator    | Key Feature                    | Common Use Case                      |
| ---------- | ---------- | ------------------------------ | ------------------------------------ |
| GPT-4o     | OpenAI     | Multimodal, real-time          | Chatbots, tutoring, code, images     |
| Claude 3   | Anthropic  | Safety-focused, long context   | Business, academic support           |
| Gemini     | Google     | Deep integration, multimodal   | Workspace tools, research            |
| Mistral    | Mistral AI | Lightweight, fast open models  | On-device AI, custom finetuning      |
| LLaMA 3    | Meta       | Open weights, scalable         | Research, education, experimentation |
| Command R+ | Cohere     | Retrieval-augmented generation | Enterprise search and Q\&A           |
| Grok       | xAI (Elon) | Social media integration       | X (Twitter) assistant                |
| Falcon     | TII (UAE)  | Efficient open model           | General NLP tasks                    |
| Yi         | 01.AI      | Chinese-English optimization   | Multilingual generation              |
| BLOOM      | BigScience | Community and multilingual     | Research, fairness studies           |

## Summary of Langchain Access Overview

| Model                         | Free for Chat           | Free with LangChain          | Chat Access                                                   | LangChain Access                                                              |
| ----------------------------- | ----------------------- | ---------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **GPT-4o / GPT-3.5** (OpenAI) | ✅ (3.5) / ❌ (4o)      | ❌ (pay for 4/4o) / ✅ (3.5) | [chat.openai.com](https://chat.openai.com)                    | `OpenAI()` in LangChain with API key (`openai` module)                        |
| **Claude 3** (Anthropic)      | ✅ (Sonnet) / ❌ (Opus) | ❌ (all tiers paid via API)  | [claude.ai](https://claude.ai)                                | `ChatAnthropic()` via API key (`anthropic` module)                            |
| **Gemini 1.5** (Google)       | ✅ (basic use)          | ✅ (limited free usage)      | [gemini.google.com](https://gemini.google.com)                | `ChatGoogleGenerativeAI()` in LangChain via `google-genai` plugin             |
| **Mistral / Mixtral**         | ✅ (chat + API)         | ✅ (fully open-source)       | [HuggingFace Chat](https://huggingface.co/chat)               | `ChatOpenAI()` or `ChatMistralAI()` with Hugging Face or self-hosted endpoint |
| **LLaMA 2/3** (Meta)          | ✅ (via 3rd parties)    | ✅ (open weights)            | [HuggingFace](https://huggingface.co/chat), LM Studio         | Use LangChain with custom endpoints or models like `LlamaCpp`/`Transformers`  |
| **Command R / R+** (Cohere)   | ✅ (open models)        | ❌ (API access is paid)      | Some demos, no native UI                                      | `CohereRAGRetriever()` or custom wrapper via Cohere’s API                     |
| **Grok (xAI)**                | ❌ (X Premium+)         | ❌ No API                    | Inside X (Twitter)                                            | ❌ Not usable with LangChain                                                  |
| **Falcon** (TII)              | ✅ (fully open)         | ✅ (open weights)            | [HuggingFace](https://huggingface.co/TII)                     | Use LangChain via `Transformers` or custom endpoints                          |
| **Yi** (01.AI)                | ✅ (open)               | ✅ (open weights)            | Hugging Face, LM Studio                                       | Use LangChain via `Transformers` or `LlamaCpp`                                |
| **BLOOM** (BigScience)        | ✅ (open)               | ✅ (open weights)            | [BLOOM on HF](https://huggingface.co/spaces/bigscience/bloom) | Use via `Transformers` in LangChain                                           |

## Summary of Free Access

| Model      | Free with LangChain? | Comment                           |
| ---------- | -------------------- | --------------------------------- |
| GPT-3.5    | ✅ Yes               | Free tier has limits              |
| GPT-4/4o   | ❌ No                | Requires Plus/API subscription    |
| Claude 3   | ❌ No                | API requires paid account         |
| Gemini     | ✅ Yes (limited)     | Google AI Studio provides credits |
| Mistral    | ✅ Yes               | Fully open-source                 |
| LLaMA      | ✅ Yes               | Open-source, local or HF use      |
| Command R+ | ❌ No                | Paid API only                     |
| Grok       | ❌ No                | Closed platform                   |
| Falcon     | ✅ Yes               | Use via Hugging Face or local     |
| Yi         | ✅ Yes               | Open-source                       |
| BLOOM      | ✅ Yes               | Open-source                       |

# LLM vs Chat Model in LangChain

| Feature                      | `LLM` (langchain.llms)                     | `ChatModel` (langchain.chat_models)                     |
| ---------------------------- | ------------------------------------------ | ------------------------------------------------------- |
| **Input Format**             | Plain string (text prompt)                 | List of `ChatMessage` objects (`System`, `User`, `AI`)  |
| **Output**                   | Plain text                                 | `ChatMessage` object (role + content)                   |
| **Optimized For**            | Completion-style models                    | Chat-oriented models (like GPT-4, Claude, Gemini)       |
| **Example Models**           | OpenAI `text-davinci-003`, GPT-NeoX        | OpenAI `gpt-3.5-turbo`, Claude, Gemini                  |
| **Tool Support**             | Limited tool use, more basic               | Supports tool calling, system messages, memory, etc.    |
| **Conversation Memory**      | Harder to manage manually                  | Integrated memory support via message history           |
| **Function Calling Support** | ❌ Not available                           | ✅ Yes (for chat APIs like OpenAI's function calling)   |
| **Use Case Examples**        | Text generation, summarization, extraction | Conversational agents, multi-turn dialogues, assistants |

## When to Use What?

- Use `LLM` when:
  - You are using a completion model (like `text-davinci-003`)
  - You just want a single input → output generation
  - You don’t need roles (`user`, `assistant`, `system`)
- Use `ChatModel` when:
  - You’re using chat-based models like `gpt-4`, `Claude`, or `Gemini`
  - You need multi-turn memory, role distinction, or tool/function calling
  - You want better control over structured conversations

## Example

Install and set up:

```shell
pip install langchain google-generativeai
```

Set your API key:

```python
import os
os.environ["GOOGLE_API_KEY"] = "your-google-api-key"
```

LangChain's Gemini integration supports:

- `ChatGoogleGenerativeAI` → for **ChatModels** (recommended)
- `GoogleGenerativeAI` → for **LLMs** (text-only completions, not role-based)

### Using `LLM` (Plain Completion Model)

```python
from langchain_google_genai import GoogleGenerativeAI

# Instantiate the plain LLM
llm = GoogleGenerativeAI(model="gemini-2.0-flash")

# Send a simple string prompt
prompt = "List three benefits of drinking green tea."
response = llm.invoke(prompt)

print("LLM response:\n", response)
```

#### Explanation:

- A simple text prompt is sent.
- `LLM` returns plain text.
- Good for classic prompt→output patterns.

### Using `ChatModel`

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Instantiate the chat model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Define the structured messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What are the health benefits of green tea?")
]

# Get response
response = llm.invoke(messages)
print("ChatModel response:\n", response.content)
```

#### Explanation:

- Messages are role-tagged (`System`, `User`, `AI`) — simulating a real chat.
- Enables chat-specific features like memory, role control, and better context handling.

### Differences

| Feature                | `ChatGoogleGenerativeAI` (Chat Model)   | `GoogleGenerativeAI` (LLM)      |
| ---------------------- | --------------------------------------- | ------------------------------- |
| Input Format           | List of `SystemMessage`, `HumanMessage` | Single prompt string            |
| Output                 | `ChatMessage` (use `.content`)          | Raw string                      |
| Roles (System/User/AI) | ✅ Yes                                  | ❌ No                           |
| Use Case               | Assistant, dialog, memory, tools        | Simple completions, single-shot |
| Model Used             | `gemini-2.0-flash`                      | `gemini-2.0-flash`              |








# JSON supported prompt example

**Example 1:**

```
You are an API that returns user profiles in JSON format. Please follow this exact structure:

{
  "name": string,
  "age": integer,
  "email": string,
  "is_active": boolean,
  "preferences": {
    "language": string,
    "newsletter": boolean
  }
}

Generate a JSON object for a user named Sarah Connor, age 35, email "sconnor@example.com", active user, prefers English and is subscribed to the newsletter.

Respond only with a valid JSON object.
```

**Example 2:**

```
You are a system that generates user data in JSON format. Follow this format exactly, without any explanation or extra text. Your response must be valid JSON.

Format:
{
  "id": integer,
  "username": string,
  "email": string,
  "is_verified": boolean,
  "roles": [string]
}

Generate a JSON object for a user with:
- ID: 1023
- Username: "techwiz"
- Email: "techwiz@domain.com"
- Verified: true
- Roles: "admin", "editor"
```

### Prompting Tips

| Tip                                 | Why It Matters                                                                  |
| ----------------------------------- | ------------------------------------------------------------------------------- |
| Use `Respond only with valid JSON.` | Prevents the model from adding commentary or markdown.                          |
| Show the format first               | Few-shot style helps it infer the structure.                                    |
| Keep JSON structure simple          | Avoid deeply nested structures unless absolutely necessary.                     |
| Test with `temperature = 0.0`       | In APIs or apps like LM Studio, this reduces randomness and increases validity. |
| Add newline before JSON block       | Helps model understand it's entering “data mode.”                               |

# Notes

- [Few shot prompt technique is based on the Language Models are Few-Shot Learners paper](https://arxiv.org/abs/2005.14165)
- few things of Shot prompt template (1,2,3,4), read detail from documenation and gpt and add to the notebook
- llm model gula older, chat model gula latest
- chatPromptTemplate
  - what can be use as role
  - do a deep research on role attribute
- explore the command `pip install -U ...`
- Important Documentation:
  - [Conceptual Guides](https://python.langchain.com/docs/concepts/)
  - [Integration](https://python.langchain.com/docs/integrations/providers/)
  - [How To](https://python.langchain.com/docs/how_to/)
- [Providers Features](https://python.langchain.com/docs/integrations/chat/#featured-providers)
- For large dataset use `lazy_load` instead of `load`
- Tools
  - Runnable Interface
  - invoke
  - ainvoke
- what's the differences between structured and unstrutured data in langchain
# JSON Support

- Few Shot Prompt Template
- output parser
- prompting, function calling / tool calling - openai
- natural prompt - claude(excellent)
- prompt - Gemini(good)
- RAG - Command R+(Strong)
- when asked - Command GPT(good)
- [Check JSON Support](https://python.langchain.com/docs/integrations/chat/#featured-providers)