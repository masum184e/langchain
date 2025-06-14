# Retrievers

A Retriever is an interface in LangChain that, given a query, returns a list of relevant documents. These documents are usually retrieved from:

- A Vector Store (e.g., Chroma, FAISS, Pinecone, etc.)
- A document store (like Elasticsearch)
- A custom source (e.g., files, APIs)

Retrievers abstract away the underlying search mechanism and provide a consistent interface for querying.

It is used to fetch relevant documents or information from a knowledge base (like a vector store) in response to a query. Retrievers are typically used before passing the result to an LLM to generate answers grounded in external data.

## Why use a Retriever?

- LLMs don’t have up-to-date knowledge or access to private/corporate data.
- Retrievers allow combining static LLMs with dynamic external knowledge.
- Great for chatbots, Q&A systems, agents, and search tools.

## Common Retriever Types

| Retriever Type                   | Description                                                                 |
| -------------------------------- | --------------------------------------------------------------------------- |
| `VectorStoreRetriever`           | Retrieves documents based on vector similarity search                       |
| `MultiQueryRetriever`            | Uses multiple reformulations of a query to retrieve more comprehensive data |
| `ContextualCompressionRetriever` | Compresses retrieved documents to only return relevant content              |
| `TimeWeightedVectorRetriever`    | Prioritizes documents based on recency and relevance                        |
| `SelfQueryRetriever`             | Uses an LLM to query metadata filters in a vector store                     |

## Retriever Flow

1. Embed the documents and store in a vector store
2. Create a Retriever from the vector store
3. Use the Retriever to fetch documents for a user query
4. (Optionally) Feed retrieved docs into an LLM for final answer

## Example of Retrievers

```py
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.get_relevant_documents(query)
```

## Ensemble

This is particularly useful when you have multiple retrievers that are good at finding different types of relevant documents.

```py
# Initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_store_retriever], weights=[0.5, 0.5]
)
```

## Differences with `similarity_search()`

| Feature                       | `similarity_search()`               | `as_retriever().get_relevant_documents()`                    |
| ----------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| Type                          | Direct method on vector store       | Abstracted retriever interface                               |
| Return Type                   | List of `Documents`                 | List of `Documents`                                          |
| Use Case                      | Simple or one-off similarity search | Plug into chains, agents, retriever pipelines                |
| Extensible                    | ❌ Not easily extensible            | ✅ Supports advanced retrievers (e.g., filters, compression) |
| Integrates with `RetrievalQA` | ❌ Not directly                     | ✅ Yes                                                       |
| Preferred for RAG             | ⚠️ Okay for quick tests             | ✅ Recommended                                               |

- Use `similarity_search()` for quick demos or if you don’t need chains.
- Use `as_retriever().get_relevant_documents()` if you’re building serious apps, using agents, chains, or want extensibility.

## Retrieval

Retrieval is the process of fetching relevant data in response to a query.
It uses a retriever behind the scenes.

So:

- `retrieval` = action
- `retriever` = tool that performs the action

In LangChain, when you use `retriever.get_relevant_documents()`, you are performing retriever

# Retrieval augmented generation (RAG)

Retrieval-Augmented Generation (RAG) is a technique that combines retrieval of relevant data from external sources (like vector databases) with generative models (LLMs) to answer questions or generate content more accurately and factually.

```
RAG = Retriever (retrieval) + LLM (generation)
```

## How RAG Works

LangChain provides a modular and easy way to implement RAG using the following components:

1. **Document Loaders** – Load unstructured data (e.g., PDFs, websites, markdowns).
2. **Text Splitters** – Break large text into chunks.
3. **Embeddings** – Convert chunks to vectors.
4. **Vector Store (e.g., Chroma, FAISS)** – Store the vectors for retrieval.
5. **Retriever** – Fetch relevant chunks from the vector store.
6. **LLM (e.g., OpenAI, Gemini)** – Generate responses using both the user query and retrieved documents.
7. **RAG Chain** – The final LangChain pipeline that brings it all together.

## Example

**RAG pipeline**

```py
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Other options: "map_reduce", "refine"
    retriever=retriever,
    return_source_documents=True,
)
```

**Ask Question:**

```py
query = "What are the core benefits of using RAG architecture?"
result = rag_chain.invoke({"query": query})

print("Answer:", result['result'])
print("Source Docs:", result['source_documents'])
```
