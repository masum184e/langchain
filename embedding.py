from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

split_docs = [
    Document(
        metadata={'source': 'T:/project/programming_notes/ai/pandas.md'},
        page_content='Pandas is used for working with data sets, it is used to analyze data. It has functions for analyzing, cleaning, exploring, and manipulating data.'),
    Document(
        metadata={'source': 'T:/project/programming_notes/ai/pandas.md'}, 
        page_content='__What Can Pandas Do?__\nPandas gives you answers about the data. Like:\n- Is there a correlation between two or more columns?\n- What is average value?\n- Max value?\n- Min value?')
    ]

embedding_model = OpenAIEmbeddings()

vector = embedding_model.embed_query(text)

print("Vector length:", len(vector))
print("Vector (first 10 values):", vector[:10])  # print first 10 for brevity