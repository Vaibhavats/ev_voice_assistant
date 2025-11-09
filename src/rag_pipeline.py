# ----------------------------------------------------
# 🚗 EV Voice Assistant — Modern RAG Pipeline (LangChain v1.0.5+)
# ----------------------------------------------------

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import pandas as pd
import os

# ----------------------------------------------------
# 1️⃣ Load and Prepare Dataset
# ----------------------------------------------------
data_path = "/Users/vaibhavkumar/Downloads/cleaned_ev_specs.csv"
df = pd.read_csv(data_path)

docs = []
for _, row in df.iterrows():
    text = f"""
    Brand: {row.get('brand')}
    Model: {row.get('model')}
    Range: {row.get('range_km')} km
    Battery Capacity: {row.get('battery_capacity_kWh')} kWh
    Top Speed: {row.get('top_speed_kmh')} km/h
    Body Type: {row.get('car_body_type')}
    Segment: {row.get('segment')}
    """
    docs.append(Document(page_content=text))

# ----------------------------------------------------
# 2️⃣ Split Text and Create Vector Store
# ----------------------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Local (free) Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create Chroma vector database
vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ----------------------------------------------------
# 3️⃣ LLM Setup
# ----------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# ----------------------------------------------------
# 4️⃣ Build RAG Chain (Compatible with LangChain v1.0+)
# ----------------------------------------------------
prompt = ChatPromptTemplate.from_template("""
You are an intelligent EV assistant.
Use the following vehicle data to answer clearly and concisely.

Context:
{context}

Question:
{input}
""")

# Equivalent to create_stuff_documents_chain()
document_chain = prompt | llm | StrOutputParser()

# ✅ Correct mapping so retriever gets a plain text query
rag_chain = (
    {
        "context": lambda x: retriever.invoke(x["input"]),
        "input": RunnablePassthrough()
    }
    | document_chain
)

# ----------------------------------------------------
# 5️⃣ Run a Query
# ----------------------------------------------------
query = "Which EV has the longest range and good top speed?"

# Preview top 3 retrieved EVs
retrieved_docs = retriever.invoke(query)
print("\n📄 Top 3 Retrieved EVs:\n")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"{i}. {doc.page_content.strip()}\n{'-'*60}")

# Get the final LLM-generated answer
response = rag_chain.invoke({"input": query})

print("\n🔍 Query:", query)
print("\n💬 Answer:", response)

