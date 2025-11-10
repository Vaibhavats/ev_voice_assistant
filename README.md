# ⚡ EV Voice Assistant (Generative AI for Electric Vehicles)

## 🚀 Project Overview
The **EV Voice Assistant** is an intelligent generative AI system designed to enhance the **in-vehicle experience** for Electric Vehicle (EV) users.  
It enables natural voice interaction to answer queries related to **vehicle specifications, charging options, and route optimization** — powered by **Generative AI + RAG (Retrieval-Augmented Generation)**.

This project demonstrates how AI can be embedded into EV systems to create smarter, conversational interfaces that improve driver safety, convenience, and efficiency.

---

## 🎯 Project Objectives
- Develop a **Generative Voice Assistant** prototype for EVs.  
- Enable **natural-language querying** for EV specifications, charging, and performance data.  
- Integrate a **Retrieval-Augmented Generation (RAG)** pipeline for intelligent reasoning using real EV datasets.  
- Prototype a **voice-based interaction layer** (speech-to-text and text-to-speech).

---

## 🧠 Core Features (Planned)
1. **EV Data Understanding** — AI assistant can retrieve detailed information about any EV (e.g., range, top speed, battery capacity).  
2. **Smart Charging Queries** — Find charging stations or estimate charging time.  
3. **Natural Language Reasoning** — Complex queries like _“Compare Tesla Model 3 and BYD Seal on battery performance.”_  
4. **Voice Interface** — Speech recognition and TTS output for hands-free operation.

---

## 🗓️ Week 1 Progress — _Data Preparation & Setup_

### ✅ Key Tasks Completed
1. **Project Setup**
   - Initialized virtual environment and VS Code workspace.
   - Created clean modular structure: `data/`, `src/`, `notebooks/`, `prompts/`.
   - Added `.gitignore`, `requirements.txt`, and `README.md`.

2. **Data Cleaning**
   - Loaded and cleaned raw EV dataset.
   - Removed duplicates and handled missing values.
   - Standardized numeric columns (e.g., `range_km`, `battery_capacity_kWh`).
   - Saved clean dataset as `cleaned_ev_specs.csv`.

3. **Text Chunk Generation**
   - Converted structured EV data into human-readable text chunks.
   - Saved as `ev_text_chunks.csv` for use in RAG pipeline.

4. **Prompt Engineering**
   - Designed initial LLM prompt templates for EV-related queries.
   - Tested query response logic in Jupyter notebooks.

5. **Environment Setup**
   - Installed and configured essential libraries:  
     `openai`, `langchain`, `chromadb`, `pandas`, `gtts`, `speechrecognition`.

---

## 💻 Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python 3.11+ |
| **AI Model** | OpenAI GPT / Llama 3 |
| **Framework** | LangChain / LlamaIndex |
| **Vector Database** | ChromaDB |
| **Speech** | SpeechRecognition, gTTS |
| **Development** | VS Code, Jupyter Notebook |
| **Version Control** | Git + GitHub |

---

🧠 Week 2 Progress — RAG Pipeline Development
🚀 Key Improvisations

Implemented RAG architecture using LangChain and ChromaDB.

Integrated HuggingFace Embeddings (MiniLM model) for semantic text representation.

Built Conversational Retrieval Chain for context-aware question answering.

Optimized Text Chunking: Improved context retention using RecursiveCharacterTextSplitter.

Connected RAG with OpenAI GPT models for accurate and contextual responses.

Tested complete query flow — from user query → retrieval → LLM response.

Improved data pipeline efficiency and ensured reproducibility.

✅ Outcome

By the end of Week 2, a functional RAG-based EV assistant prototype was achieved that can understand and respond to EV-related queries using the structured knowledge base.

💻 Tech Stack
Category	Tools / Libraries
Language	Python 3.11+
AI Models	OpenAI GPT / Llama 3
Frameworks	LangChain / LlamaIndex
Vector Database	ChromaDB
Speech	SpeechRecognition, gTTS
Development	VS Code, Jupyter Notebook
Version Control	Git + GitHub

🔮 Week 3 Goals — Voice Interface & Integration

Integrate voice input/output with the RAG pipeline.

Add speech-to-text and text-to-speech capabilities.

Enhance conversation memory and contextual understanding.

Build a minimal GUI or terminal interface for user interaction.

Begin preparing for deployment (local or web app).
## 📁 Project Structure

EV_VOICE_ASSISTANT/
│
├── data/
│ ├── cleaned_ev_specs.csv
│ └── ev_text_chunks.csv
│
├── notebooks/
│ ├── data_cleaning.ipynb
│ └── create_ev_text_chunks.ipynb
│
├── src/
│ ├── main.py
│ ├── rag_pipeline.py
│ ├── utils.py
│ └── voice_interface.py
│
├── prompts/
│ └── basic/
│
├── requirements.txt
└── README.md



