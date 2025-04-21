## Chat with Your Data - RAG-based Question Answering


This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that allows users to ask questions based on documents (PDFs).

------------------------------

## 📁 Project Structure
```
chat_with_data
  ├── data/ # Folder to store input PDF files 
  ├── loader.py # Document loader and pre-processing (PDF, cleaning, etc.) 
  ├── vectorstore.py # Vector store creation & retrieval using FAISS 
  ├── model.py # Load LLM model (supports quantized models) 
  ├── rag.py # Builds and runs the RAG chain 
  ├── main.py # Entry point to ask questions to the RAG system 
  └── notebook/ └── run.ipynb # Jupyter notebook for testing the pipeline interactively
```
## 🔧 Test Instructions

 1. Clone the repository

```bash
git clone https://github.com/Kiem-cmd/chat-with-data.git
cd chat_with_data
```
2. Run run.ipynb in folder _notebook_
