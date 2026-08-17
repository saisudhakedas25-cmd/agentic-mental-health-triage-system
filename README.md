# agentic-mental-health-triage-system
An Agentic Retrieval-Augmented Mental Health Triage System for Explainable Depression Risk Assessment   
# Explainable Agentic AI Mental Health Triage System

An Agentic RAG-based NLP system for explainable depression-risk assessment from patient conversations.

## Overview

This project combines:

- RoBERTa → Multi-label depression symptom classification
- Risk Assessment → Calculates risk score and risk category
- SHAP → Explains symptom-level predictions
- NICE Guidelines → Clinical knowledge source
- Sentence Transformers + FAISS → Semantic guideline retrieval
- RAG + LLM → Generates grounded clinical explanations
- Multi-Agent Workflow→ Coordinates the different system components
- Gradio → Interactive user interface

##  System Pipeline


Patient Conversation -> Data Processing -> RoBERTa Symptom Classification -> Risk Assessment-> SHAP Explainability ->NICE Guidelines->Sentence Transformers + FAISS Retrieval->RAG + LLM
Multi-Agent Workflow -> Clinical Explanation / Summary




## Model Storage & Automated Setup

Because the fine-tuned RoBERTa model file is large (560 MB), it is not stored directly in this GitHub repository - beyond the limit of github. Instead, it is hosted securely on Google Drive and to run it 

you have to use following code --as all .pkl, .modeltensor files are present in google drive.




##  Core Initialization & Asset Pipeline

The code below shows how the system dynamically checks for local files from my google drive


import os
import pickle
import torch
import faiss
import gdown
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sentence_transformers import SentenceTransformer

# 1. Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on :", device)

# 2. Local File Configurations (Updated from Google Colab paths)
ROBERTA_MODEL_PATH = "./best_roberta_model"
MLB_PATH = "./mlb.pkl"
FAISS_INDEX_PATH = "./faiss.index"
CHUNKS_PATH = "./chunks.pkl"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Helper function 
def verify_and_download_assets():
    # Example mapping: local path to its Google Drive shared file link
    assets = {
        ROBERTA_MODEL_PATH: "https://google.com",
        MLB_PATH: "https://google.com",
        FAISS_INDEX_PATH: "https://google.com",
        CHUNKS_PATH: "https://google.com"
    }
    
    for path, url in assets.items():
        if not os.path.exists(path):
            print(f" Downloading missing asset: {path}...")
            gdown.download(url, path, quiet=False, fuzzy=True)
            print(f" {path} downloaded successfully.")

# Run asset verification before loading the models
verify_and_download_assets()

# 3. Load RoBERTa Tokenizer
print("\nLoading RoBERTa Tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_PATH)
print("Tokenizer Loaded.")

# 4. Load Trained RoBERTa Model
print("\nLoading Trained RoBERTa Model...")
model = RobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_PATH)
model.to(device)
model.eval()
print("RoBERTa Model Loaded Successfully.")

# 5. Load MultiLabelBinarizer
print("\nLoading MultiLabelBinarizer...")
with open(MLB_PATH, "rb") as f:
    mlb = pickle.load(f)
print("Labels Loaded:", mlb.classes_)

# 6. Load SentenceTransformer
print("\nLoading SentenceTransformer...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print("SentenceTransformer Ready.")

# 7. Load FAISS Index
print("\nLoading FAISS Index...")
index = faiss.read_index(FAISS_INDEX_PATH)
print("Vectors inside FAISS :", index.ntotal)

# 8. Load NICE Chunks
print("\nLoading NICE Guideline Chunks...")
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)
print("Total Chunks :", len(chunks))

print("\nSystem Ready.")

# Note:
1)  code works perfectly fine --- after the tab/cell : Loading all the pkl files and models from here for RAG without XAI--i.e. because after training completion, all the saved models are called here onwards
2)  End to End Coding pipeline works perfectly fine on my personal Laptop and all test cases are checked, if anything is needed , I am happy to resolve the issue.

