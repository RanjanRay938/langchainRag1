"""
LangChain Document Loaders – Single File Demo
Author: Ranjan Ray

This script demonstrates how to load documents using LangChain Community
from different sources:
1. Text file
2. PDF file
3. Web page
4. Wikipedia

Run this file step by step or as a whole after installing dependencies.
"""

# =========================
# REQUIRED INSTALLATIONS
# =========================
# Run these once in terminal or Jupyter:
# pip install langchain-community
# pip install pypdf
# pip install bs4
# pip install wikipedia

# =========================
# IMPORTS
# =========================
import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    WikipediaLoader
)

# Optional: avoid USER_AGENT warning
os.environ["USER_AGENT"] = "Mozilla/5.0"

# =========================
# 1. TEXT FILE LOADER
# =========================
print("\n--- TEXT FILE LOADER ---")
text_loader = TextLoader("speech.txt")
text_docs = text_loader.load()

print("Text file content (first 300 chars):")
print(text_docs[0].page_content[:300])

# =========================
# 2. PDF FILE LOADER
# =========================
print("\n--- PDF FILE LOADER ---")
pdf_loader = PyPDFLoader("attention.pdf")
pdf_docs = pdf_loader.load()

print(f"Total PDF pages loaded: {len(pdf_docs)}")
print("PDF Page 1 content (first 300 chars):")
print(pdf_docs[0].page_content[:300])

# =========================
# 3. WEB PAGE LOADER
# =========================
print("\n--- WEB PAGE LOADER ---")
web_loader = WebBaseLoader(
    web_paths=("https://www.w3schools.com/html/html_links.asp",)
)

web_docs = web_loader.load()

print("Web page content (first 300 chars):")
print(web_docs[0].page_content[:300])

# =========================
# 4. WIKIPEDIA LOADER
# =========================
print("\n--- WIKIPEDIA LOADER ---")
wiki_loader = WikipediaLoader(
    query="Generative artificial intelligence",
    load_max_docs=2
)

wiki_docs = wiki_loader.load()

print(f"Total Wikipedia docs loaded: {len(wiki_docs)}")
print("Wikipedia article content (first 300 chars):")
print(wiki_docs[0].page_content[:300])

# =========================
# DONE
# =========================
print("\n✅ All document loaders executed successfully.")
print("You now have data ready for text splitting, embeddings, and RAG.")
