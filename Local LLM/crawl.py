import requests
import json
from PyPDF2 import PdfReader
import ollama
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
import os
from uuid import uuid4
#import streamlit as st

##upload PDF
path = r"C:\Users\PC\Desktop\program\crawler\.crawler\file"
def get_pdf_text(path):
    document = []
    doc_ids = []
    text = []
    summary_text = []
    #id_key = "doc_id"

    dir_list = os.listdir(path)
    #for i in range(len(dir_list)):
        #pdf.append(dir_list[i])
    for j in dir_list:   
        doc = PdfReader(f"{path}/{j}")
        for pages in range(len(doc.pages)):
            text.append(doc.pages[pages].extract_text())
        #text_chunks = get_text_chunks(text)
        #doc_ids=[str(uuid4()) for _ in text_chunks]

        summary_text = [
            Document(
                page_content=summary,
                metadata={"source": j},
                          
            ) for i, summary in enumerate(text)
        ]
        text.clear()
        document.extend(summary_text)
    return document

def get_pdf(pdf_doc):
    text = []
    document = []
    doc = PdfReader(pdf_doc)
    for pages in range(len(doc.pages)):
        text.append(doc.pages[pages].extract_text())
    return text

def get_text_chunks(text):
    textsplitter1 = CharacterTextSplitter(
        separator= '\n',
        chunk_size = 500,
        chunk_overlap = 100,
        length_function = len
    )
    chunks = textsplitter1.split_documents(text)
    return chunks


#splitting and chunking the pdf
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=5000)
def get_vectors(text_chunks):
    #add to vector database
    embeddings = OllamaEmbeddings(
            model = "nomic-embed-text"
        )
    vectordb = Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory=r"C:\Users\PC\Desktop\program\crawler\.crawler\chroma_db")
    return vectordb
    
def retrieve_vectors():
    local_model = "llama3.2"
    llm = ChatOllama(model = local_model)

    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=get_vectors().as_retriever(search_type = "mmr"), llm=llm)
    return retriever_from_llm
embeddings = OllamaEmbeddings(
            model = "nomic-embed-text"
        )

# get_text = get_pdf_text(path)
# text_chunks = get_text_chunks(get_text)
# print(type(get_text))
# # vectordb = get_vectors(text_chunks)

vectordb = Chroma(persist_directory = r"C:\Users\PC\Desktop\program\crawler\.crawler\chroma_db", 
                                 embedding_function=embeddings)

#print(vectordb.get())

#docs = vectordb.similarity_search("BubbleML")
#print(docs)

#LLM Ollama 3.2 model
local_model = "llama3.2"
llm = ChatOllama(model = local_model)

#retriever_from_llm = MultiQueryRetriever.from_llm(
        #retriever=vectordb.as_retriever(), llm=llm)
# instruction
# QUERY_PROMPT = PromptTemplate(
#     input_variables = ["question"],
#     template = """You are an AI Language Model assistant. Your task is to generate five
#     different versions of the given user question to retrieve relevant documents from
#     a vector database. By generating multiple perspective on the user question, your goal
#     is to help the user overcome some of the limitation of the distance-based similarity search.
#     Provide these alternative questions separated by newlines.
#     Original Question: {question}""",
# )

# #RAG project
# template = """Answer the question based ONLY on the following context:
#  {context}
# Question: {question}
#  """
# while True:
#     prompt = ChatPromptTemplate.from_template(template=template)

#     chain = (
#         {"context": retrieve_vectors(), "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     print(chain.invoke(input("")))

