import streamlit as st
import os
import re
from crawl import get_pdf_text
from vectordatabase import vectorlocal_db
import chroma_db
from io import StringIO
import io
import ollama
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

## iniialized
temp_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = r"C:\Users\PC\Desktop\program\crawler\.crawler\file"
working_dir = r"C:\Users\PC\Desktop\program\crawler\.crawler\file"
#LLM Ollama 3.2 model
local_model = "llama3.2"
llm = ChatOllama(model = local_model)
embeddings = OllamaEmbeddings(
            model = "nomic-embed-text"
        )
persist_directory = r"C:\Users\PC\Desktop\program\crawler\.crawler\chroma_db"
collection_name="testing_database"

     
vector_store = Chroma(
    collection_name = collection_name,
    embedding_function = embeddings,
    persist_directory = persist_directory
)
###initialize vectorlocal class###
vectorlocal = vectorlocal_db(embeddings=embeddings, persist_dir=persist_directory, collection_name=collection_name, vector_store=vector_store)

###instruction###
QUERY_PROMPT = PromptTemplate(
    input_variables = ["question"],
    template = """You are an AI Language Model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspective on the user question, your goal
    is to help the user overcome some of the limitation of the distance-based similarity search.
    Provide these alternative questions separated by newlines.
    Original Question: {question}""",
)

#RAG project
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template=template)
output_from_llm = vectorlocal.query_llm(llm=llm)
chain = (
            {"context": output_from_llm, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

def main():
    #Header
    st.header("Chat with PDF using GEMINI")
    message = st.container(height=600)
   #sidebar for document upload
    with st.sidebar:
        st.title("good night, :sleeping:")
        pdffile_uploads = st.file_uploader("Upload PDF File", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & process"):
            with st.spinner("Processing..."):
                byte_data = pdffile_uploads[0].read()
                filename = pdffile_uploads[0].name
                with open(os.path.join(working_dir,filename),"wb") as f:
                     text = f.write(byte_data) 
                text_chunks = get_pdf_text(file_dir)
                vectordb = vectorlocal.add_document_to_vector(text_chunks)   
                st.success("Done")
     #input prompt
    if input := st.chat_input("Enter text"):
            message.chat_message("user").write(input)
            print(input)
            llm_output = chain.invoke(input)
            #text_llm_output = print(llm_output)
            message.chat_message("Ms.Ollama").write(f"{llm_output}")

if __name__ == "__main__":
    main()