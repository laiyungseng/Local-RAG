from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
import chromadb
from uuid import uuid4
from langchain_core.documents import Document
import os
import shutil
from crawl import get_pdf_text
from crawl import get_text_chunks
from langchain.retrievers.multi_query import MultiQueryRetriever

embeddings = OllamaEmbeddings(model="nomic-embed-text")
collection_name="testing_database"
persist_directory = r"C:\Users\PC\Desktop\program\crawler\.crawler\chroma_db"
doc = get_pdf_text(r"C:\Users\PC\Desktop\program\crawler\.crawler\file")

vector_store = Chroma(
    collection_name = collection_name,
    embedding_function = embeddings,
    persist_directory = persist_directory
)

persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection(collection_name)

vector_store_from_client = Chroma(
    client=persistent_client,
    collection_name="testing_database",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
)

class vectorlocal_db:
    def __init__(self, embeddings,persist_dir, collection_name, vector_store):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        self.vector_store = Chroma(
            collection_name = self.collection_name,
            embedding_function = self.embeddings,
            persist_directory = self.persist_dir
        )
        self.persistent_client = chromadb.PersistentClient()
        self.collection = self.persistent_client.get_or_create_collection(self.collection_name)

    #add pdf to vector database
    def add_vector_db(self,doc):
        idk = [str(x) for x in range(len(doc))]
        stored_data = self.vector_store.add_documents(documents=doc, ids=idk)
        return stored_data
    
    def add_document_to_vector(self, doc):
        uuid = [str(uuid4()) for _ in range(len(doc))]
        stored_data = self.vector_store.from_documents(
            documents= doc,
            embedding=self.embeddings,
            ids = uuid,
            persist_directory = self.persist_dir,
        )
        return stored_data
    
    ## delete whole document in the cvectorstore ##
    ## delete_collection and delete chroma vectorstore is different database##
    def delete_collection(self,collection_name):
        self.vector_store._client.delete_collection(collection_name)
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir, ignore_errors=True)
        print(f"collection: {self.collection_name} deleted...")

    ## delete documents ##
    def delete_document(self, x):
        all_docs = self.vector_store.get()["ids"][x]
        self.vector_store.delete(ids=all_docs)
        print(f"Done delete document{all_docs}")

    ## delete chroma vectorstore ##
    def delete_chroma_vectorstore(self):
        self.vector_store.delete_collection()

        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function = self.embeddings,
            collection_name = self.collection_name
        )
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir, ignore_errors=True)
            if self.vector_store.get()["ids"] == []:
                print(f"Done delete document in collection {self.collection_name}")
            else:
                print(f"Fail to delete document in collection {self.collection_name}")

    def delete_chroma_vectorstore_id(self, id):
        self.vector_store.delete(id)

    def query_from_vector(self, query_input):
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        results  = vector_store.as_retriever(
            search_type="mmr", search_kwargs ={"k":1, "fetch_k":5}
        )
        result = results.invoke(query_input)
        return result
    def query_llm(self, llm):
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(), llm=llm)
        return retriever_from_llm
#////////////////////////////////////////////////////////////////////////#
# document_1 = Document(
#     page_content = "I have a pen and an apple, what do you have?",
#     metadata={"source": "tweet",
#               "language":"EN",
#               "Author":"Lai"},
#     id = 1
# )

# document_2 = Document(
#     page_content = "Trumpt annoucced bitcoin will be official payment method.",
#     metadata={"source":"news",
#               "language":"EN",
#               "Author":"Lee"},
#     id = 2
# )
# document_3 = Document(
#     page_content = "I am rich, i have multiple different sport car in my garage",
#     meta={"source": "tweet",
#     "language": "EN",
#     "Author": "Ang"},
#     id =3
# )

# documents = [document_1 , document_2 , document_3]
# uuids = []
# stored_data = []

## add random text into vectorstore
def add_vector_db(documents):
    uuids = [str(uuid4()) for _ in range(len(documents))]
    stored_data = vector_store.from_documents(
            documents= documents,
            embedding=embeddings,
            ids = uuids,
            persist_directory = persist_directory,
        )
    return stored_data, uuids
def add_vector_db1(documents):
    uuids = [str(uuid4()) for _ in range(len(documents))]
    stored_data = vector_store.add_documents(documents=documents, ids=uuids)
    return stored_data, uuids
### add pdf into vectordatabase
def add_vector_db2(doc):
    idk = [str(x) for x in range(len(doc))]
    stored_data = vector_store.add_documents(documents=doc, ids=idk)
    return stored_data

# ##count data in chrome database: before
# #print(vector_store._collection.count())

#stored_data, uuids = add_vector_db(documents=doc)

# #print(stored_data)
#add_vector_db2(doc=doc)

# delete collection
def delete_collection(collection_name):
    vector_store._client.delete_collection(collection_name)
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)
        if vector_store.get()["ids"] == []:
            print("delete collection successful")
        else:
            print("fail to delete collection")
def delete_chroma_vectorstore():
    #init vector store
    vector_store = Chroma(
        collection_name = collection_name,
        embedding_function = embeddings,
        persist_directory = persist_directory
    )
    #delete collection in chroma vectorstore
    vector_store.delete_collection()
    ### reset collection after deleted document in chroma vector store
    vector_store = Chroma(
        collection_name = collection_name,
        embedding_function = embeddings,
        persist_directory = persist_directory
    )
    #delete file in chromadb folder
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)
        if vector_store.get()["ids"] == []:
            print("delete collection successful")
        else:
            print("fail to delete collection")
# ## delete documents
# def delete_document(ids):
#     vector_store.delete(ids=ids)
#     print(f"Done delete document{ids}")

#delete_collection(collection_name=collection_name)
# delete_document(ids = x)
#delete_chroma_vectorstore()

#print(f"print count after delete {vector_store._collection.count()}")
# #count data in chrome database: after
# #print(vector_store._collection.count()) 

#vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# results  = vector_store.as_retriever(
#     search_type="mmr", search_kwargs ={"k":1, "fetch_k":5}
# )
# print(results.invoke("apple"))
# # print(f"search for similar text: ",vector_store.similarity_search(query="apple",k=1))

# g = vector_store.get()
# print(f"list of document: {g}")

# ###init vectordb module###
#vecto_db = vectorlocal_db(embeddings=embeddings,persist_dir=persist_directory, collection_name=collection_name, vector_store=vector_store)
#data = vecto_db.add_document_to_vector(doc)
# ###delete function###
# vecto_db.delete_chroma_vectorstore()
# vecto_db.delete_collection(collection_name)

# import ollama
# from langchain_ollama import OllamaLLM, ChatOllama
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain.prompts import ChatPromptTemplate, PromptTemplate
#vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# results  = vector_store.as_retriever(
#     search_type="mmr", search_kwargs ={"k":1, "fetch_k":5}
# )
# print(results.invoke("reinforment learning"))
# g = vector_store.get()
# print(f"deleted document:{g}")

# ## query function###
# llm = OllamaLLM(model="llama3.2")


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
# {context}
# Question: {question}
# """

# prompt = ChatPromptTemplate.from_template(template=template)
# output_from_llm = vecto_db.query_llm(llm=llm)

# chain = (
# {"context": output_from_llm, "question": RunnablePassthrough()}
# | prompt
# | llm
# | StrOutputParser()
# )
# x = input("")
# llm_output = chain.invoke(x)
# #print(llm_output)
# text_llm_output = print(llm_output)





# update_document_1 = Document(
#     page_content = "I had chocolate chip pancakes and fried eggs for breakfast this morning.",
#     metadata = {"source": "tweet"},
#     id=1,
# )

# #vector_store.update_document(document_id=x, document=update_document_1)
# #print("done update existing document")

# #print(results.invoke("chocolate"))

# #get stored data from database
# #all_docs = vector_store.get()
# #print(all_docs["ids"][0])
