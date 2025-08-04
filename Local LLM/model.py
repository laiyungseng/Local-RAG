import ollama
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template("Provide me information of {topic}")
output_parser = StrOutputParser()
model = ChatOllama(model = "llama3.2")
llm = OllamaLLM(model="llama3.2")

chain = ({"topic": RunnablePassthrough()}
         |prompt 
         | llm 
         | output_parser)

print(chain.invoke(input("")))

#message = prompt.invoke({"topic": "samsung"})

#output_message=model.invoke(message)
#print("ChatOllama model:\n", output_message)



#llm_output = llm.invoke(message)
#print("OllamaLLM model:\n",llm_output)
