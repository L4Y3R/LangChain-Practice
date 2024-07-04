from dotenv import load_dotenv
load_dotenv()

import colorama
colorama.init()
from colorama import Fore

from langchain_fireworks import ChatFireworks
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_fireworks import FireworksEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

def doc_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def create_db(docs):
    embedding = FireworksEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    llm = ChatFireworks(
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        temperature=0.4
    )

    template = ChatPromptTemplate.from_messages([
        ("system", "Answer the following questions asked by the user based on the provided context (don't reveal to the user you have a context file): {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=template
    )

    history_aware_prompt = ChatPromptTemplate.from_messages([
         MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the information, generate a search querry to look up the relevant information")
    ])

    retriver = vectorStore.as_retriever(search_kwargs={"k": 3})
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriver,
        prompt=history_aware_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )

    return retrieval_chain

def chat(chain, question, chat_history):
    response= chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    return response["answer"]

if __name__ == "__main__":  
    docs = doc_from_web('https://www.pinecone.io/learn/series/langchain/langchain-expression-language/')
    vectorStore = create_db(docs)
    chain = create_chain(vectorStore)

    chat_history = []

    while True:
        human = input (Fore.CYAN + "\n You: ")
        if human.lower() == "exit":
            break
        response = chat(chain, human, chat_history)
        chat_history.append(HumanMessage(content=human))
        chat_history.append(AIMessage(content=response))
        print(Fore.WHITE + "\n AI: ", response)
        print(Fore.WHITE + "_________________________________________________________________________________________")