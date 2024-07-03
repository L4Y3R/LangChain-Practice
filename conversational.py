from dotenv import load_dotenv
load_dotenv()

from langchain_fireworks import ChatFireworks
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_fireworks import FireworksEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

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

    template = ChatPromptTemplate.from_template("""
            Answer the user's question:
            Context: {context}
            Qustion : {input}
                                            """)

    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=template
    )

    retriver = vectorStore.as_retriever(search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(
        retriver,
        chain
    )

    return retrieval_chain

def chat(chain, question):
    response= chain.invoke({
        "input": question
        })

    return response["answer"]

if __name__ == "__main__":  
    docs = doc_from_web('https://www.pinecone.io/learn/series/langchain/langchain-expression-language/')
    vectorStore = create_db(docs)
    chain = create_chain(vectorStore)

    human = input ("You: ")
    response = chat(chain, human)
    print("AI: ", response)