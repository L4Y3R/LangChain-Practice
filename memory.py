from dotenv import load_dotenv
load_dotenv()

from langchain_fireworks import ChatFireworks
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

llm = ChatFireworks(
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    temperature=0.9
)

template = ChatPromptTemplate.from_messages([
    ("system", "You are a funny AI assistant with a 100 percent comedy setting"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

chain = memory | template | llm

msg = {
    "input": "What is the meaning of life?"
}

response = chain.invoke(msg)
print(response)