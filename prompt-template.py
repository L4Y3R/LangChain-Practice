from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from langchain_fireworks import ChatFireworks
from langchain_core.prompts import ChatPromptTemplate

chat = ChatFireworks(
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    temperature=2
)

# set a template
template = ChatPromptTemplate.from_template("Tell me a dad joke about {subject}")

template_2 = ChatPromptTemplate.from_messages([
    ("system", "You are an AI chef, genrate me a super tasty recipy using this ingrediant"),
    ("human","{input}")
])

#chains
chain_template = template | chat
chain_message = template_2 | chat

#response_template =chain_template.invoke({"subject":"chicken"})
#print(response_template.content)

response_message =chain_message.stream({"input":"dog"})
for chunks in response_message:
    print(chunks.content, end="", flush=True)