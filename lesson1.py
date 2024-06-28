from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from langchain_fireworks import ChatFireworks

llm=ChatFireworks(
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    temperature=0
)

response_normal=llm.invoke("Hello. Tell me abot fireworks") 
print(response_normal)

response_batch=llm.batch(["Hello", "Tell me abot fireworks"]) # these run in parallel
print(response_batch)

response=llm.stream("Hello. Tell me abot fireworks") # return response in chunks

for chunks in response:
    print(chunks.content, end="", flush=True)
