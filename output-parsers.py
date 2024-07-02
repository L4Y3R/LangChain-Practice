from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from langchain_fireworks import ChatFireworks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

chat = ChatFireworks(
    model = "accounts/fireworks/models/mixtral-8x7b-instruct",
    temperature=0.1,
)

def string_output_parser():
    template = ChatPromptTemplate.from_messages([
        ("system", "You are Gordon Ramsey and you are very mad right now, review these food items for me"),
        ("human", "{food}")
    ])

    parser = StrOutputParser()

    chain = template | chat | parser

    return chain.invoke({"food":"Dominos Pizza"})

#print(string_output_parser())

def list_output_parser():
    template = ChatPromptTemplate.from_messages([
        ("system", "Give me a comma seperated python list of 10 food items related to this category."),
        ("human", "{fooditem}")
    ])

    parser =  CommaSeparatedListOutputParser()

    chain = template | chat | parser

    return chain.invoke({"fooditem":"junk food"})

#print(list_output_parser())


def json_output_parser():
    template = ChatPromptTemplate.from_messages([
        ("system", "Give me a json object from the following descriptin.\n Fomating instrusctions: {format}"),
        ("human", "{text}")
    ])

    class Output(BaseModel):
        name: str = Field(description="The name of the food item")
        ingediants: str = Field(description="The number of ingrediants mentioned")

    parser =  JsonOutputParser(pydantic_object=Output)

    chain = template | chat | parser

    return chain.invoke({
        "text":string_output_parser,
        "format": parser.get_format_instructions()
    })

print(type(json_output_parser()))