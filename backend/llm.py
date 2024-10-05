from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, ValidationError
from langchain_core.prompts import PromptTemplate
from typing import List, Literal, Optional
from datetime import time
from langchain.output_parsers import PydanticOutputParser

class Topics(BaseModel):
    topics: List[str]

class UserResponse(BaseModel):
    topic: str
    action: Literal['add', 'delete',"-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
    memory: str
    time: Optional[time]
    reasoning: str

def generate_topics(llm, user_input: str) -> List[str]:
    parser = PydanticOutputParser(pydantic_object=Topics)
    prompt = PromptTemplate(
        template="Generate a list of topics of information that the user would like to know based on the following user input. Generate around 10 topics based on the user input. : {format_instructions} \n{user_input}",
        input_variables=["user_input"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | llm | parser
    return chain.invoke({"user_input": user_input}).topics

def classify_user_response(llm, topics, query):
    parser = PydanticOutputParser(pydantic_object=UserResponse)
    prompt = PromptTemplate(
        template="""You are a helpful assistant that decides whether you like the given topic, dislike it, want to add a new topic or delete an existing topic. 
        If the user likes or dislikes, give a score of 0 to 10. Respond with 'add', or 'delete' or scores from 0 to 10. 
        If the user wants to add or update the schedule, respond with 'add' or 'or suggest a time to post it. Fitness is a good topic early in the morning. Breaking news is best early mornings. Programming and transport is best in the evening. Use numerical time like 12:00.
        For memory, keep it in natural language. Summarize the interaction with the user. Be qualitative and include non numerical facts. Dt no list out the topics in the memory.
        For reasoning, explain your reasoning for the score or action. Use evidence based reasoning.
        Current topics: {topics}
        {format_instructions}
        {query}""",
        input_variables=["query", "topics"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser

    return chain.invoke({"query": query, "topics": topics})

