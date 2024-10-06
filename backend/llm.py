from pydantic import BaseModel, Field, ValidationError, confloat
from enum import Enum
from typing import Dict, Literal, Union, List
import datetime
import os
from initial_state import new_recommender
from llm import ChatMistralAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate


class SamplerTypeEnum(str, Enum):
    GNEWS = 'GNEWS'
    XKCD = 'XKCD'
    ARXIV = 'ARXIV'


class Params(BaseModel):
    query: Dict[str, Union[str, int, datetime.time]]

class ArmModel(BaseModel):
    name: str
    # Params for GNEWS, but could be an empty dict for others like XKCD
    params: Union[Params, Dict]
    sampler_type: SamplerTypeEnum
    score: float = Field(ge=1.0, le=10.0)

class ArmsConfig(BaseModel):
    arms: List[ArmModel]

# Function to update the scores using the LLM


def update_arm_scores(llm, arms_config: dict):
    # Define the Pydantic parser
    parser = PydanticOutputParser(pydantic_object=ArmsConfig)

    # Create the prompt template
    prompt = PromptTemplate(
        template="""Here is the configuration of arms in JSON format:

{topics}
Update the arms based on the query:
{query}

{format_instructions}
""",
        input_variables=["topics", "query"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()},
    )

    # Create the chain (Prompt | LLM | Parser)
    chain = prompt | llm | parser

    # Invoke the LLM with the arm configuration as input
    return chain.invoke({"topics": arms_config, "query": "A user has expressed interest in news about the football world cup and the NBA basketball and some NBA teams like the Lakers."})


# HOW TO USE:
# updated_arms = update_arm_scores(llm, original_arms)
# rec.update_arms(updated_arms.dict())