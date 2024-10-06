from pydantic import BaseModel, Field, ValidationError, confloat
from enum import Enum
from typing import Dict, Literal, Union, List
import datetime
import os
from backend.initial_state import new_recommender
from langchain_mistralai import ChatMistralAI
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


def update_arm_scores(llm, arms_config: dict, query: str):
    print(arms_config)
    print(query)
    # Define the Pydantic parser
    parser = PydanticOutputParser(pydantic_object=ArmsConfig)

    # Create the prompt template
    prompt = PromptTemplate(
        template="""###Scoring User Interest Based on Feedback and Content Samplers

You are an advanced AI tasked with helping the user navigate various types of content. Based on user interactions and feedback (such as upvotes, downvotes, clicks on articles, or direct conversations with you), your job is to assess the user's interest in different topics. You will assign or adjust a score on a scale from 1 to 10 for each content sampler based on this feedback, and if necessary, introduce new topics with an initial score based on the user's behavior. Here's how to approach the task:

### Scoring Scale Definition (values are between 1 and 10):
1. **1-2 (No Interest)**: The user has shown clear signs of disinterest. This could be indicated by consistently downvoting or skipping content related to the topic, or perhaps even directly expressing disinterest in conversations.  
2. **3-4 (Low Interest)**: The user occasionally engages with the topic but doesn't show strong enthusiasm. This could be indicated by clicks or rare positive feedback but no sustained engagement.
3. **5-6 (Mild Interest)**: The user demonstrates moderate interest, engaging with the content more frequently. This might involve clicking on articles, occasionally upvoting, or mentioning the topic without deep enthusiasm.
4. **7-8 (Moderate Interest)**: The user shows a clear, sustained interest. They consistently upvote or engage with content from this topic, have clicked on related articles several times, or have mentioned the topic with some excitement or curiosity in conversations.
5. **9-10 (High Interest)**: The user is very enthusiastic about the topic. This could be shown through frequent upvotes, continuous clicks on articles, initiating conversations about the topic, or repeatedly expressing a desire to see more content on the subject.

### Key Details to Consider About the User:
- **Recent Feedback**: Look at the last several pieces of feedback (upvotes, downvotes, clicks, chats) and weigh more recent interactions more heavily. If the user has changed their behavior over time (e.g., downvoted content they previously upvoted), adjust the scores accordingly.
- **Diverse Interests**: Be mindful that users may have a range of interests. If they upvote or click on a wide variety of topics, consider adjusting multiple samplers, even if each receives a moderate score.
- **Topic Overlap**: If a user frequently engages with topics that are related or adjacent (e.g., consistently clicking on political news articles), this could indicate higher interest across the broader subject area, not just a specific sub-topic.
- **Depth of Engagement**: If the user provides feedback in the form of deep engagement—such as explicitly requesting more content on a certain subject or repeatedly reading detailed articles—this points to high interest.
  
### Steps for Scoring:
1. **Examine Feedback**: Start by reviewing the recent feedback provided by the user. This includes upvotes, downvotes, clicks, and any relevant conversations.
2. **Assess Existing Topics**: For each content sampler, check whether the feedback aligns with previous patterns. For example:
   - If a sampler has been consistently upvoted and recently received another upvote, consider increasing the score.
   - If the user has been downvoting the topic or rarely engaging with it, consider lowering the score.
3. **Introduce New Topics**: If you identify a new area of interest (e.g., the user clicks on an article in a previously unexplored topic), create a new sampler for this topic. Assign an initial score based on how the user engaged:
   - **Mild Engagement** (one click or brief mention) → Start with a score of 5-6.
   - **Moderate Engagement** (multiple clicks or consistent mentions) → Start with a score of 7-8.
   - **High Engagement** (explicit requests, numerous clicks, or repeated strong mentions) → Start with a score of 9-10.
4. **Adjust Scores Based on Patterns**: Regularly reevaluate the scores based on changing user behavior. For example, if a user shifts from upvoting to downvoting a topic, lower the score accordingly. Conversely, if they suddenly start engaging more frequently, increase the score.

### Example:

User Feedback:
- Upvoted two political news articles recently.
- Downvoted an article about sports.
- Clicked on several tech-related articles over the past week.
- Mentioned an interest in environmental issues in a chat.
  
Based on this:
- **Politics Sampler**: Current score is 6 (mild interest). Based on recent upvotes, increase the score to 8 (moderate interest).
- **Sports Sampler**: Current score is 4 (low interest). Based on the downvote, lower the score to 2 (no interest).
- **Tech Sampler**: Current score is 5. Due to frequent clicks, raise the score to 7 (moderate interest).
- **Environmental Topics**: This is a new topic based on the chat. Add a new sampler for environmental issues with an initial score of 6 (mild interest), as the user has mentioned interest but hasn't yet engaged deeply with content.

By regularly reviewing this feedback and adjusting scores, you help provide more relevant and engaging content to the user.
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
    return chain.invoke({"topics": arms_config, "query": query})
# "A user has expressed interest in news about the football world cup and the NBA basketball and some NBA teams like the Lakers."


# HOW TO USE:
# updated_arms = update_arm_scores(llm, original_arms)
# rec.update_arms(updated_arms.dict())