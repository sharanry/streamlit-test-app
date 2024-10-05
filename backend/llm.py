from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, ValidationError
from langchain_core.prompts import PromptTemplate
from typing import List, Literal, Optional
from datetime import time
from langchain.output_parsers import PydanticOutputParser

class Topics(BaseModel):
    topics: List[str]


class UserResponse(BaseModel):
    reasoning: str
    topic: str
    action: Literal['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    memory: str
    time: Optional[time]


def generate_topics(llm, user_input: str) -> List[str]:
    parser = PydanticOutputParser(pydantic_object=Topics)
    prompt = PromptTemplate(
        template="Generate a list of topics of information that the user would like to know based on the following user input. Generate around 10 topics based on the user input. : {format_instructions} \n{user_input}",
        input_variables=["user_input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | llm | parser
    return chain.invoke({"user_input": user_input}).topics


def classify_user_response(llm, topics, query):
    parser = PydanticOutputParser(pydantic_object=UserResponse)
    prompt = PromptTemplate(
        template="""Here’s a detailed and structured prompt that you can use to guide the LLM in making decisions about user interest based on their feedback and interactions. This approach includes explicit definitions of scores, considerations for context, and strategies for how the LLM should adapt its scoring and recommendations.

---

### Prompt for LLM: Scoring User Interest in Content

You are tasked with evaluating a user's interest in specific topics based on their interactions with content samplers, such as news articles, and their direct feedback through upvotes, downvotes, clicks, and conversations. Your goal is to dynamically adjust the scores for these topics on a scale from 1 to 10, where 1 indicates no interest and 10 indicates strong interest. In some cases, you may also need to introduce new topics and assign them an initial interest score based on the user's behavior.

#### **Key Definitions:**

1. **Interest Score:**
   - **1-3 (Low Interest)**: Little to no engagement or negative feedback. The user has shown disinterest or avoidance of content related to this topic.
   - **4-6 (Mild Interest)**: The user has shown moderate engagement. They might be curious but not strongly driven to consume more content about this topic.
   - **7-8 (High Interest)**: The user actively engages with the topic and provides positive feedback. They seem genuinely interested and would likely want more content of this type.
   - **9-10 (Very High Interest)**: The user shows strong enthusiasm and continuous engagement. They are highly likely to want more content related to this topic and may even prefer it over other topics.

#### **Input Information for You (LLM):**

1. **User Actions:**
   - **Upvote (Positive Feedback)**: The user explicitly indicates enjoyment or approval of the content.
   - **Downvote (Negative Feedback)**: The user explicitly signals disapproval or lack of interest.
   - **Clicks (Engagement)**: Clicking on an article indicates moderate to high interest depending on the frequency and recency.
   - **Conversation (Explicit Feedback)**: The user directly provides feedback through chatting with you. This could range from discussing specific preferences, explicitly asking for certain content, or expressing disinterest in other topics.

2. **Context Data:**
   - **Recent Feedback**: The last few upvotes, downvotes, or clicked articles within a given timeframe (e.g., last 10 interactions).
   - **Conversation History**: A record of recent conversations the user has had with you, including questions they’ve asked, comments made about content, and preferences they’ve shared.
   - **Previous Scores**: The existing scores you’ve assigned to topics based on the user's earlier interactions.

#### **Steps to Evaluate and Adjust Scores:**

1. **Review Recent User Feedback**: Analyze recent interactions—upvotes, downvotes, clicks, and conversations. Prioritize more recent interactions, but also consider patterns from the past (e.g., consistent upvotes on certain types of content).
   
2. **Adjust Existing Topic Scores**:
   - **Increase Scores**: Raise the score when the user has upvoted, frequently clicked, or shown enthusiasm through conversations. Use the definitions for high interest (7-10) if feedback has been consistently positive.
   - **Decrease Scores**: Lower the score when the user downvotes or consistently avoids engaging with a topic. Drop the score to the low-interest range (1-3) if the user shows repeated disinterest or expresses negativity.
   - **Fine-Tune Scores**: If the user shows mixed feedback (e.g., an upvote but no further interaction), adjust the score into the mild interest range (4-6). 

3. **Introduce New Topics (if applicable)**:
   - **Identify New Topics**: If a user starts engaging with content outside the existing topics or expresses interest in new areas during conversation, you may create a new topic category.
   - **Initial Score Assignment**: Assign a score based on the type of feedback:
     - **Moderate Engagement or Neutral Conversations**: Start with a score of 6.
     - **Strong Positive Feedback or Interest**: Start with a score of 8 or 9.
     - **Caution with Initial Scores**: Avoid assigning a 10 immediately unless there is overwhelming evidence of interest.

4. **Consideration of User-Specific Traits**:
   - **Interest Shifts**: If the user frequently changes interests or indicates that their preferences are evolving, be more flexible in how you adjust scores.
   - **User Preferences**: Take into account the specific preferences they’ve communicated. If the user has explicitly said they prefer certain topics, prioritize those and consider raising the scores faster when positive feedback occurs.
   - **Context Sensitivity**: Be aware of contextual factors (e.g., user mood, urgency of certain topics). For example, if the user inquires about a trending news topic, consider starting with a higher score in that topic area.

#### **Sample Workflow:**

1. **Review User's Recent Interactions**:
   - Upvotes two articles about "Climate Change" and clicks on a related story.
   - Downvotes one article about "Celebrity Gossip" and does not click on a similar topic.
   - Discusses interest in "AI Developments" during a recent conversation.

2. **Adjust Existing Scores**:
   - **Climate Change**: Given multiple positive signals (upvotes + clicks), increase the score to 8 or 9 (high interest).
   - **Celebrity Gossip**: With negative feedback (downvote), reduce the score to 2 (low interest).
   - **AI Developments**: Based on a conversation expressing interest, increase or introduce this as a topic with a starting score of 7 or 8 (high interest).

3. **Introduce New Topic**:
   - If the user suddenly starts clicking on "Space Exploration" articles and discusses space-related technologies, introduce this new topic with an initial score of 6 (mild interest) or 7 (high interest).

#### **Summary**:
Your goal is to continually refine your understanding of what the user finds interesting by analyzing feedback and conversations. Be responsive to changes in their preferences, and adjust your scoring accordingly. Keep the scores updated in real-time and ensure new topics are introduced when necessary.

---

This prompt provides clarity on how the LLM should assess, score, and adjust content interest levels based on user feedback, behavior, and explicit communication. You can further tweak it based on how the LLM performs in practice.
        Current topics the user expressed interest in: {topics}
        {format_instructions}
        {query}""",
        input_variables=["query", "topics"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser

    return chain.invoke({"query": query, "topics": topics})