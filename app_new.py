import plotly.express as px
import streamlit as st
import backend.initial_state as initial_state
import backend.recsys as recsys
import backend.llm as llm
import os
import sys
import asyncio
import copy
import pandas as pd
from tabulate import tabulate

testdir = os.path.dirname(__file__)
srcdir = './frontend'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
srcdir = './backend'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

st.set_page_config(layout="wide")

if "thread" not in st.session_state:
    st.session_state.thread = copy.deepcopy(initial_state.thread)
if "topics" not in st.session_state:
    st.session_state.topics = copy.deepcopy(initial_state.topics)

# Instantiating ML stuff
llm2 = llm.ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=5,
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)

rec = initial_state.new_recommender()

if "arms" not in st.session_state:
    st.session_state.arms = rec.get_arms()["arms"]

def show_message(item):
    if isinstance(item, recsys.GNewsItem):
        with st.chat_message("AI", avatar="🗞️"):
            st.markdown(f"""**[{item.title}]({item.url})**""")
            st.markdown(item.description)
    if isinstance(item, recsys.XKCDItem):
        with st.chat_message("AI", avatar="😂"):
            st.markdown(f"**XKCD comic #{item.number}**")
            st.markdown(f"![{item.title}]({item.image_link})")
            # st.markdown(f"*{item.alt_text}*")
    elif isinstance(item, recsys.ChatItem):
        with st.chat_message(item.sender):
            st.markdown(item.message)

st.header("Fred")
st.subheader("Your friendly neighborhood AI Curator")


with st.container(border=False):
    arms = st.session_state.arms
    topics = [arm["name"] for arm in arms]
    weights = [arm["score"] for arm in arms]
    # times = [arm.pulls for arm in arms]

    # Sort topics and weights from most weighed to least weighed
    sorted_topics_weights = sorted(zip(weights, topics), reverse=True)
    weights, topics = zip(*sorted_topics_weights)


    fig = px.bar(x=topics, y=weights, labels={
                    'x': 'Topics', 'y': 'User Preference'}, title='Histogram of Topics and Weights', color=topics)
    fig.update_layout(showlegend=False)
    fig.update_layout(height=300, font=dict(size=12))  # Set the height to be smaller

    st.plotly_chart(fig)


for item in st.session_state.thread:
    show_message(item)


if user_input := st.chat_input("Talk to your AI Curator", key="user_input"):
    print("user_input ", user_input)
    print(F"User action: Sending message - {user_input}")
    item = recsys.ChatItem({"sender": "user", "message":user_input})
    st.session_state.thread.append(item)
    if isinstance(item, recsys.GNewsItem):
        with st.chat_message("AI", avatar="🗞️"):
            st.markdown(f"""**[{item.title}]({item.url})**""")
            st.markdown(item.description)
    if isinstance(item, recsys.XKCDItem):
        with st.chat_message("AI", avatar="😂"):
            st.markdown(f"**XKCD comic #{item.number}**")
            st.markdown(f"![{item.title}]({item.image_link})")
            # st.markdown(f"*{item.alt_text}*")
    elif isinstance(item, recsys.ChatItem):
        with st.chat_message(item.sender):
            st.markdown(item.message)   
    original_arms = rec.get_arms()
    topics_old = [arm["name"] for arm in original_arms["arms"]]
    weights_old = [arm["score"] for arm in original_arms["arms"]]
    updated_arms = llm.update_arm_scores(llm2, original_arms, user_input)
    rec.update_arms(updated_arms.dict())
    st.session_state.arms = rec.get_arms()["arms"]
    topics = [arm["name"] for arm in rec.get_arms()["arms"]]
    weights = [arm["score"] for arm in rec.get_arms()["arms"]]
    print(weights)
    print(topics)
    sorted_topics_weights = sorted(zip(weights, topics), reverse=True)
    weights, topics = zip(*sorted_topics_weights)
    message = f"""Updating preferences from\n
    {str(dict(zip(topics_old,   weights_old)))}\n
    to\n
    {str(dict(zip(topics, weights)))}
    """
    st.session_state.thread.append(recsys.ChatItem({"sender": "AI", "message": message}))
    with st.chat_message("AI"):
        st.markdown(message)
        


if st.button("Refresh State"):
    st.session_state.thread = copy.deepcopy(initial_state.thread)
    st.session_state.topics = copy.deepcopy(initial_state.topics)
    print("State refreshed")
    st.rerun()



@st.fragment(run_every=10)
def run_recommendation_system():
    print("Polling recommendation system")
    item = rec.sample()
    st.session_state.thread.append(item) 
    # display_chat()
    st.rerun()

run_recommendation_system()

# print(vars(chat_container))
# @st.fragment
# def send_message():
#     user_input = st.session_state.user_input
#     print("user_input ", user_input)
#     print(F"User action: Sending message - {user_input}")
#     item = recsys.ChatItem({"sender": "user", "message":user_input})
#     # print(f"User action: Message sent - {item}")
#     st.session_state.thread.append(item)
#     display_chat()
#     # st.rerun()
