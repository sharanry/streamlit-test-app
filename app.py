import plotly.express as px
import streamlit as st
from backend import initial_state
from backend import recsys
from backend import llm
import os
import sys
import asyncio

testdir = os.path.dirname(__file__)
srcdir = './frontend'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
srcdir = './backend'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))


st.set_page_config(layout="wide")

st.session_state.thread = initial_state.thread
st.session_state.topics = initial_state.topics
print("User action: Chat history initialized.")

# Instantiating ML stuff
llm = llm.ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=5,
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)

# Set up the main layout
col1, col2 = st.columns([3, 2])


def clear_input():
    st.session_state.user_input = ""


rec = initial_state.new_recommender()

# @st.fragment(run_every=5)
async def run_recommendation_system():
    while True:
        await asyncio.sleep(5)
        print("Running recommendation system")
        st.session_state.thread.append({"role": "AI", "message": "test"})
        print(len(st.session_state.thread))
        st.rerun(scope="chat")
    # item = rec.sample()
    # print(item)
    # st.session_state.thread.append(item) 


with col1:
    st.session_state.thread = initial_state.thread
    st.session_state.topics = initial_state.topics
    st.header("AI Curator")
    # Display chat threads in a scrollable container
    chat_container = st.container(key="chat", height=400, border=False)

    @st.fragment
    def display_chat():
        with chat_container:
            for message in st.session_state.thread:
                role = message["role"]
                text = message["message"]
                
                if isinstance(item, recsys.GNewsItem):
                    news = message["news"]
                    with st.chat_message(item.sender, avatar="🗞️"):
                        st.markdown(item.title)
                        st.markdown(f"""**[{item.title}]({""})**
                        {item.description}
                        """)
                elif isinstance(item, recsys.ChatItem):
                    with st.chat_message(role):
                        st.markdown(text)

    display_chat()
    @st.fragment
    def send_message():
        user_input = st.session_state.user_input
        print(F"User action: Sending message - {user_input}")
        if user_input:
            message = {"sender": "User", "message": user_input}
            st.session_state.thread.append(message)
            print(f"User action: Message sent - {user_input}")
            # clear_input()
    
    st.chat_input("Talk to your AI Curator", key="user_input", on_submit=send_message)


# Notification system in the right column
with col2:
    topics = [topic["topic"] for topic in st.session_state.topics]
    weights = [topic["weight"] for topic in st.session_state.topics]
    times = [topic["time"] for topic in st.session_state.topics]

    # Sort topics and weights from most weighed to least weighed
    sorted_topics_weights = sorted(zip(weights, topics, times), reverse=True)
    weights, topics, times = zip(*sorted_topics_weights)

    labels = [f"{topic}\n({time.strftime('%I %p')})" for topic,
              time in zip(topics, times)]

    # print(labels, weights, topics, times)

    fig = px.bar(x=labels, y=weights, labels={
                 'x': 'Topics', 'y': 'User Preference'}, title='Histogram of Topics and Weights', color=topics)
    fig.update_layout(showlegend=False)
    fig.update_layout(height=400)  # Set the height to be smaller

    st.plotly_chart(fig)
<<<<<<< HEAD
=======


# asyncio.run(run_recommendation_system())
>>>>>>> b0bcb86 (sharan updates)
