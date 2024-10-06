import plotly.express as px
import streamlit as st
from backend import initial_state
from backend import recsys
from backend import llm
import os
import sys

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


@st.fragment(run_every=5)
def run_recommendation_system():
    item = rec.sample()
    print(item)
    st.session_state.thread.append(item)


run_recommendation_system()

with col1:
    st.header("AI Curator")
    # Display chat threads in a scrollable container
    chat_container = st.container(key="chat")
    with chat_container:
        chat_container.markdown(
            """
            <div style='height: 20vh; overflow-y: hidden; position:fixed;'>
            """,
            unsafe_allow_html=True
        )
        for item in st.session_state.thread:
            print(type(item), isinstance(item, recsys.ChatItem))
            # Initialize an empty string for the content
            content = ""

            if isinstance(item, recsys.ChatItem):
                content = f"""
                <div style='text-align: right; background-color: rgba(0, 123, 255, 0.2); color: white; padding: 10px; margin: 10px; border-radius: 10px; max-width: 60%; float: right; clear: both;'>
                    <div><b>{item.sender}</b></div>
                    <div>{item.message}</div>
                </div>
                """

            elif isinstance(item, recsys.XKCDItem):
                print("has xkcd")
                news = item.title
                print(news)
                content = f"""
                <div style='text-align: left; background-color: rgba(40, 167, 69, 0.2); color: white; padding: 10px; margin: 10px; border-radius: 10px; max-width: 60%; float: left; clear: both;'>
                    <div><b>Title:</b> <a href="{item.link}" target="_blank">{item.title}</a></div>
                    <div><b>Description:</b> {item.alt_text}</div>
                </div>
                """

            elif isinstance(item, recsys.GNewsItem):
                extra_info = ""
                content = f"""
                <div style='text-align: left; background-color: rgba(220, 53, 69, 0.2); color: white; padding: 10px; margin: 10px; border-radius: 10px; max-width: 60%; float: left; clear: both;'>
                    <div><b>{item.title}</b></div>
                    <div>{item.description}</div>
                </div>
                """

            # Call chat_container.markdown once, after constructing the content
            if content:
                chat_container.markdown(content, unsafe_allow_html=True)

        chat_container.markdown("</div>", unsafe_allow_html=True)

    # Input for new messages at the footer of the page
    # Footer section for input and send button
    st.markdown("<div style='position: fixed; bottom: 0; height: 20%; width: 75%;'>",
                unsafe_allow_html=True)

    @st.fragment
    def send_message():
        user_input = st.session_state.user_input
        print(F"User action: Sending message - {user_input}")
        if user_input:
            message = {"sender": "User", "message": user_input}
            st.session_state.thread.append(message)
            print(f"User action: Message sent - {user_input}")
            # clear_input()

    st.text_input("Type your message here:", key="user_input")

    if st.button("Send", key="send_button"):
        send_message()
        # st.session_state.user_input = ""
        st.rerun()

    if st.session_state.user_input and st.session_state.user_input.endswith('\n'):
        send_message()
    st.markdown("</div>", unsafe_allow_html=True)

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
