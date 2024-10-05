import os
import sys

testdir = os.path.dirname(__file__)
srcdir = './frontend'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
srcdir = './backend'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

from backend import llm
from backend import recsys
from backend import initial_state

import streamlit as st
import plotly.express as px

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

# Instantiating data sources
sampler_map = {
    recsys.SamplerType.GNEWS: recsys.GNewsSampler(),
}

st.set_page_config(layout="wide")



# Set up the main layout
col1, col2 = st.columns([3, 2])

def clear_input():
    st.session_state.user_input = ""
    
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
        for message in st.session_state.thread:
            role = message["role"]
            text = message["message"]
            if role == "User":
                chat_container.markdown(f"""
                <div style='text-align: right; background-color: #007bff; color: white; padding: 10px; margin: 10px; border-radius: 10px; max-width: 60%; float: right; clear: both;'>
                    <div><b>{role}</b></div>
                    <div>{text}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                chat_container.markdown(f"""
                <div style='text-align: left; background-color: #dc3545; color: white; padding: 10px; margin: 10px; border-radius: 10px; max-width: 60%; float: left; clear: both;'>
                    <div><b>{role}</b></div>
                    <div>{text}</div>
                </div>
                """, unsafe_allow_html=True)
                if "weather" in message:
                    print("has weather")
                    weather = message["weather"]
                    weather_info = f"""
                    <div>
                        <img src="https://example.com/temperature_icon.png" alt="Temperature Icon" style="width:20px;height:20px;">
                        Temperature: {weather['temperature']}°C
                    </div>
                    <div>
                        <img src="https://example.com/humidity_icon.png" alt="Humidity Icon" style="width:20px;height:20px;">
                        Humidity: {weather['humidity']}%
                    </div>
                    <div>
                        <img src="https://example.com/wind_icon.png" alt="Wind Icon" style="width:20px;height:20px;">
                        Wind: {weather['wind']}
                    </div>
                    <div>
                        <img src="https://example.com/rain_icon.png" alt="Rain Icon" style="width:20px;height:20px;">
                        Rain: {weather['rain']}
                    </div>
                    """
                    chat_container.markdown(weather_info, unsafe_allow_html=True)
                
        chat_container.markdown("</div>", unsafe_allow_html=True)

    # Input for new messages at the footer of the page
    # Footer section for input and send button
    st.markdown("<div style='position: fixed; bottom: 0; height: 20%; width: 75%;'>", unsafe_allow_html=True)
    
    @st.fragment
    def send_message():
        user_input = st.session_state.user_input
        print(F"User action: Sending message - {user_input}")
        if user_input:
            message = {"role": "User", "message": user_input}
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

    labels = [f"{topic}\n({time.strftime('%I %p')})" for topic, time in zip(topics, times)]

    # print(labels, weights, topics, times)

    fig = px.bar(x=labels, y=weights, labels={'x': 'Topics', 'y': 'User Preference'}, title='Histogram of Topics and Weights', color=topics)
    fig.update_layout(showlegend=False)
    fig.update_layout(height=400)  # Set the height to be smaller

    st.plotly_chart(fig)


