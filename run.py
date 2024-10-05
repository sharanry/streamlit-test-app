import streamlit as st
from push_notfications import send_push

import plotly.express as px

st.set_page_config(layout="wide")

# Set up the main layout
col1, col2 = st.columns([3, 1])

if 'thread' not in st.session_state:
    st.session_state.thread = ["""test"""]
    print("User action: Chat history initialized.")



def clear_input():
    st.session_state.user_input = ""
# Chat box in the left column
with col1:
    st.header("Chat")
    # Clear chat history button
    if st.button("Clear Chat"):
        st.session_state.threads = []
        print("User action: Clearing chat history.")    

    # Display chat threads in a scrollable container
    chat_container = st.container(key="chat")
    with chat_container:
        chat_container.markdown(
            """
            <div style='height: 80%; overflow-y: auto;'>
            """,
            unsafe_allow_html=True
        )
        for message in st.session_state.thread:
            st.text(message)
        chat_container.markdown("</div>", unsafe_allow_html=True)   

    # Input for new messages at the footer of the page
    # Footer section for input and send button
    st.markdown("<div style='position: fixed; bottom: 0; height: 20%; width: 75%;'>", unsafe_allow_html=True)
    
    @st.fragment
    def send_message():
        user_input = st.session_state.user_input
        print(F"User action: Sending message - {user_input}")
        if user_input:
            st.session_state.thread.append(f"User: {user_input}")
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
    # st.subheader("Notifications")
    # if st.button("Send Notification"):
    #     send_push("New message received!")
    #     print("User action: Notification sent.")
    #     st.success("Notification sent.")
    # with st.expander("Show/Hide Topics and Weights Chart"):
    topics = ["Ukraine War", "Weather", "Cricket"]
    weights = [0.5, 0.3, 0.2]

    # Sort topics and weights from most weighed to least weighed
    sorted_topics_weights = sorted(zip(weights, topics), reverse=True)
    weights, topics = zip(*sorted_topics_weights)

    fig = px.bar(x=topics, y=weights, labels={'x': 'Topics', 'y': 'User Preference'}, title='Histogram of Topics and Weights', color=topics)
    fig.update_layout(showlegend=False)
    fig.update_layout(height=400)  # Set the height to be smaller

    st.plotly_chart(fig)

