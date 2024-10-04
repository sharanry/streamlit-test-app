import streamlit as st
from push_notfications import send_push

st.title('Health Assistant')

if st.button("Send Notification"):
    send_push(
            title="Hello, World!",
            body="This is a test notification.",
            only_when_on_other_tab = False,
            tag="test"
        )

