import os
import sys

testdir = os.path.dirname(__file__)
srcdir = './frontend'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

import streamlit as st

from frontend import auth
auth.show_login_modal()
if st.session_state['authentication_status']:
    auth.authenticator.logout()
    print(st.session_state)
    st.header(f'Hey **{st.session_state["name"]}**!')
    st.write('You are logged in!')
elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')

