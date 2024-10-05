import os
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator import LoginError
import yaml
from yaml.loader import SafeLoader


with open(os.path.join(os.path.dirname(__file__), 'auth.toml')) as file:
    config = yaml.load(file, Loader=SafeLoader)

# Pre-hashing all plain text passwords once
# stauth.Hasher.hash_passwords(config['credentials'])

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)


def show_login_modal():
    try:
        authenticator.login()
        # authenticator.experimental_guest_login('Login with Google',
        #                                        provider='google',
        #                                        oauth2=config['oauth2'])
        print("User logged in")
    except LoginError as e:
        st.error(e)