import streamlit as st
import json
from feynman_twin import FeynmanTwin, safe_chat, extract_and_save_memory

st.title("🧪 Feynman Digital Twin")
st.caption("Talk to Richard P. Feynman — Nobel Prize-winning physicist")

# Initialize
if "feynman" not in st.session_state:
    st.session_state.feynman = FeynmanTwin()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show memory in sidebar
with st.sidebar:
    st.header("🧠 What Feynman Remembers")
    mem = st.session_state.feynman.long_memory
    st.json(mem)

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
user_input = st.chat_input("Ask Feynman anything...")
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get Feynman response
    with st.chat_message("assistant"):
        with st.spinner("Feynman is thinking..."):
            reply = safe_chat(st.session_state.feynman, user_input)
            extract_and_save_memory(st.session_state.feynman, user_input)
            st.write(reply)
    
    st.session_state.messages.append({"role": "assistant", "content": reply})