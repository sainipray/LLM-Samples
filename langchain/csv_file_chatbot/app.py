import os
import tempfile
import uuid

import pandas as pd
import streamlit as st
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()

session_id = st.session_state.id


def reset_chat():
    """Reset the chat history."""
    st.session_state.messages = []


def display_csv(file):
    """Display the uploaded CSV file as a preview."""
    st.markdown("### CSV Preview")
    df = pd.read_csv(file)
    st.dataframe(df)
    return df


@st.cache_resource
def load_llm():
    """Load OllamaChat LLM."""
    return ChatOllama(model="llama3.2", request_timeout=120.0)


with st.sidebar:
    st.header("Add Your Documents!")

    uploaded_file = st.file_uploader("Choose your `.csv` file", type=["csv"])

    if uploaded_file:
        try:
            # Save the file in a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)

                # Write the uploaded file to the temp directory
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Cache the agent using the temp file path
                file_key = f"{session_id}-{uploaded_file.name}"
                df = display_csv(temp_file_path)
                if file_key not in st.session_state.get("file_cache", {}):
                    llm = load_llm()
                    agent = create_pandas_dataframe_agent(
                        llm,
                        df,
                        verbose=True,
                        allow_dangerous_code=True,
                        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
                    )
                    st.session_state["file_cache"] = {file_key: agent}
                else:
                    agent = st.session_state["file_cache"][file_key]

                st.success("Ready to Chat!")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header("Chat with Your CSV Data using LangChain + Ollama 🚀")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Use the agent to query the CSV
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            agent = st.session_state["file_cache"][file_key]
            response = agent.run(prompt)
            message_placeholder.markdown(response)
            full_response = response
        except Exception as e:
            full_response = f"An error occurred: {e}"
            message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
