# Import necessary libraries
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Set header
st.header("Simple Chat app")
st.markdown(
  """
  <style>
  .stApp {
    background-color: #f3ece0;
    font-family: monospace
  }
  </style>
  """,
  unsafe_allow_html=True
)

# initialize session state for messages
st.session_state.setdefault("messages", [])

# Model selection
selected_model = st.sidebar.selectbox(
  label="Choose a model", 
  options=[
    "Cloud: qwen/qwen3-14b:free",
    "Cloud: moonshotai/kimi-k2:free",
    "Cloud: openai/gpt-oss-20b:free",
    "Local: qwen2.5-coder:1.5b",
    "Local: gemma3:1b",
    "Local: deepseek-r1:1.5b",
  ])

# Initialize LLM based on selected model
if selected_model.startswith('Cloud'):
  model_name = selected_model.replace('Cloud: ', '')
  llm = ChatOpenAI(
    model=model_name, 
    api_key=os.getenv('OPENROUTER_API_KEY'), 
    base_url="https://openrouter.ai/api/v1"
  )
else:
  # Initialize local LLM
  model_name = selected_model.replace('Local: ', '')
  llm = ChatOllama(
    model=model_name,
    base_url="http://192.168.1.44:11434"  
  )

# Display all messages in the chat
for msg in st.session_state['messages']:
  with st.chat_message(msg['role']):
    st.write(msg['content'])

# Handle user input and add to messages
if prompt := st.chat_input(f"Hi, this is {model_name.split(':')[0]}. Ask me anything!"):
  st.session_state['messages'].append({'role': 'user', 'content': prompt})
  with st.chat_message('user'):
    st.write(prompt)

# Build context from messages
context = ""
for msg in st.session_state['messages']:
  context += msg['role'] + ': ' + msg['content']

# Generate response from LLM and add to messages
response = llm.invoke(context)
with st.chat_message('assistant'):
  st.write(response.content)

# Append assistant's response to messages
st.session_state['messages'].append({'role': 'assistant', 'content': response.content})
