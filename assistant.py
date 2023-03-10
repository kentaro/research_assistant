import os

import openai
import streamlit as st

openai.api_key = os.environ["OPENAI_API_KEY"]

system_content = """
あなたは研究アシスタントです。ユーザは研究者で、あなたに研究に関する質問を投げかけます。
アシスタントとして、論文執筆に役立つ回答を、できる限り根拠を示した上で返してください。
"""

def on_click_handler(user_content):
  assistant_content = st.session_state.assistant_content or ""
  answer = call_chatgpt(user_content, assistant_content)

  st.session_state.assistant_content = answer
  return answer

def call_chatgpt(user_content, assistant_content):
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": system_content},
      {"role": "user", "content": user_content},
      {"role": "assistant", "content": assistant_content},
    ],
  )

  return response.choices[0]["message"]["content"].strip()

st.title("研究アシスタント")
if "assistant_content" not in st.session_state:
    st.session_state.assistant_content = ""

with st.form("研究アシスタントに質問する"):
  user_content = st.text_area("質問を入力してください")
  submitted = st.form_submit_button("質問する")
  if submitted:
    answer = on_click_handler(user_content)
    st.write(answer)
