import os
import streamlit as st

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

system_message = """
あなたは研究アシスタントです。ユーザは研究者で、あなたに研究に関する質問を投げかけます。
アシスタントとして、論文執筆に役立つ回答を、できる限り根拠を示した上で返してください。"""
prompt = ChatPromptTemplate.from_messages([
  SystemMessagePromptTemplate.from_template(system_message),
  MessagesPlaceholder(variable_name="history"),
  HumanMessagePromptTemplate.from_template("{input}")
])

@st.cache_resource
def get_conversation():
  llm = ChatOpenAI(temperature=0)
  memory = ConversationBufferMemory(return_messages=True)
  conversation = ConversationChain(
    memory=memory,
    prompt=prompt,
    llm=llm
  )
  return conversation

st.title("研究アシスタント")
with st.form("研究アシスタントに質問する"):
  user_message = st.text_area("質問を入力してください")
  submitted = st.form_submit_button("質問する")
  if submitted:
    conversation = get_conversation()
    answer = conversation.predict(input=user_message)
    st.write(answer)
