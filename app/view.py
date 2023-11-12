import streamlit as st
# import the function generate_assistant_response from the file geneerate_response.py
from generate_response import generate_assistant_response
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings



def save_chat_history(prompt, memory):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    with st.chat_message("user"):
        st.markdown(prompt)
    assistant_response = generate_assistant_response(prompt, memory)
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_response,
        }
    )


def main():
    st.title("ChatGPT Clone with ConversationChain")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    embedding_function = OpenAIEmbeddings()
    db = Chroma(
        persist_directory="./db_chemo_guide",
        embedding_function=embedding_function,
    )
    retriever = db.as_retriever(search_type="mmr", k=4) 

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("What is up?")

    if prompt:
        save_chat_history(prompt, memory, retriever)


if __name__ == "__main__":
    main()