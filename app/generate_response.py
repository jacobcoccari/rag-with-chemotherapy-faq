from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from utls.compose_prompt import create_prompt
from utls.format_memory import get_chat_history
from utls.moderation import harmful_content_check


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def generate_assistant_response(query, retriever, streamlit_memory):
    history = get_chat_history(streamlit_memory)
    prompt = create_prompt(history)
    model = ChatOpenAI(model = 'gpt-4-1106-preview')
    check = harmful_content_check(query)
    if check is not None:
        print(check)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
    )
    return chain.invoke(query).content
