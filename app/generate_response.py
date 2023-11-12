from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA




load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from utls.compose_prompt import create_prompt


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def generate_assistant_response(query, memory):
    model = ChatOpenAI(model = 'gpt-4-1106-preview')
    prompt = create_prompt()
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
    )
    return chain.invoke(query).content
