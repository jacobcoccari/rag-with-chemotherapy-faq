from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def generate_assistant_response(query, memory):
    model = ChatOpenAI(model = 'gpt-4-1106-preview')
    embedding_function = OpenAIEmbeddings()
    db = Chroma(
        persist_directory="./db_chemo_guide",
        embedding_function=embedding_function,
    )
    # print(db.get_document_count())

    retriever = db.as_retriever(search_type="mmr")  
    return retriever.get_relevant_documents(query, k=3)
    # template = """Answer the question based only on the following context:

    # {context}

    # Question: {question}
    # """
    # prompt = ChatPromptTemplate.from_template(template)
    # model = ChatOpenAI()

    # chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | model
    #     | StrOutputParser()
    # )
    # return chain.invoke(query)
