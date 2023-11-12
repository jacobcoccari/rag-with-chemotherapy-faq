from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


load_dotenv()



def generate_assistant_response(prompt):
    model = ChatOpenAI(model = 'gpt-4-1106-preview')
    memory = ConversationBufferMemory(return_messages=True)
    embedding_function = OpenAIEmbeddings()
    db = Chroma(
        persist_directory="./11-Langchain-Bot/langchain_documents_db",
        embedding_function=embedding_function,
    )

    retriever = db.as_retriever(search_type="mmr")

    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
    )
    response = qa(prompt)

    return response["result"]

