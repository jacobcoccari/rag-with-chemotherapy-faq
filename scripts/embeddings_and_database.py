# pip install "unstructured[md]"
# pip install unstructured
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
import pickle
from dotenv import load_dotenv

load_dotenv()

def load_pickle(path):
    # load pickled document from file\
    documents = pickle.loads(
        open("./11-Langchain-Bot/langchain_documents.pkl", "rb").read()
    )
    return documents

def embed_and_store(docs):
    # create embedding function, spin up chroma, and embed all documents.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(
        embedding_function=embedding_function,
        persist_directory="./db_chemo_guide/",
    )
    db.add_documents(docs)
    db.persist()


def main():
    load_pickle_path = "./documents/pickled_documents.pkl"
    docs = load_pickle(load_pickle_path)
    embed_and_store(docs)

if __name__ == "__main__":
    main()
