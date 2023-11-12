# pip install "unstructured[md]"
# pip install unstructured
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
import pickle
from dotenv import load_dotenv
import time

load_dotenv()
embedding_function = OpenAIEmbeddings()

def load_pickle(path):
    # load pickled document from file\
    with open(path, "rb") as f:
        pickled_docs = f.read()
    return pickled_docs

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



if __name__ == "__main__":
    main()
