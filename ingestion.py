import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
index_name = os.getenv("INDEX_NAME")

if __name__ == "__main__":
    print("ingesting process begun..")
    loader = TextLoader("/Users/darlingtonnnam/Desktop/programming/Python:Langchain/vector-intro/mediumblog1.txt")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    print("ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=index_name)
    print("finished!")