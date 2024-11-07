import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
index_name = os.getenv("INDEX_NAME")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print("retrieving...")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    template = """
        Use the following pieces of context to answer the question at the end. If the context does not contain an answer, feel free to hallucinate mildly.
        Keep the answer as concise as possible, and always say "thanks for asking!" at the end of the answer.

        {context}

        Question: {question}

        Helpful Answer:
    """

    query = "What is account abstraction? explain in a lay man's terms"

    vectorstore = PineconeVectorStore(
        index_name=index_name, embedding=embeddings
    )

    custom_rag_prompt = PromptTemplate.from_template(template)
    rag_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs, 
            "question": RunnablePassthrough()
        } | custom_rag_prompt | llm
    )

    res = rag_chain.invoke(query)
    print(res.content)
