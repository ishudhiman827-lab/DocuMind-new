from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

def create_db(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(docs, embeddings)
    return db

def get_ai_answer(query, db):
    docs = db.similarity_search(query)

    llm = ChatOpenAI(temperature=0.3)
    response = llm.predict(
        f"Answer based on this:\n{docs[0].page_content}\n\nQuestion: {query}"
    )

    return response