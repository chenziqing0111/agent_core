# /tools/rag_engine/rag_orchestrator.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document


text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = OpenAI(temperature=0)


def answer_question_with_rag(question: str, docs: list[str]) -> str:
    # 1. 文本切分 + 构建 Document
    chunks = text_splitter.split_documents([Document(page_content=doc) for doc in docs])

    # 2. 建立向量索引
    vectorstore = FAISS.from_documents(chunks, embedder)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 3. 构建 RAG 问答链
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa.run(question)
    return result
