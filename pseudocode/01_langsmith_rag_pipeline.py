import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add root directory to sys.path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

# 1. Environment setup
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# 2. Initialize LLM and Embeddings
llm = ChatOpenAI(
    model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# 3. Build FAISS vector store
def build_vectorstore():
    print("Indexing knowledge base...")
    kb_path = Path("data/knowledge_base.txt")
    if not kb_path.exists():
        raise FileNotFoundError("Knowledge base file not found at data/knowledge_base.txt")
        
    text = kb_path.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# 4. RAG prompt template
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context below to answer the question. If the answer is not in the context, say you don't know.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

# 5. Build the RAG chain
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# 6. Traced query function
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    return chain.invoke(question)

# 7. Load QA pairs
from qa_pairs import QA_PAIRS

def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    try:
        vectorstore = build_vectorstore()
        chain, retriever = build_rag_chain(vectorstore)

        print(f"Running {len(QA_PAIRS)} questions...")
        for i, qa in enumerate(QA_PAIRS, 1):
            question = qa["question"]
            answer = ask(chain, question)
            print(f"[{i:02d}/{len(QA_PAIRS)}] Q: {question[:60]}...")

        print(f"\n[SUCCESS] {len(QA_PAIRS)} traces sent to LangSmith.")
        print("   Open https://smith.langchain.com to view traces.")
        
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
