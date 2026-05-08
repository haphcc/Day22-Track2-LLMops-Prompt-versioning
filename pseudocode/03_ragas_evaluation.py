import os
import sys
import json
import numpy as np
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Add root directory to sys.path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

warnings.filterwarnings("ignore")
load_dotenv()

from qa_pairs import QA_PAIRS

SYSTEM_V1 = "You are a concise assistant. Answer using the context.\n\nContext:\n{context}"
PROMPT_V1 = ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")])

SYSTEM_V2 = "You are a detailed assistant. Use bullet points if helpful. Answer using the context.\n\nContext:\n{context}"
PROMPT_V2 = ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}

def build_vectorstore():
    kb_content = Path("data/knowledge_base.txt").read_text(encoding="utf-8")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(kb_content)
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    return FAISS.from_texts(chunks, embeddings)

def collect_rag_outputs(vectorstore, prompt_version: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"))
    prompt = PROMPTS[prompt_version]
    results = []
    print(f"Collecting RAG outputs for {prompt_version}...")
    for i, qa in enumerate(QA_PAIRS, 1):
        question = qa["question"]
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        ctx_str = "\n\n".join(contexts)
        answer = (prompt | llm | StrOutputParser()).invoke({"context": ctx_str, "question": question})
        results.append({"question": question, "reference": qa["reference"], "answer": answer, "contexts": contexts})
        if i % 10 == 0: print(f"  Processed {i}/50...")
    return results

def run_ragas_eval(rag_results, version):
    print(f"Running RAGAS evaluation for {version}...")
    samples = [SingleTurnSample(user_input=r["question"], response=r["answer"], retrieved_contexts=r["contexts"], reference=r["reference"]) for r in rag_results]
    dataset = EvaluationDataset(samples=samples)
    llm_eval = ChatOpenAI(model="gpt-4o-mini")
    emb_eval = OpenAIEmbeddings(model="text-embedding-3-small")
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall, context_precision], llm=llm_eval, embeddings=emb_eval)
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        scores[key] = float(np.mean([v for v in raw if v is not None]))
    return scores

def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)
    vectorstore = build_vectorstore()
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")
    print("\n--- Comparison Table ---")
    for m in v1_scores:
        s1, s2 = v1_scores[m], v2_scores[m]
        winner = "V1" if s1 > s2 else "V2"
        print(f"  {m:20}: V1={s1:.4f}  V2={s2:.4f}  (Winner: {winner})")
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    print(f"\n[DONE] Best faithfulness = {best_faith:.4f}")
    report = {"v1": v1_scores, "v2": v2_scores}
    with open("data/ragas_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
