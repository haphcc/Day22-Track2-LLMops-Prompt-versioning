import os
import sys
import hashlib
from dotenv import load_dotenv

# Add root directory to sys.path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

# 1. Environment setup
load_dotenv()
client = Client()

# 2. Define two prompt versions
SYSTEM_V1 = "You are a concise assistant. Answer the question in 1-2 sentences using the context.\n\nContext:\n{context}"
PROMPT_V1 = ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")])

SYSTEM_V2 = "You are a detailed assistant. Provide a structured answer with bullet points if necessary, using the context.\n\nContext:\n{context}"
PROMPT_V2 = ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")])

# 3. Push to Prompt Hub
def push_prompts():
    print("Pushing prompts to LangSmith Prompt Hub...")
    try:
        client.push_prompt("rag-prompt-v1", object=PROMPT_V1, description="Concise RAG prompt")
        client.push_prompt("rag-prompt-v2", object=PROMPT_V2, description="Detailed RAG prompt")
        print("[SUCCESS] Prompts pushed to Hub.")
    except Exception as e:
        print(f"[WARNING] Could not push prompts: {e}")

# 4. Pull from Hub
def pull_prompts():
    print("Pulling prompts from LangSmith Prompt Hub...")
    try:
        p1 = client.pull_prompt("rag-prompt-v1")
        p2 = client.pull_prompt("rag-prompt-v2")
        return p1, p2
    except Exception as e:
        print(f"[ERROR] Could not pull prompts: {e}")
        return PROMPT_V1, PROMPT_V2 

# 5. A/B Routing logic
def get_prompt_version(request_id: str) -> str:
    h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return "v1" if h % 2 == 0 else "v2"

# 6. Instrumented RAG function
@traceable(name="rag-ab-test")
def ask_with_routing(llm, prompt, question: str, context: str, version: str) -> str:
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})

# 7. Load QA pairs
from qa_pairs import QA_PAIRS

def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub & A/B Routing")
    print("=" * 60)
    
    push_prompts()
    p1, p2 = pull_prompts()
    prompts = {"v1": p1, "v2": p2}
    
    llm = ChatOpenAI(model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"))
    mock_context = "LangChain is a framework for building LLM applications. LangSmith is for tracing and evaluation."
    
    print(f"Running A/B test on {len(QA_PAIRS)} questions...")
    for i, qa in enumerate(QA_PAIRS, 1):
        version = get_prompt_version(qa["question"])
        ask_with_routing(llm, prompts[version], qa["question"], mock_context, version)
        print(f"[{i:02d}/50] [{version}] Q: {qa['question'][:50]}...")
        
    print(f"\n[SUCCESS] A/B routing test completed.")

if __name__ == "__main__":
    main()
