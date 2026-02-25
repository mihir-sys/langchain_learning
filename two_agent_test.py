from dotenv import load_dotenv
import os
import json
import re
import time
from datetime import datetime
from google import genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

# ---------- Debug Logger ----------
def log(step: str, message: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{step}] {message}")

# ---------- LLM Builder ----------
def build_llm(temp: float):
    log("LLM_INIT", f"Initializing model={MODEL} | temperature={temp}")
    return ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=temp,
    )

# ---------- Answer Agent ----------
def answer_agent(question: str) -> str:
    log("ANSWER_AGENT", "Started")
    start = time.time()

    llm = build_llm(0.7)

    messages = [
        SystemMessage(content="You are AnswerAgent. Provide a clear and accurate answer."),
        HumanMessage(content=question),
    ]

    response = llm.invoke(messages)

    duration = round(time.time() - start, 2)
    log("ANSWER_AGENT", f"Completed in {duration}s")
    log("ANSWER_AGENT", f"Raw response length={len(response.content)} chars")

    return response.content

# ---------- Validator Agent ----------
def validator_agent(question: str, answer: str) -> dict:
    log("VALIDATOR_AGENT", "Started")
    start = time.time()

    llm = build_llm(0.2)

    messages = [
        SystemMessage(content=(
            "You are ValidatorAgent.\n"
            "Check the answer for correctness and hallucinations.\n"
            "Return STRICT JSON:\n"
            "{ verdict: APPROVE|REVISE, issues: [], revised_answer: '', confidence: 0-1 }"
        )),
        HumanMessage(content=f"Question:\n{question}\n\nAnswer:\n{answer}")
    ]

    response = llm.invoke(messages)

    duration = round(time.time() - start, 2)
    log("VALIDATOR_AGENT", f"Completed in {duration}s")
    log("VALIDATOR_AGENT", f"Raw output:\n{response.content}")

    # Extract JSON safely
    match = re.search(r"\{.*\}", response.content, re.DOTALL)
    if not match:
        log("VALIDATOR_AGENT", "Failed to parse JSON")
        return {
            "verdict": "REVISE",
            "issues": ["Validator output not valid JSON"],
            "revised_answer": "",
            "confidence": 0.3,
            "raw": response.content
        }

    return json.loads(match.group(0))

# ---------- Main Execution ----------
def run(question: str):
    log("SYSTEM", f"Starting multi-agent pipeline for question: {question}")

    answer = answer_agent(question)

    print("\n--- AnswerAgent Output ---\n")
    print(answer)

    validation = validator_agent(question, answer)

    print("\n--- ValidatorAgent Output ---\n")
    print(json.dumps(validation, indent=2))

    if validation.get("verdict") == "REVISE" and validation.get("revised_answer"):
        print("\n--- Final Revised Answer ---\n")
        print(validation["revised_answer"])
    else:
        print("\n--- Final Approved Answer ---\n")
        print(answer)

    log("SYSTEM", "Pipeline finished")


if __name__ == "__main__":
    text = "What are the 7 wonders of the world?"
    run(text)
    resp = client.models.count_tokens(
    model=MODEL,
    contents=text
    )

    print("Token count:", resp.total_tokens)
