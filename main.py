from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
load_dotenv(override=True)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
def main():
    print("Hello, World! from main.py langchain learning")

    question = "WHAT IS TODAYS DATE AND TIME IN Dubai"

    # Initialize Gemini model  
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",temperature=0,
    google_api_key=GEMINI_API_KEY
    )

    # Initialize Ollama model
    llm_lamma = ChatOllama(model="gemma3:270m", temperature=0)
    response = llm_lamma.invoke(question)
    print("Olamma Response:",  response.content)

    response = llm.invoke(question)
    print("Gemini Response:",  response.content)

if __name__ == "__main__":
    main()