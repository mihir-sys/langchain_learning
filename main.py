from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv(override=True)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
def main():
    print("Hello, World! from main.py langchain learning")

    # Initialize model  
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY
    )

    question = "Explain what is Langchain and how it works in 20 words"
    response = llm.invoke(question)
    print("Response:",  response.content)

if __name__ == "__main__":
    main()