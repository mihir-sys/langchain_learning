import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
def main():
    print("Hello, World! from main.py langchain learning")
    print("Print google api key")
    api_key = os.environ.get("GEMINI_API_KEY")
    print(api_key)

    # Initialize model  
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
    )
if __name__ == "__main__":
    main()