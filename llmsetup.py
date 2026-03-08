import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def main():
    print("Print google api key")
    api_key = os.environ.get("GEMINI_API_KEY")

    # Initialize model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

    # Test call
    response = llm.invoke("Write a haiku about airplanes")
    print("Response:", response.content)

    response1 = llm.invoke("What is worlds 7 wonders as of today")
    print(response1.content)


if __name__ == "__main__":
    main()
