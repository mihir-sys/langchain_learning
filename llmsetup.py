# from langchain_google_genai import ChatGoogleGenerativeAI
# import os

# # set your Google API key
# os.environ["GOOGLE_API_KEY"] = "AIzaSyBmXtFvlnNeaNyndcdk3G-28APnSVN3y78"

# # initialize model (Gemini 1.5 Pro as example)
# #llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.environ["GOOGLE_API_KEY"])

# # make a simple call
# response = llm.invoke("What is worlds 7 wonders as of today")
# print(response.content)

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def main():
    print("Print google api key")
    print(os.environ.get("GEMINI_API_KEY"))
    api_key = os.environ.get("GEMINI_API_KEY")

    # Initialize model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)

    # Test call
    response = llm.invoke("Write a haiku about airplanes")
    print("Response:", response.content)

    response1 = llm.invoke("What is worlds 7 wonders as of today")
    print(response1.content)


if __name__ == "__main__":
    main()
