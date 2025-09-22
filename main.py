# To get the content output instead of all ai message
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


def main():
    information = "List the New 7 Wonders of the World (official 2020 list)"

    summary_template = """Instruction: {information}

    Task:
    - Return the official list of the New 7 Wonders of the World declared in 2007.
    - Present them as a numbered list (1–7).
    - Do not invent new wonders or add descriptions, just give the correct names.
    """

    prompt = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    prompt = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    # Connect to your Ollama model
    llm = ChatOllama(model="gemma3:270m", temperature=0)

    parser = StrOutputParser()

    # Create a chain: prompt → llm
    chain = prompt | llm | parser

    # Run the chain
    response = chain.invoke({"information": information})
    print("\n--- Model Output ---")
    print(response)


if __name__ == "__main__":
    main()
