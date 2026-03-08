from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
load_dotenv(override=True)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
def main():
    information = """The Lord of the Rings is a trilogy of epic fantasy films directed by Peter Jackson. The films are based on the novel The Lord of the Rings by J. R. R. Tolkien, and are titled identically to the three volumes of the novel: The Fellowship of the Ring (2001), The Two Towers (2002), and The Return of the King (2003). Produced and distributed by New Line Cinema with the co-production of Jackson's WingNut Films, the films feature an ensemble cast.

Set in the fictional world of Middle-earth, the films follow the hobbit Frodo Baggins as he and the Company of the Ring embark on a quest to destroy the One Ring to defeat its maker, the Dark Lord Sauron. The Company eventually splits up and Frodo continues the quest with his loyal companion Sam and, eventually, the treacherous Gollum. Meanwhile, Aragorn, heir in exile to the throne of Gondor, along with the elf Legolas, the dwarf Gimli, Merry, Pippin, Boromir, and the wizard Gandalf, unite to save the Free Peoples of Middle-earth from the forces of Sauron and rally them in the War of the Ring to aid Frodo by distracting Sauron's attention.

The three films were shot simultaneously in Jackson's native New Zealand from 11 October 1999 until 22 December 2000, with pick-up shots from 2001 to 2003. It was one of the biggest and most ambitious film projects ever undertaken, with a budget of $281 million (equivalent to $543 million in 2025). The first film in the series premiered at the Odeon Leicester Square in London on 10 December 2001; the second film premiered at the Ziegfeld Theatre in New York City on 5 December 2002; the third film premiered at the Embassy Theatre in Wellington on 1 December 2003. An extended edition of each film was released on home video a year after its release in cinemas."""

    summary_template = """Instruction: {information}
    Task:
    - Share the movie name .
    - What is the release date and year of the movie.
    - Who is the director of the movie.
    - Who is the producer of the movie.
    - Who is the writer of the movie.
    - Who is the composer of the movie.
    - Who is the actor of the movie.
    - Who is the actress of the movie.
    - Who is the music director of the movie.
    - Who is the lyricist of the movie.
    """
    summary_prmpt_templae = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    # Initialize Gemini model  
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",temperature=0,
    google_api_key=GEMINI_API_KEY
    )

    # Initialize Ollama model
    #llm_ollamma = ChatOllama(model="gemma3:270m", temperature=0)
    # response = llm_ollamma.invoke(question)
    # print("Olamma Response:",  response.content)

    cretae_chain = summary_prmpt_templae | llm
    response = cretae_chain.invoke({"information": information})
    print("Response:",  response.content)

if __name__ == "__main__":
    main()