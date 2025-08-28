from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os
load_dotenv()


def main():
    print("Hello from langchain-course!")
    print("Loaded API Key:", os.getenv("OPENAI_API_KEY"))
    information = """
    Gandhi's vision of an independent India based on 
    religious pluralism was challenged in the early 
    1940s by a Muslim nationalism which demanded a 
    separate homeland for Muslims within British India. 
    In August 1947, Britain granted independence, but the British Indian Empire was partitioned into two dominions, a Hindu-majority India and a Muslim-majority Pakistan. As many displaced Hindus, Muslims, and Sikhs made their way to their new lands, religious violence broke out, especially in the Punjab and Bengal. Abstaining from the official celebration of independence, Gandhi visited the affected areas, attempting to alleviate distress. In the months following, he undertook several hunger strikes to stop the religious violence. The last of these was begun in Delhi on 12 January 1948, when Gandhi was 78. The belief that Gandhi had been too resolute in his defence of both Pakistan and Indian Muslims spread among some Hindus in India. Among these was Nathuram Godse, a militant Hindu nationalist from Pune, western India, who assassinated Gandhi by firing three bullets into his chest at an interfaith prayer meeting in Delhi on 30 January 1948.
    """
    summary_template = """
    given the {information} about a person I want you to create:
    1. A short summary
    2. Two Interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    #llm = ChatOpenAI(temperature=0, model="gpt-5")
    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)
if __name__ == "__main__":
    main()
