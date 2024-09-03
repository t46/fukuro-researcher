from modules.hypothesis_generation import generate_hypothesis
from modules.verification_execution import perform_verification
from modules.paper_writing import perform_writeup

llm_name = "openai"

def main():
    generate_hypothesis("I want to build a house", llm_name=llm_name)
    perform_verification("I want to build a house", llm_name=llm_name)
    perform_writeup("I want to build a house", llm_name=llm_name)


if __name__ == "__main__":
    main()