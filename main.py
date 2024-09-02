from modules.hypothesis_generation import generate_hypothesis
from modules.verification_execution import perform_verification
from modules.paper_writing import perform_writeup

def main():
    generate_hypothesis("I want to build a house", )
    perform_verification("I want to build a house", )
    perform_writeup("I want to build a house", )


if __name__ == "__main__":
    main()