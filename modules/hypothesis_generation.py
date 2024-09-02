
hypothesis_candidates_template = """
How can we solve the problem described below? Please provide multiple hypotheses in list format.

Problem:
{problem}
"""

hypothesis_selection_template = """
Please select the easiest-to-test hypothesis from among the hypotheses below.

Hypotheses:
{hypotheses}
"""

def generate_hypothesis(problem: str, llm: object) -> str:
    hypothesis_candidates = llm(hypothesis_candidates_template.format(problem=problem))
    selected_hypothesis = llm(hypothesis_selection_template.format(hypotheses=hypothesis_candidates))
    return selected_hypothesis