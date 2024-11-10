from modules.utils import run_llm
import subprocess
import sys
import json
import shutil
import os
from subprocess import TimeoutExpired
MAX_STDERR_OUTPUT = 1500

verification_plan_template = """
Given the problem and accompanying hypothesis below, how can we verify the hypothesis? Please provide a detailed verification plan composed of structured sentences. 
Ensure that the plan is sufficiently detailed and concrete so that it can be executed by a large language model and computer. 
Outline the procedure in a step-by-step manner. If necessary, break down a single task into multiple sub-tasks and list them hierarchically. 
The verification plan should be realistic and feasible, making use of existing resources rather than requiring the creation of new ones.

Problem:
{problem}

Hypothesis:
{hypothesis}

Output the verification plan in JSON format.
```json
{{
    "verification_plan": ...
}}
```
"""

verification_code_generation_template = """
You are a helpful assistant who should strictly adhere to the following guidelines:
- **DO NOT** include `api-key` in the code, as it has already been specified.
- **DO NOT** output placeholders, end up with comments, or use just a sequence of dots without fully implementing the contents of the code. Ensure that you fully implement the contents.

You are an excellent engineer. In accordance with the verification plan provided below, please output/edit Python code `experiment.py` to execute said plan. Note that you must comply with the instructions above.

Verification plan:
{verification_plan}

After you complete each change, we will run the command `python experiment.py` to execute the code. YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
You can then implement the next thing on your list.
```
"""

experiment_update_template = """
Run completed. Here are the results:
{results}

Decide if you need to re-plan your experiments given the result (you often will not need to).

Someone else will be using `notes.txt` to perform a writeup on this in the future.
Please include *all* relevant information for the writeup on this run, including an experiment description and the run number. Be as verbose as necessary.

Then, implement the next thing on your list.
We will then run the command `python experiment.py'.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
If you are finished with experiments, respond with 'ALL_COMPLETED'.
"""

plot_update_template = """
Great job! Please modify `plot.py` to generate the most relevant plots for the final writeup. 

In particular, be sure to fill in the "labels" dictionary with the correct names for each run that you want to plot.

Only the runs in the `labels` dictionary will be plotted, so make sure to include all relevant runs.

We will be running the command `python plot.py` to generate the plots.
"""

note_update_template = """
Please modify `notes.txt` with a description of what each plot shows along with the filename of the figure. Please do so in-depth.

Somebody else will be using `notes.txt` to write a report on this in the future.
"""

def run_experiment(output_directory, run_num, timeout=7200):
    current_working_directry = os.path.abspath(output_directory)

    # LAUNCH COMMAND
    command = [
        "python",
        "experiment.py"
    ]
    try:
        result = subprocess.run(command, cwd=current_working_directry, stderr=subprocess.PIPE, text=True, timeout=timeout)

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"Run failed with return code {result.returncode}")
            print(f"Run failed with the following error {result.stderr}")
            stderr_output = result.stderr
            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]
            run_result_message = f"Run failed with the following error {stderr_output}"
        else:
            with open(os.path.join(current_working_directry, "final_info.json"), "r") as f:  # TODO: 実験結果を json で保存するか否かは要検討
                results = json.load(f)
            run_result_message = experiment_update_template.format(results=results)
        return result.returncode, run_result_message
    except TimeoutExpired:
        print(f"Run {run_num} timed out after {timeout} seconds")
        run_result_message = f"Run timed out after {timeout} seconds"
        return 1, run_result_message

# RUN PLOTTING
def run_plotting(output_directory, timeout=600):
    current_working_directry = os.path.abspath(output_directory)
    # LAUNCH COMMAND
    command = [
        "python",
        "plot.py",
    ]
    try:
        result = subprocess.run(
            command, cwd=current_working_directry, stderr=subprocess.PIPE, text=True, timeout=timeout
        )

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"Plotting failed with return code {result.returncode}")
            run_result_message = f"Plotting failed with the following error {result.stderr}"
        else:
            next_prompt = ""
        return result.returncode, next_prompt
    except TimeoutExpired:
        print(f"Plotting timed out after {timeout} seconds")
        run_result_message = f"Plotting timed out after {timeout} seconds"
        return 1, run_result_message


def execute_verification(problem: str, hypothesis: str, llm_name: str, coder: object, output_directory: str) -> str:
    verification_plan = run_llm(llm_name, verification_plan_template.format(problem=problem, hypothesis=hypothesis), format="json")
    MAX_ITERS = 2
    MAX_RUNS = 2
    run = 0
    current_iter = 0
    for run in range(MAX_RUNS + 1):
        if current_iter >= MAX_ITERS:
            print("Max iterations reached")
            break
        coder_output = coder.run(verification_code_generation_template.format(verification_plan=verification_plan))
        return_code, run_result_message = run_experiment(output_directory, run)
        if "ALL_COMPLETED" in coder_output:
            break
        if return_code == 0:
            run += 1
            current_iter = 0
        current_iter += 1
    if current_iter >= MAX_ITERS:
        print("Not all experiments completed.")
        return False

    current_iter = 0
    while True:
        coder_output = coder.run(plot_update_template)
        return_code, next_prompt = run_plotting(output_directory)
        current_iter += 1
        if return_code == 0 or current_iter >= MAX_ITERS:
            break
    coder.run(note_update_template)

    return True