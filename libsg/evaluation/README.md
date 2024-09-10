# Evaluation

Code to evaluate scene generation methods.

## Data

You will need to create a CSV file at `.data/evaluation/test_prompts.csv` containing the list of prompts you
want to test. The currently expected columns include
* `prompt` - text description of scene
* `num_iterations` - number of iterations to run each prompt
* `test_diversity` - if `True`, evaluates prompts using diversity metrics only, else if `False`, evaluates prompts using aggregate metrics

As the test prompts we are using have not been finalized yet, please contact the authors if interested in the prompts.

## Usage

From the root directory, run
```
python -m libsg.evaluation.main
```