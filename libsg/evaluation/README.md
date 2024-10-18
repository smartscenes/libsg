# Evaluation

Code to evaluate scene generation methods.

## Data

You will need to create a CSV file at `.data/evaluation/test_prompts.csv` containing the list of prompts you
want to test. The format of the CSV is as follows:
```
prompt,hasArchitecture,hasOpening,hasEntity,hasAttribute,hasRelationship,num_iterations,test_diversity,word_count,length
"Generate a small bedroom with a twin-sized bed and a wardrobe.",bedroom,,"bed - quantityExact - 1;\nwardrobe - quantityExact - 1","bedroom - small;\nbed - twin-sized",,1,FALSE,11,Short
```
* `prompt` - text description of scene
* `hasArchitecture` - a comma-delimited list of rooms which should be present in the architecture (e.g. `bedroom, closet, bathroom`)
* `hasOpening` - a semicolon-delimited list of constraints of openings, of the form `<opening> - quantity<QuantityType> - <value> - <location_relationship> - <location_subject>[ - <location_object>]`
* `hasEntity` - a semicolon-delimited list of constraints of objects in the scene, of the form `<objeect> - quantity<QuantityType> - <value>`.
* `hasAttribute` - a semicolon-delimited list of constraints of attributes of objects, of the form `<subject> - <attribute>`
* `hasRelationship` - a semicolon-delimited list of constraints of relationships between objects, of the form `<subject>[<index>] - <relationship> - <object>[<index>]`
* `num_iterations` - number of iterations to run each prompt
* `test_diversity` - if `True`, evaluates prompts using diversity metrics only, else if `False`, evaluates prompts using aggregate metrics
* `word_count` - number of words in prompt
* `length` - "Short" (<25 words), "Medium" (25-200 words), or "Long" (>200 words)

As the test prompts we are using have not been finalized yet, please contact the authors if interested in the prompts.

## Usage

### Inference

```bash
python -m libsg.evaluation.inference
```

### Evaluation 

From the root directory, run
```bash
python -m libsg.evaluation.main
```

### Rendering

```bash
python -m libsg.evaluation.generate_renders
```