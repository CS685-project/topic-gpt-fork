## Setup
- Install the requirements: `pip install -r requirements.txt`
- Add togetherAI key into env file

## Data
- Prepare your `.jsonl` data file in the following format:
    ```
    {
        "id": "Optional IDs",
        "text": "Documents",
        "label": "Optional ground-truth labels"
    }
    ```
- Put the data file in `data/input`. There is also a sample data file `data/input/sample.jsonl` to debug the code.

## Pipeline
- For every generation objective, create a folder `generation_objective` in `./prompt` which contains `generation_1.txt` and `seed_1.md`. Modify the prompt and seed topics according to your needs.
- For every generation objective, create a folder `generation_objective` in `./data/[model_name]\output`
- Run `./script/example.ipynb`


## Results
We ran experiments on 3 objectives: `innovation_type`, `transport_innovation` and `Y02E_electrical_components`. \\
The files in `./data/output/[model_name]/[generation_objective]` are:
- `generation_1.md` contains the list of topics generated after the topic generation step
- `generation_1_manual_refinement.md` contains the list of topics after manual refinement
- `assignment_[model_name]` contains the output from assignment step using the respective models

