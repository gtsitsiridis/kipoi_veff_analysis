# Variant effect prediction using Kipoi

## Setup:

### Create conda environment
```bash
mamba env create -f envs/conda-env.yml -n kipoi-enformer-<your_name>
conda activate kipoi-enformer-<your_name>
pip install -e .
```

### Convert percent scripts to jupyter notebooks
```bash
find workflow/notebooks/ -iname "*[.py.py|.R.R]" -exec jupytext --sync {} \;
```

### Pytest
To run the tests, execute the following command:
```bash
pytest tests/
````

### Snakemake workflow
#### Local profile
To run the snakemake workflow locally using the dev config, execute the following command:
```bash
snakemake --workflow-profile=workflow/profiles/dev
```

#### Slurm profile
To run the snakemake workflow on a slurm cluster using the production config, execute the following command:
```bash
conda activate kipoi-enformer-<your_name>
CONDA_OVERRIDE_CUDA="11.8" snakemake --workflow-profile=workflow/profiles/prod
```
