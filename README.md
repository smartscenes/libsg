# libSG: A Flexible Library for Scene Generation

libsg is a backend library for scene generation.

## Data setup

`libsg` was developed using the HSSD dataset. To setup your environment to use HSSD, follow the below steps:

1. Createa base directory for data and set `base_dir` in [conf/config.yaml] to your base directory.

2. Clone the [Floorplanner SceneBuilder (FPSB) repository](https://huggingface.co/datasets/3dlg-hcvc/fpsb) into `base_dir` to get the floorplanner scenes (this repo is currently private but will no longer be required soon.)

3. Clone the [fphab repository](https://huggingface.co/datasets/fpss/fphab) into `base_dir` to get the GLB objects used during retrieval.

## Local development

Use `conda` to setup the enviroment needed for running the flask app.

```bash
conda env create -f environment.yml
conda activate sb
```

## Local build

If you want to use `libsg` with other modules, install `libsg` locally via `pip`:

```bash
pip install --upgrade build
python -m build
pip install -e .
```

## Usage

Start the server using the following command (by default, the server runs at `localhost:5000`):
```bash
./start_server.sh
```

Use the `--background` option if you want the process to run detached from the shell.

### API examples

Retrieve a complete scene from the scene dataset:
```bash
curl localhost:5000/scene/retrieve/
```

Generate a new scene and write to `test.scene_instance.json` in STK format:
```bash
curl -X POST localhost:5000/scene/generate -H 'Content-Type: application/json' -d '{"type": "text", "input": "<scene generation prompt>", "format": "STK"}' -o test.scene_instance.json
```

The current API supports simple prompts that must mention the type of room you are looking to generate 
(e.g. `"bedroom"`, `"dining room"`, `"living room"`), and current supported scene formats include STK and HAB.


See [libsg/app.py] for the public-facing API and JSON payloads that can be used with above endpoints and [libsg/api.py] 
for the internal API, which exposes methods for easy use in downstream code.

## Packaging

Package `libsg` for deployment on PyPI (so other people can install via `pip install libsg`).
Below packaging follows guidelines at https://packaging.python.org/en/latest/tutorials/packaging-projects/
Generate tokens on pypi and store in `.pypirc` file as below:
```ini
[testpypi]
  username = __token__
  password = pypi-XXX
[pypi]
  username = __token__
  password = pypi-XXX
```

*NOTE*: uploads with a specific version are only allowed once.
Thus, be careful about current `version` tag in `pyproject.toml` file.

Deploying test package:
```bash
pip install --upgrade build twine
python -m build
python -m twine upload --repository testpypi dist/*
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps libsg
```

Deploying package to real pypi index is same as above except for much simpler upload and install commands:
```bash
python -m twine upload dist/*
pip install libsg
```
