# libSG: A Flexible Library for Scene Generation

## Local dev:

Use `conda` to setup the enviroment needed for running the flask app.

Set local dataset `base_dir` path in `conf/config.yaml`.
Fetch stk/scenestate data and place under your `base_dir`.

```bash
conda env create -f environment.yml
conda activate sb
```

## Local build:
Install `libsg` locally via `pip` so that it is available for other modules to use.

Note: first two commands may not be needed.

```bash
pip install --upgrade build
python -m build
pip install -e .
```

## Running Flask App and Usage Examples

Start Flask App (by default runs server at `localhost:5000`):
```bash
flask --app libsg.app --debug run
```

Retrieve a complete scene from the scene dataset:
```bash
curl localhost:5000/scene/retrieve/
```

Generate a new scene and write to `test.scene_instance.json` in Habitat format:
```bash
curl localhost:5000/scene/generate -o test.scene_instance.json
```

Add an object matching specified category at specified position:
```bash
curl -X POST -H 'Content-Type:application/json' -d @object_add_category_position.json http://127.0.0.1:5000/object/add -o object_add_category_position_output.json
```

Remove objects matching specified category from the scene:
```bash
curl -X POST -H 'Content-Type:application/json' -d @object_remove_category.json http://127.0.0.1:5000/object/remove -o object_remove_category_output.json
```

See libsg/api.py for examples of JSON payloads that can be used with above endpoints.

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
