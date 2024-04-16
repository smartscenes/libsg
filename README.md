# libSG: A Flexible Library for Scene Generation

`libsg` is a backend library for scene generation, intended to be used with the scene toolkit frontend.

## Package Contents

The folder structure breaks down as follows (grouped in accordance with function):
```
conf  # configuration files for libsg and models
├── layout_generator  # layout generation model configurations
│   ├── layout_mapping.yaml  # defines general layout parameters and mapping from model and room type to configuration
├── config.yaml  # general libsg config
libsg  # main code
│   ├── api.py                    # main internal API for handling requests within libsg
│   ├── app.py                    # main external API for handling requests from stk or other sources. Light wrapper
                                    around api.py.
│   ├── scene_builder.py          # main class for generating or manipulating scenes, which calls all downstream modules
│   ├── scene_parser.py           # main interface class for handling parsing of scene description
│   ├── arch_builder.py           # code for retrieving or generating architecture of scene
│   ├── object_placement.py       # main class for handling selection and placement of objects
│   ├── io.py                     # main class for defining export to scene formats
│   ├── arch.py                   # main class defining internal data representation for architectures
│   ├── scene.py                  # main class defining internal data representation for scenes
│   ├── scene_types.py            # an array of helper classes to define object types and specification classes
│   ├── model                     # model code for text parsing, layout generation, shape generation, etc.
│   │   ├── atiss.py              # model code for ATISS
│   │   ├── diffuscene            # model code for DiffuScene
│   │   ├── instructscene         # model code for InstructScene
│   │   ├── sg_parser             # code for scene graph parsing
│   │   │   ├── base.py           # base class for scene graph (SG) parsing
│   │   │   ├── instructscene.py  # SG parsing based on InstructScene method
│   │   │   ├── llm_sg_parser.py  # SG parsing based on LLM
│   │   │   ├── room_type.py      # SG parsing for simple room type lookup
│   │   ├── layout.py             # main interface code for calling layout models, incl. pre- and post-processing of outputs
│   │   ├── utils.py              # utility functions for models
│   ├── assets.py                 # database class for managing and retrieving assets
│   ├── simscene.py               # class for handling simulation of a scene, e.g. for object placement collision detection
│   ├── simulator.py              # class for base method simulation code
│   ├── config.py                 # code to load main configuration
│   ├── geo.py                    # auxiliary code to handle object transforms
```

## Data setup

`libsg` was developed using the HSSD dataset. To setup your environment to use HSSD, follow the below steps:

1. Create a base directory for data and set `base_dir` in [conf/config.yaml] to your base directory.

2. To use the structured3d dataset of room architectures, you must link the [configuration](conf/config.yaml) to the 
structured3d.rooms.csv file, which can be retrieved at present 
[here](https://github.com/smartscenes/sstk-metadata/blob/master/data/structured3d/structured3d.rooms.csv). 
Modify the `arch_db` parameters to link to the CSV for scene lookup.

3. (Optional) Clone the [hssd-hab repository](https://huggingface.co/datasets/hssd/hssd-hab) into `base_dir` to get the GLB objects 
used during retrieval to check object collision detection during placement.

## Local development

Use `conda` to setup the enviroment needed for running the flask app.

```bash
conda env create -f environment.yml
conda activate sb
```

Alternatively, if you want to use `libsg` with other modules, install `libsg` locally via `pip`:

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

The current API, by default, supports simple prompts that must mention the type of room you are looking to generate 
(e.g. `"bedroom"`, `"dining room"`, `"living room"`). For instance, to generate a bedroom, run
```bash
curl -X POST localhost:5000/scene/generate -H 'Content-Type: application/json' -d '{"type": "text", "input": "Generate a bedroom", "format": "STK"}' -o test.scene_instance.json
```

Current supported scene formats include STK and HAB.

You can additionally pass configuration options to the API to customize the backend generation of the scene:
* `sceneInference.parserModel` - specifies the model to use for scene graph parsing (`InstructScene`, `LLM`, `RoomType`)
* `sceneInference.layoutModel` - specifies the model to use for layout generation (`ATISS`, `DiffuScene`, `InstructScene`)
* `sceneInference.passTextToLayout` - if True, the code will attempt to pass the raw text input to the model. Currently
  only applicable to `DiffuScene`. `ATISS` does not condition on text, and `InstructScene` uses the text by default currently.
* `sceneInference.object.genMethod` - specify generation method for objects (`generate`, `retrieve`)
* `sceneInference.object.retrieveType` - specify retrieval method for objects (`category`, `embedding`)

For instance, to specify in the request that you would like to generate scenes using DiffuScene with the raw text input, 
run
```bash
curl -X POST localhost:5000/scene/generate -H 'Content-Type: application/json' -d '{"type": "text", "input": "Generate a dining room with a table and four chairs around it.", "format": "STK", "config": {"sceneInference.layoutModel": "DiffuScene", "sceneInference.passTextToLayout": "True"}}' -o test.scene_instance.json
```

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
