# libSG: A Flexible Library for Scene Generation

`libsg` is a backend library for scene generation, intended to be used with the scene toolkit frontend.

To setup your own instance of `libsg`, please follow the data and configuration setup sections below.

## Package Contents

The folder structure breaks down as follows (grouped in accordance with function):
```
conf  # configuration files for libsg and models
├── arch_generator                    # layout generation model configurations
│   ├── arch_generator_mapping.yaml   # defines arch gen parameters and model-to-config mapping
├── layout_generator                  # layout generation model configurations
│   ├── layout_mapping.yaml           # defines layout parameters and model-to-config mapping
├── scene_parser                      # scene praser configurations   
│   ├── parser_mapping.yaml           # defines scene parser configurations
├── config.yaml                       # main libsg config
├── evaluation.yaml                   # main libsg config for evaluation script (libsg/evaluation/main.py)
├── inference.yaml                    # main libsg config for inference script (libsg/evaluation/inference.py)
libsg  # main code
│   ├── api.py                        # main internal API for handling requests within libsg
│   ├── app.py                        # main external API for handling requests from STK or other sources. Light wrapper
                                        around api.py.
│   ├── scene_builder.py              # main class for generating or manipulating scenes, which calls all downstream modules
│   ├── scene_parser.py               # main interface for parsing scene description 
│   ├── arch_builder.py               # code for retrieving and generating scene architecture
│   ├── object_placement.py           # main class for handling objects selection and placement
│   ├── io.py                         # main class for scene format export
│   ├── arch.py                       # main class for defining internal architectures representation 
│   ├── scene.py                      # main class for defining internal scene representation
│   ├── scene_types.py                # helper classes for object types and specifications
│   ├── model                         # model code for text parsing, layout generation, shape generation, etc.
│   │   ├── atiss.py                  # ATISS model implementaion
│   │   ├── diffuscene                # DiffuScene model implementation
│   │   ├── instructscene             # InstructScene model implmentation
│   │   ├── arch_generator            # code for architecture generators
│   │   │   ├── base.py               # base class for architecture generators
│   │   │   ├── square.py             # class for generating simple square room
│   │   ├── sg_parser                 # code for scene graph parsing
│   │   │   ├── base.py               # base class for scene graph (SG) parsing
│   │   │   ├── instructscene.py      # InstructScene-based SG parsing
│   │   │   ├── llm_sg_parser.py      # LLM-based SG parsing 
│   │   │   ├── room_type.py          # simple room type lookup
│   │   ├── layout.py                 # main interface code for calling layout models, incl. pre- and post-processing of outputs
│   │   ├── utils.py                  # utility functions for models
│   ├── assets.py                     # database class for managing and retrieving assets
│   ├── simscene.py                   # class for scene simulation, e.g. for object placement collision detection
│   ├── simulator.py                  # class for base method simulation code
│   ├── config.py                     # code to load main configuration
│   ├── geo.py                        # auxiliary code for object transforms
│   ├── evaluation                    # code for defining evaluation script and metrics for generated scenes
│   │   ├── metrics                   # definitions of scene generation metrics for evaluation
│   │   ├── inference.py              # inference code for running scene generation at scale for a method (no metrics)
│   │   ├── main.py                   # evaluation code for using a single config to generate scenes and compute metrics
│   │   ├── utils.py                  # utility functions for evaluation

```

Key components:

1. **Configuration** (`conf/`): Contains all configuration files for the library and its models.

2. **Main Library** (`libsg/`):
   - API layers (`api.py`, `app.py`): Handle internal and external requests.
   - Core functionality (`scene_builder.py`, `scene_parser.py`, `arch_builder.py`, `object_placement.py`): Main classes for scene generation and manipulation.
   - Data representation (`arch.py`, `scene.py`, `scene_types.py`): Define internal data structures.
   - I/O handling (`io.py`): Manages import/export of scenes.

3. **Models** (`libsg/model/`):
   - Implements various scene generation models (ATISS, DiffuScene, InstructScene).
   - Scene graph parsing models in `sg_parser/`.
   - `layout.py` provides a unified interface for all layout models.

4. **Asset Management** (`assets.py`): Handles database operations for scene assets.

5. **Simulation** (`simscene.py`, `simulator.py`): Provides simulation capabilities, particularly for object placement and collision detection.

6. **Utilities** (`config.py`, `geo.py`): Helper functions and configuration management.

## Data setup

`libsg` was developed using the HSSD dataset. To setup your environment to use HSSD, follow the below steps:

1. Create a base directory for data and set `base_dir` in [conf/config.yaml] to your base directory.

2. To use the structured3d dataset of room architectures, you must link the [configuration](conf/config.yaml) to the 
structured3d.rooms.csv file, which can be retrieved at present 
[here](https://github.com/smartscenes/sstk-metadata/blob/master/data/structured3d/structured3d.rooms.csv). 
Modify the `arch_db` parameters to link to the CSV for scene lookup.

3. (Optional) Clone the [hssd-hab repository](https://huggingface.co/datasets/hssd/hssd-hab) into `base_dir` to get the GLB objects 
used during retrieval to check object collision detection during placement.

## Configuration Setup

Several of the paths in the configuration files ([conf/config.yaml] and [conf/evaluation.yaml]) must be specified before
use. You will need to specify all those defined as `???`, including
* `base_url`
* `structured3d_path`
* `scene_builder.model_db.generation.output_dir`
* `scene_builder.model_db.generation.metadata_file`
* `scene_builder.solr_url`

Further information about these paths can be found in the configuration files themselves.

## Local development

Requires `CUDA 12.1` for Shap-e GPU acceleraction \
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

Afterwards, run
```bash
python3 -c "import nltk; nltk.download('cmudict')"
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
* `sceneInference.assetSources` - specify asset sources for retrieval. If not specified, all sources are used (e.g. `3dfModel,fpModel`)
* `sceneInference.arch.genMethod` - specify generation method for architecture (`generate`, `retrieve` (default))
* `sceneInference.arch.genModel` - specify model for arch generation (`SquareRoomGenerator`)
* `sceneInference.retrievalSources` - specify asset sources for retrieval. If not specified, all sources are used (e.g. `3dfModel,fpModel`)
* `sceneInference.useCategory` - if True, code will enforce usage of wnsynset key to retrieve objects in addition to embeddings or other metadata. Default: false (`3dfModel`s do not have wnsynset keys)

Examples:
```bash
# basic ATISS bedroom with a square floor plan (no walls)
curl -X POST localhost:5000/scene/generate -H 'Content-Type: application/json' -d '{"type": "text", "input": "Generate a bedroom", "format": "STK", "config": {"sceneInference.parserModel": "RoomType", "sceneInference.arch.genMethod": "generate", "sceneInference.arch.genModel": "SquareRoomGenerator", "sceneInference.layoutModel": "ATISS", "sceneInference.object.genMethod": "retrieve", "sceneInference.object.retrieveType": "category", "sceneInference.retrievalSources": "fpModel", "sceneInference.useCategory": "true"}}' -o test.scene_instance.json

# DiffuScene using raw text input
curl -X POST localhost:5000/scene/generate -H 'Content-Type: application/json' -d '{"type": "text", "input": "Generate a dining room with a table and four chairs around it.", "format": "STK", "config": {"sceneInference.layoutModel": "DiffuScene", "sceneInference.passTextToLayout": "True"}}' -o test.scene_instance.json

# InstructScene with embedding retrieval from fpModel and 3dfModel
curl -X POST localhost:5000/scene/generate -H 'Content-Type: application/json' -d '{"type": "text", "input": "Generate a bedroom", "format": "STK", "config": {"sceneInference.parserModel": "InstructScene", "sceneInference.arch.genMethod": "retrieve", "sceneInference.layoutModel": "InstructScene", "sceneInference.object.genMethod": "retrieve", "sceneInference.object.retrieveType": "embedding", "sceneInference.retrievalSources": "fpModel,3dfModel", "sceneInference.useCategory": "false"}}' -o test.scene_instance.json
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
