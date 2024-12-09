import json
import os

from hydra import compose, initialize

if os.path.exists("overrides.json"):
    with open("overrides.json", "r") as f:
        overrides_raw = json.load(f)
overrides = [f"{k}={v}" for k, v in overrides_raw.items()]

initialize(version_base=None, config_path='../conf')
config = compose('config.yaml', overrides=overrides)
# print(OmegaConf.to_container(config))

