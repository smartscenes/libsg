# from omegaconf import OmegaConf
from hydra import compose, initialize

initialize(version_base=None, config_path='../conf')
config = compose('config.yaml')
# print(OmegaConf.to_container(config))