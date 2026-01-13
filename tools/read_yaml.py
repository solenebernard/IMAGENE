import yaml
import socket
import numpy as np

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Identify the environment
hostname = socket.gethostname()
print(hostname)

if "maestro" in hostname:  # Adjust this for your cluster
    env = "cluster"
elif "imod" in hostname:
    env = "local_solene"
elif ("MacBook-Air-2" in hostname)or('arpa' in hostname):
    env = "local_solene_perso"
else:
    env = "local_isabella"

# Environment depending on which computer
ROOTPATH = config[env]["rootpath"]
if env=="local_solene":
    ROOTPATH_ATLAS = config[env]["rootpath_atlas"]
MODEL_PATH = config[env]['model_path']
DEVICE = config[env]["device"]
NUM_WORKERS = config[env]["num_workers"]

# Constants
SCALE = float(eval(config["constants"]["scale"]))
MAX_DIST_MATCHING = config["constants"]["max_dist"]
SIZE_RADIUS = config["constants"]["size_radius"]
BACKWARD = config["constants"]["backward"]

# Experiment parameters
EXP_PATH = config["experiment"]["exp_path"]
CONDITION = config["experiment"]["condition"]
GENES = config["experiment"]["genes"]
ROUNDS = config["experiment"]["rounds"]
N_FEATURES_DIM = config["experiment"]["n_features_dim"]

print(f"Running on {DEVICE}")