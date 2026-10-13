import yaml
import socket
import numpy as np

# Load config: config_confidential.yaml (own paths, not tracked by git) if it exists, else config.yaml
import os
CONFIG_PATH = "config_confidential.yaml" if os.path.exists("config_confidential.yaml") else "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Identify the environment
hostname = socket.gethostname()
print(hostname)

if "maestro" in hostname:  # Adjust this for your cluster
    env = "cluster"
else:
    env = "local"

# Environment depending on which computer
ROOTPATH = config[env]["rootpath"]
ROOTPATH_ATLAS = config[env].get("rootpath_atlas")
MODEL_FINETUNE_PATH = config[env]['model_finetune_path']
DEVICE = config[env]["device"]
NUM_WORKERS = config[env]["num_workers"]

# Constants
SCALE = float(eval(config["constants"]["scale"]))
MAX_DIST_MATCHING = config["constants"]["max_dist"]
SIZE_RADIUS = config["constants"]["size_radius"]

# Experiment parameters
EXP_PATH = config["experiment"]["exp_path"]
CONDITION = config["experiment"]["condition"]
GENES = config["experiment"]["genes"]
ROUNDS = config["experiment"]["rounds"]
N_FEATURES_DIM = config["experiment"]["n_features_dim"]

# Step 1: brightfield (live) images
N_FRAMES = config["live"]["n_frames"]
LIVE_N_ROWS = config["live"]["n_rows"]
LIVE_N_COLS = config["live"]["n_cols"]
LIVE_N_TILES = LIVE_N_ROWS*LIVE_N_COLS
LIVE_TILE_SIZE = config["live"]["tile_size"]
LIVE_OVERLAP = config["live"]["overlap"]
LIVE_OVERLAP_PIXELS = int(LIVE_TILE_SIZE*LIVE_OVERLAP)
# Size of the merged image
LIVE_HEIGHT = (LIVE_TILE_SIZE-LIVE_OVERLAP_PIXELS)*(LIVE_N_ROWS-1) + LIVE_TILE_SIZE
LIVE_WIDTH = (LIVE_TILE_SIZE-LIVE_OVERLAP_PIXELS)*(LIVE_N_COLS-1) + LIVE_TILE_SIZE
Z_FINETUNE = config["live"]["z_finetune"]
BORDER_MARGIN = config["live"]["border_margin"]
IOU_THRESHOLD = config["live"]["iou_threshold"]
MAX_CENTROID_DISTANCE = config["live"]["max_centroid_distance"]
MIN_FRAMES = config["live"]["min_frames"]
MIN_CELL_AREA = config["live"]["min_cell_area"]

# Step 2: FISH images
FISH_N_TILES = config["fish"]["n_tiles"]
FISH_N_CHANNELS = config["fish"]["n_channels"]
FISH_TILE_SIZE = config["fish"]["tile_size"]
FISH_OVERLAP = config["fish"]["overlap"]
FISH_DEFAULT_PATTERN = config["fish"]["default_pattern"]
FISH_MIP_ROUNDS = config["fish"]["mip_rounds"]
VOXEL_SIZE_NM = tuple(config["fish"]["voxel_size_nm"])
OBJECT_RADIUS_NM = tuple(config["fish"]["object_radius_nm"])
CLUMPS_DETECTOR = config["fish"]["clumps_detector"]
CLUMPS_MIN_SIZE = config["fish"]["clumps_min_size"]

# Step 3: matching of BF and FISH
DAPI_MASK = config["matching"]["dapi_mask"]
TRANSLATION_N = config["matching"]["translation_n"]
TRANSLATION_DELTA = config["matching"]["translation_delta"]
TRANSLATION_DELTA_REFINED = config["matching"].get("translation_delta_refined", 300)

# Step 4: features
FOURIER_ORDER = config["features"]["fourier_order"]
FOURIER_N_POINTS = config["features"]["fourier_n_points"]
MSD_N_MAX = config["features"]["msd_n_max"]
NEIGHBORS_THRESHOLDS = config["features"]["neighbors_thresholds"]

# Step 5: learning
LEARNING_EXP_PATHS = config["learning"]["exp_paths"]
LEARNING_CONDITIONS = config["learning"]["conditions"]
LEARNING_CONDITIONS_GENE_PROFILE = config["learning"]["conditions_gene_profile"]
FILTER_ERROR = config["learning"]["filter_error"]
MAX_SPEED = config["learning"]["max_speed"]
ALPHA_MAX_STD_ERR = config["learning"]["alpha_max_std_err"]
LEARNING_GENES = config["learning"]["genes"]
GENE_THRESHOLDS = config["learning"]["gene_thresholds"]
RATIO_QC = config["learning"]["ratio_qc"]
N_SPLITS = config["learning"]["n_splits"]
N_ITER = config["learning"]["n_iter"]
N_EXTREME = config["learning"]["n_extreme"]
N_PERMUTATIONS = config["learning"]["n_permutations"]
N_BOOTSTRAP = config["learning"]["n_bootstrap"]
RANDOM_STATE = config["learning"]["random_state"]
MIN_MEAN_RNA = config["learning"].get("min_mean_rna", 3)
N_JOBS = config["learning"].get("n_jobs", -1)
PARAM_DIST = config["learning"]["param_dist"]

print(f"Running on {DEVICE}")
