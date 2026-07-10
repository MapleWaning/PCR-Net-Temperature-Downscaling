from pathlib import Path
import os


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("PCR_DATA_ROOT", ROOT / "data" / "processed")).resolve()

STANDARD_DATA_ROOT = DATA_ROOT / "standard"
PHYSICAL_DATA_ROOT = DATA_ROOT / "physical"
CATBOOST_INFERENCE_ROOT = DATA_ROOT / "catboost_inference"
MODEL_INPUT_ROOT = DATA_ROOT / "model_inputs"

STATION_META = STANDARD_DATA_ROOT / "metadata" / "high-quality-meta.csv"
TRAIN_META_CSV = MODEL_INPUT_ROOT / "splits" / "train_meta.csv"
VAL_META_CSV = MODEL_INPUT_ROOT / "splits" / "val_meta.csv"
TEST_META_CSV = MODEL_INPUT_ROOT / "splits" / "test_meta.csv"

STATIC_PATH = MODEL_INPUT_ROOT / "static.npy"
TRUTH_PATH = MODEL_INPUT_ROOT / "truth.npy"

TBASE_ADVANCE_DIR = PHYSICAL_DATA_ROOT / "t_base"
GFS_DIR = STANDARD_DATA_ROOT / "era5"
LST_PATH = STANDARD_DATA_ROOT / "lst"

CB_SPATIAL_BASE = CATBOOST_INFERENCE_ROOT / "catboost_lst"
CB_BASE = CATBOOST_INFERENCE_ROOT / "baseline_cb"
RF_BASE = CATBOOST_INFERENCE_ROOT / "baseline_rf"
PURE_RF_BASE = RF_BASE

FULL_YEAR = [2008]
TRAIN_YEAR = [2008]
VERIFY_YEAR = [2008]
TEST_YEAR = [2008]

EPOCHS = 1
BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
MAX_LR = 1e-3
MAX_NORM = 1.0
WEIGHT_DECAY = 1e-5
INPUT_CHANNELS = 20
OUTPUT_CHANNELS = 3
TEST_INPUT_CHANNELS = INPUT_CHANNELS
TEST_OUTPUT_CHANNELS = OUTPUT_CHANNELS
LAMBDA_GRAD = 0.1
ALPHA_TERRAIN = 0.1

MODEL_PATH = ROOT / "assets" / "pretrained" / "pcr_net" / "best_model.pth"
FINAL_MODEL_PATH = ROOT / "assets" / "pretrained" / "pcr_net" / "final_model.pth"
PICNUM = 2
EXPERIMENT = "pcr-net"
LABEL = EXPERIMENT
seed = 42
