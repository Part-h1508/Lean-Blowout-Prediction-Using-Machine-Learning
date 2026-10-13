from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "lbo_prediction" / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

TIME_COLUMN = 0
SIGNAL_COLUMN = 1
N_STATES = 4
N_ITERATIONS = 100
RANDOM_STATE = 42
SAMPLING_TOLERANCE = 0.05
MAX_ACF_LAG = 500
PSD_SEGMENT_LENGTH = 1024