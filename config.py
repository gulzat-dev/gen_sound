# config.py
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).parent
DATA_RAW_DIR = BASE_DIR / "datasets"
DATA_PROCESSED_DIR = BASE_DIR / "data_processed"
PLOTS_DIR = BASE_DIR / "plots_alertness"
MODELS_DIR = BASE_DIR / "models_alertness"
FEATURES_DIR = DATA_PROCESSED_DIR / "features_alertness"
RESULTS_FILE = BASE_DIR / "project_summary_results.txt"

# --- Modular Audio Directories ---
BASE_SEGMENTS_DIR = DATA_PROCESSED_DIR / "audio_segments_base"
FINAL_AUDIO_DIR = DATA_PROCESSED_DIR / "audio_segments_final" # Input for feature extraction

# --- Audio Parameters ---
# Using 22.05 kHz is a standard for audio analysis, capturing the full range
# of human hearing relevant for environmental sounds (~20 kHz).
SAMPLE_RATE = 22050

# A 5-second duration allows capturing longer events like sirens or ambient noise
# more effectively than a shorter duration. This matches the paper's methodology.
DURATION_S = 5.0

# Standard FFT window size for good frequency resolution.
N_FFT = 2048

# A hop length of 512 provides a good overlap (~75%) between windows,
# which is standard for capturing temporal changes smoothly.
HOP_LENGTH_FFT = 512

# --- Feature Parameters ---
# Using 128 Mel bands provides a richer, more detailed frequency representation
# for the CNN to learn from, compared to a lower number like 64. This is a common
# choice for deep learning models.
N_MELS = 128

# N_MFCC is used for the classic models (SVM, RF). 40 MFCCs is a robust choice,
# often cited in literature, balancing detail with dimensionality.
# (The paper mentions 40 MFCCs, which is a better choice than the old 20).
N_MFCC = 40

# # --- Data Mapping ---
# SOURCE_TO_PROXY_MAP = {
#     "proxy_joy_excitement": {
#         "esc50": ["laughing", "fireworks", "clapping", "cheering", "dog"],
#         "urbansound8k": ["children_playing"]
#     },
#     "proxy_alert_stress": {
#         "esc50": ["siren", "car_horn", "chainsaw", "helicopter", "engine", "clock_alarm", "glass_breaking", "hand_saw"],
#         "urbansound8k": ["siren", "car_horn", "gun_shot", "jackhammer", "drilling", "dog_bark"]
#     },
#     "proxy_distress_discomfort": {
#         "esc50": ["crying_baby", "sneezing", "coughing", "breathing", "snoring"],
#         "urbansound8k": []
#     },
#     "proxy_ambient_neutral": {
#         "esc50": ["wind", "rain", "sea_waves", "crickets", "chirping_birds", "footsteps", "clock_tick", "pouring_water", "washing_machine", "vacuum_cleaner", "rooster"],
#         "urbansound8k": ["air_conditioner", "street_music", "engine_idling"]
#     }
# }

# --- Comprehensive Data Mapping (All 60 Classes) ---
# Each class from the ESC-50 and UrbanSound8K datasets is mapped to a semantic proxy class
# based on its prototypical urgency and emotional valence.

SOURCE_TO_PROXY_MAP = {
    "proxy_alert_stress": {
        "esc50": [
            "siren",
            "car_horn",
            "chainsaw",
            "helicopter",
            "engine",
            "train",
            "glass_breaking",
            "thunderstorm"
        ],
        "urbansound8k": [
            "siren",
            "car_horn",
            "gun_shot",
            "jackhammer",
            "drilling",
            "dog_bark"
        ]
    },
    "proxy_joy_excitement": {
        "esc50": [
            "laughing",
            "fireworks",
            "clapping",
            "cheering",
            "dog",
            "church_bells"
        ],
        "urbansound8k": [
            "children_playing"
        ]
    },
    "proxy_distress_discomfort": {
        "esc50": [
            "crying_baby",
            "sneezing",
            "coughing",
            "breathing",
            "snoring",
            "hand_saw",
            "door_wood_creaks",
            "door_wood_knock",
            "pig",
            "cow",
            "hen",
            "cat",
            "sheep"
        ],
        "urbansound8k": []
    },
    "proxy_ambient_neutral": {
        "esc50": [
            "wind",
            "rain",
            "sea_waves",
            "crickets",
            "chirping_birds",
            "rooster",
            "footsteps",
            "clock_tick",
            "pouring_water",
            "washing_machine",
            "vacuum_cleaner",
            "can_opening",
            "toilet_flush",
            "brushing_teeth",
            "drinking_sipping",
            "keyboard_typing",
            "mouse_click",
            "crackling_fire",
            "water_drops",
            "frog",
            "insects",
            "crow",
            "airplane"
        ],
        "urbansound8k": [
            "air_conditioner",
            "street_music",
            "engine_idling"
        ]
    }
}

ESC50_MAP = {label: proxy for proxy, sources in SOURCE_TO_PROXY_MAP.items() for label in sources.get("esc50", [])}
URBANSOUND8K_MAP = {label: proxy for proxy, sources in SOURCE_TO_PROXY_MAP.items() for label in sources.get("urbansound8k", [])}

# --- Model Training ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3

# --- CNN ---
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001