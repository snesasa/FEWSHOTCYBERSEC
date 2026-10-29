"""
config.py  —  Member 2: Few-Shot Embedding & Metric Learning
All hyperparameters and paths in one place.
Change values here; every other file imports from here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root
HANDOFF_DIR = os.path.join(BASE_DIR, "dataset", "member2_handoff")
HOLDOUT_DIR = os.path.join(BASE_DIR, "dataset", "unseen_family_holdout")
OUTPUT_DIR  = os.path.join(BASE_DIR, "member2_work", "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOT_DIR       = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR    = os.path.join(OUTPUT_DIR, "results")

# Create output dirs automatically
for _d in [OUTPUT_DIR, CHECKPOINT_DIR, PLOT_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Data files from Member 1 ───────────────────────────────────────────────────
X_TRAIN_PATH      = os.path.join(HANDOFF_DIR, "X_train.npy")
Y_TRAIN_PATH      = os.path.join(HANDOFF_DIR, "y_train.npy")
X_TEST_PATH       = os.path.join(HANDOFF_DIR, "X_test.npy")
Y_TEST_PATH       = os.path.join(HANDOFF_DIR, "y_test.npy")
PREPROCESSOR_PATH = os.path.join(HANDOFF_DIR, "feature_preprocessor.joblib")
SELECTOR_PATH     = os.path.join(HANDOFF_DIR, "anova_selector.joblib")
FEATURES_PATH     = os.path.join(HANDOFF_DIR, "selected_feature_names.csv")

# ── Holdout / unseen families ──────────────────────────────────────────────────
WITHHELD_NAMES_PATH    = os.path.join(HOLDOUT_DIR, "withheld_family_names.csv")
UNSEEN_RAW_FEAT_PATH   = os.path.join(HOLDOUT_DIR, "unseen_family_raw_features.csv")
UNSEEN_LABELS_PATH     = os.path.join(HOLDOUT_DIR, "unseen_family_labels.csv")
FAMILY_SPLIT_PATH      = os.path.join(HOLDOUT_DIR, "family_sample_counts_and_split.csv")

# ── Model Architecture ─────────────────────────────────────────────────────────
INPUT_DIM       = 100      # number of features from Member 1 (confirmed from data inspection)
HIDDEN_DIMS     = [512, 256, 128]  # 3-layer: more capacity
EMBEDDING_DIM   = 64              # final embedding size (L2-normalized)
DROPOUT_RATE    = 0.35            # increased to regularize train acc into 85-92% range

# ── Episodic Training ──────────────────────────────────────────────────────────
# Families with fewer than this are excluded from episodic training
# (need enough samples for support + query across many episodes)
MIN_SAMPLES_PER_FAMILY = 10

N_WAY       = 10   # HARDER training: 10 classes per episode forces better discrimination
K_SHOT      = 5    # support examples per class (varied in experiments)
N_QUERY     = 15   # more query samples → stronger gradient signal per episode
N_EPISODES  = 500  # episodes per epoch
N_EPOCHS    = 60   # more epochs for the deeper model

# ── Feature Augmentation ──────────────────────────────────────────────────────
FEATURE_JITTER_STD  = 0.05   # Gaussian noise std added to training features
TEMPERATURE         = 8.0    # logit sharpening in prototypical loss (lower=softer=less overfit)
LABEL_SMOOTHING     = 0.25   # reduces train acc into 85-92% zone, prevents overconfidence

# ── Optimizer ─────────────────────────────────────────────────────────────────
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
LR_SCHEDULER    = "cosine"  # "cosine" or "step"
LR_STEP_SIZE    = 20        # used only if LR_SCHEDULER = "step"
LR_GAMMA        = 0.5       # used only if LR_SCHEDULER = "step"
LR_T_MAX        = N_EPOCHS  # cosine annealing period (full training)

# ── Evaluation Shots ──────────────────────────────────────────────────────────
# These are the 4 experiments required by the role assignment
EVAL_SHOTS = [1, 5, 10, 20]
EVAL_N_WAY = 5
EVAL_EPISODES = 600   # evaluation episodes per shot setting

# ── Random Seed ───────────────────────────────────────────────────────────────
SEED = 42

# ── Checkpoint filenames ──────────────────────────────────────────────────────
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_embedding_model.pt")
FINAL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "final_embedding_model.pt")
