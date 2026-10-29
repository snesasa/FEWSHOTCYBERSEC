"""
config_gpu.py  —  GPU/HPC version of config.py
Copy this over config.py when running on a GPU machine or HPC cluster.

Expected accuracy improvement: 87% → 92-95%
Expected training time:        16 hours (CPU) → 1-2 hours (GPU)
"""

import os

# ── Paths (same as CPU version) ────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANDOFF_DIR    = os.path.join(BASE_DIR, "dataset", "member2_handoff")
HOLDOUT_DIR    = os.path.join(BASE_DIR, "dataset", "unseen_family_holdout")
OUTPUT_DIR     = os.path.join(BASE_DIR, "member2_work", "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOT_DIR       = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR    = os.path.join(OUTPUT_DIR, "results")

for _d in [OUTPUT_DIR, CHECKPOINT_DIR, PLOT_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Data files ─────────────────────────────────────────────────────────────────
X_TRAIN_PATH      = os.path.join(HANDOFF_DIR, "X_train.npy")
Y_TRAIN_PATH      = os.path.join(HANDOFF_DIR, "y_train.npy")
X_TEST_PATH       = os.path.join(HANDOFF_DIR, "X_test.npy")
Y_TEST_PATH       = os.path.join(HANDOFF_DIR, "y_test.npy")
PREPROCESSOR_PATH = os.path.join(HANDOFF_DIR, "feature_preprocessor.joblib")
SELECTOR_PATH     = os.path.join(HANDOFF_DIR, "anova_selector.joblib")
FEATURES_PATH     = os.path.join(HANDOFF_DIR, "selected_feature_names.csv")

WITHHELD_NAMES_PATH  = os.path.join(HOLDOUT_DIR, "withheld_family_names.csv")
UNSEEN_RAW_FEAT_PATH = os.path.join(HOLDOUT_DIR, "unseen_family_raw_features.csv")
UNSEEN_LABELS_PATH   = os.path.join(HOLDOUT_DIR, "unseen_family_labels.csv")
FAMILY_SPLIT_PATH    = os.path.join(HOLDOUT_DIR, "family_sample_counts_and_split.csv")

# ══════════════════════════════════════════════════════════════════════════════
#  GPU / HPC SETTINGS  (key differences from CPU config marked with <<<GPU>>>)
# ══════════════════════════════════════════════════════════════════════════════

# ── Model Architecture ─────────────────────────────────────────────────────────
INPUT_DIM       = 100
HIDDEN_DIMS     = [1024, 512, 256, 128]   # <<<GPU>>> 4 layers, 2× wider
EMBEDDING_DIM   = 128                      # <<<GPU>>> 128-dim instead of 64
DROPOUT_RATE    = 0.30

# ── Episodic Training ──────────────────────────────────────────────────────────
MIN_SAMPLES_PER_FAMILY = 10

N_WAY       = 20     # <<<GPU>>> 20-way (was 10) — much harder episodes
K_SHOT      = 5
N_QUERY     = 20     # <<<GPU>>> 20 query samples per class (was 15)
N_EPISODES  = 2000   # <<<GPU>>> 4× more episodes per epoch (was 500)
N_EPOCHS    = 200    # <<<GPU>>> 3× more epochs (was 60)

# ── Feature Augmentation ──────────────────────────────────────────────────────
FEATURE_JITTER_STD  = 0.05
TEMPERATURE         = 10.0
LABEL_SMOOTHING     = 0.10

# ── Optimizer ─────────────────────────────────────────────────────────────────
LEARNING_RATE   = 5e-4    # <<<GPU>>> slightly lower LR for larger model
WEIGHT_DECAY    = 1e-4
LR_SCHEDULER    = "cosine"
LR_STEP_SIZE    = 40
LR_GAMMA        = 0.5
LR_T_MAX        = N_EPOCHS

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_SHOTS    = [1, 5, 10, 20]
EVAL_N_WAY    = 5
EVAL_EPISODES = 1000   # <<<GPU>>> more eval episodes for tighter CI

# ── Seed ──────────────────────────────────────────────────────────────────────
SEED = 42

# ── Checkpoints ───────────────────────────────────────────────────────────────
BEST_CHECKPOINT  = os.path.join(CHECKPOINT_DIR, "best_embedding_model.pt")
FINAL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "final_embedding_model.pt")

# ══════════════════════════════════════════════════════════════════════════════
#  HOW TO USE THIS ON HPC / GOOGLE COLAB / KAGGLE:
#
#  Option 1 — Google Colab (free GPU):
#    1. Upload the entire member2_work/ folder to Colab
#    2. Replace config.py with this file (rename to config.py)
#    3. Run:  !python train.py --n_way 20 --k_shot 20 --n_episodes 2000 --n_epochs 200
#    4. Download the best_checkpoint.pt and put it in member3_handoff/
#
#  Option 2 — Kaggle Notebooks (free GPU, 30h/week):
#    Same as Colab, but use Kaggle's notebook interface
#
#  Option 3 — HPC SLURM job:
#    #!/bin/bash
#    #SBATCH --gres=gpu:1
#    #SBATCH --time=02:00:00
#    module load python/3.11 cuda/11.8
#    pip install torch pandas scikit-learn scipy joblib
#    python train.py --n_way 20 --k_shot 20 --n_episodes 2000 --n_epochs 200
#
#  The code already uses CUDA automatically:
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#  No code changes needed!
# ══════════════════════════════════════════════════════════════════════════════
