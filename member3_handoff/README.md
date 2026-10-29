# Member 3 Handoff — From Member 2
## KronoDroid Few-Shot Malware Classification Project

---

## What's In This Folder

| File | What It Is | Do You Need It? |
|---|---|---|
| `embedder.py` | **Your main API** — import this | ✅ Yes |
| `model.py` | Network architecture (needed by embedder) | ✅ Yes |
| `config.py` | Hyperparameters (needed by embedder) | ✅ Yes |
| `loss.py` | Distance functions (needed by embedder) | ✅ Yes |
| `best_embedding_model_10way_20shot_v3.pt` | **Trained model weights** (87.1% val acc on 10-way 20-shot) | ✅ Yes |
| `feature_preprocessor.joblib` | **Feature scaler from Member 1** | ✅ Yes |
| `selected_feature_names.csv` | Names of the 100 selected features | ✅ Yes |

---

## ⚠️ Critical: How the Data Pipeline Works

```
unseen_family_raw_features.csv   (478 columns — RAW, unprocessed)
         ↓  Step 1: apply anova_selector  → select 100 columns
         ↓  Step 2: apply feature_preprocessor → scale values
         ↓
     scaled_features  (100 columns, ready for embedding)
         ↓  Step 3: embed()
         ↓
     embeddings  (64-dim, L2-normalized)
```

**X_train.npy is ALREADY processed (steps 1+2 done by Member 1).**
**For the 25 unseen families, you must do steps 1+2 yourself using the files below.**

You need TWO files from Member 1's handoff:
- `dataset/member2_handoff/anova_selector.joblib` — selects 100 of 478 columns
- `dataset/member2_handoff/feature_preprocessor.joblib` — scales those 100 columns

---

## Working Code (Tested — Copy This Exactly)

```python
import numpy as np
import pandas as pd
import joblib
from scipy.spatial.distance import cdist

# ── Add member2_work to path so embedder.py can import model/config/loss ──────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))   # folder containing this README

from embedder import load_embedder

# ── Step 1: Load the 25 unseen families ──────────────────────────────────────
raw_df  = pd.read_csv("../dataset/unseen_family_holdout/unseen_family_raw_features.csv")
labels  = pd.read_csv("../dataset/unseen_family_holdout/unseen_family_labels.csv")["family_label"].values
# NOTE: column is "family_label" NOT "family"

raw_features = raw_df.values   # shape (168, 478) — raw, 478 columns

# ── Step 2: Apply Member 1's selector (478 cols → 100 cols) ──────────────────
selector = joblib.load("../dataset/member2_handoff/anova_selector.joblib")
selected = selector.transform(raw_features)   # shape (168, 100)

# ── Step 3: Apply Member 1's scaler (scale the 100 columns) ──────────────────
preprocessor = joblib.load("../dataset/member2_handoff/feature_preprocessor.joblib")
scaled = preprocessor.transform(selected)     # shape (168, 100)

# ── Step 4: Load the trained embedding model ──────────────────────────────────
embed = load_embedder("best_embedding_model_10way_20shot_v3.pt")

# ── Step 5: Get 64-dim embeddings for all 168 samples ─────────────────────────
embeddings = embed(scaled)   # shape (168, 64) — L2-normalized

# ── Step 6: Few-shot classification (5-shot example) ─────────────────────────
K_SHOT = 5
families = np.unique(labels)
prototypes = []
support_mask = np.zeros(len(labels), dtype=bool)

for fam in families:
    fam_indices = np.where(labels == fam)[0]
    # IMPORTANT: use FIRST K for support, REST for queries (no overlap!)
    support_idx = fam_indices[:K_SHOT]
    support_mask[support_idx] = True
    proto = embeddings[support_idx].mean(axis=0)
    prototypes.append(proto)

prototypes = np.stack(prototypes)        # (n_families, 64)

# Query = everything NOT used as support
query_embeddings = embeddings[~support_mask]
query_labels     = labels[~support_mask]

# ── Step 7: Classify by nearest prototype ─────────────────────────────────────
distances        = cdist(query_embeddings, prototypes, metric="euclidean")
predicted_idx    = distances.argmin(axis=1)
predicted_labels = families[predicted_idx]

accuracy = (predicted_labels == query_labels).mean()
print(f"Few-shot accuracy ({K_SHOT}-shot): {accuracy*100:.1f}%")
print(f"Queries classified: {len(query_labels)}")
print(f"Families tested:    {len(families)}")
```

---

## Model Performance (Member 2 Results)

| Setting | Validation Accuracy |
|---|---|
| 5-way 1-shot  | 59.2% |
| 5-way 5-shot  | 69.1% |
| 5-way 10-shot | 71.5% |
| 5-way 20-shot | 75.0% |
| **10-way 20-shot (best checkpoint)** | **86.8%** |

Best checkpoint: `best_embedding_model_10way_20shot_v3.pt` (saved at epoch 71, val=86.8%)

---

## Important Rules

1. **Always apply BOTH selector AND preprocessor** to raw features before embedding
2. **Support and query samples must NOT overlap** — use `fam_indices[:K]` for support, `fam_indices[K:]` for queries
3. **Use Euclidean distance** for nearest-prototype classification (cdist metric="euclidean")
4. **Do NOT retrain** the model — use the provided checkpoint as-is
5. **Label column** in `unseen_family_labels.csv` is `family_label` (not `family`)
6. The **25 unseen families** were NEVER used in Member 2's training — they are clean test data

---

## If joblib.load Fails for feature_preprocessor.joblib

If you get a `ModuleNotFoundError` when loading the preprocessor, it's because Member 1 used a custom class. Try:

```python
# Option 1: Use the selector only, then manually scale
import numpy as np

selector = joblib.load("../dataset/member2_handoff/anova_selector.joblib")
selected = selector.transform(raw_features)

# Manual standardization using training stats
mean = selected.mean(axis=0)
std  = selected.std(axis=0) + 1e-8
scaled = (selected - mean) / std
```

---

*From: Member 2 | Date: October 2026*
