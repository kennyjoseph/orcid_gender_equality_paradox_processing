import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import random
import pickle
import os

CACHE_EMBEDDINGS = "cache_embeddings.npy"
CACHE_AFFILIATIONS = "cache_affiliations.pkl"

PERIODS = [("1995-2000", 1995, 2001), ("2020-2025", 2020, 2026)]

def assign_periods(years):
    out = set()
    for y in years:
        if pd.notna(y):
            y = int(y)
            for label, lo, hi in PERIODS:
                if lo <= y < hi:
                    out.add(label)
    return out

# ── Load & filter ────────────────────────────────────────────────────────────
d = pd.read_parquet("../data/final_cleaning_dataset.parquet",
                    columns=["orcid", "clean_affiliation", "country", "start_year"])

unique_users = d.groupby("clean_affiliation")["orcid"].nunique()
frequent = unique_users[unique_users > 10].index
d_freq = d[d.clean_affiliation.isin(frequent)]

aff_countries = d_freq.groupby("clean_affiliation")["country"].apply(set).to_dict()
aff_periods   = (d_freq.groupby("clean_affiliation")["start_year"]
                       .apply(assign_periods).to_dict())

affiliations = list(aff_countries.keys())
print(f"Affiliations with >10 unique users: {len(affiliations):,}")
print(f"  — with ≥1 time period assigned: "
      f"{sum(bool(v) for v in aff_periods.values()):,}")

# ── Embeddings (cached) ──────────────────────────────────────────────────────
embeddings = None
if os.path.exists(CACHE_EMBEDDINGS) and os.path.exists(CACHE_AFFILIATIONS):
    print("Loading embeddings from cache...")
    embeddings = np.load(CACHE_EMBEDDINGS)
    with open(CACHE_AFFILIATIONS, "rb") as f:
        cached_affiliations = pickle.load(f)
    if cached_affiliations != affiliations:
        print("Cache affiliations mismatch — recomputing embeddings.")
        embeddings = None

if embeddings is None:
    model = SentenceTransformer("princeton-nlp/sup-simcse-roberta-large")
    embeddings = model.encode(affiliations, batch_size=256, show_progress_bar=True,
                              convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)
    np.save(CACHE_EMBEDDINGS, embeddings)
    with open(CACHE_AFFILIATIONS, "wb") as f:
        pickle.dump(affiliations, f)
    print("Embeddings computed and cached.")

# ── FAISS index ──────────────────────────────────────────────────────────────
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print("FAISS index built")

# ── Sampling loop ────────────────────────────────────────────────────────────
# Use large K so that after filtering to same-country neighbours there are
# still enough candidates to find one from each time period.
K = 300
N_ITER = 10_000
rng = random.Random(42)

records = []
skipped = 0

for _ in tqdm(range(N_ITER), desc="Sampling"):
    anchor_idx = rng.randrange(len(affiliations))
    anchor      = affiliations[anchor_idx]
    anchor_ctries  = aff_countries[anchor]
    anchor_periods = aff_periods[anchor]

    if not anchor_periods:
        skipped += 1
        continue

    sims, idxs = index.search(embeddings[anchor_idx: anchor_idx + 1], K)

    # Restrict to neighbours that share at least one country with the anchor
    neighbours = [
        (int(i), float(s)) for i, s in zip(idxs[0], sims[0])
        if int(i) != anchor_idx
        and aff_countries[affiliations[int(i)]] & anchor_ctries
    ]

    same_period, diff_period = None, None
    for nidx, sim in neighbours:
        nbr = affiliations[nidx]
        nbr_periods = aff_periods[nbr]
        if not nbr_periods:
            continue
        if same_period is None and anchor_periods & nbr_periods:
            same_period = (nbr, sim)
        if diff_period is None and not (anchor_periods & nbr_periods):
            diff_period = (nbr, sim)
        if same_period and diff_period:
            break

    records.append(dict(
        anchor               = anchor,
        same_period_affil    = same_period[0] if same_period else None,
        diff_period_affil    = diff_period[0] if diff_period else None,
        same_period_cos_sim  = same_period[1] if same_period else None,
        diff_period_cos_sim  = diff_period[1] if diff_period else None,
    ))

print(f"Iterations skipped (no period or no usable neighbour): {skipped}")

# ── Write output ─────────────────────────────────────────────────────────────
out = pd.DataFrame(records)
out.to_csv("affiliation_time_period_pairs.csv", index=False)
print(f"Wrote {len(out):,} rows to affiliation_time_period_pairs.csv")
print(out[["same_period_cos_sim", "diff_period_cos_sim"]].describe().to_string())
