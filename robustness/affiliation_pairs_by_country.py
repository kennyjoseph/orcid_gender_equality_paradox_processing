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

# ── Load & filter ────────────────────────────────────────────────────────────
d = pd.read_parquet("../data/final_cleaning_dataset.parquet",
                    columns=["orcid", "clean_affiliation", "country"])

unique_users = d.groupby("clean_affiliation")["orcid"].nunique()
frequent = unique_users[unique_users > 10].index
d_freq = d[d.clean_affiliation.isin(frequent)]

# Set of countries per affiliation
aff_countries = d_freq.groupby("clean_affiliation")["country"].apply(set).to_dict()
affiliations = list(aff_countries.keys())
print(f"Affiliations with >10 unique users: {len(affiliations):,}")

# ── Embeddings (cached) ──────────────────────────────────────────────────────
if os.path.exists(CACHE_EMBEDDINGS) and os.path.exists(CACHE_AFFILIATIONS):
    print("Loading embeddings from cache...")
    embeddings = np.load(CACHE_EMBEDDINGS)
    with open(CACHE_AFFILIATIONS, "rb") as f:
        cached_affiliations = pickle.load(f)
    if cached_affiliations != affiliations:
        print("Cache affiliations mismatch — recomputing embeddings.")
        os.remove(CACHE_EMBEDDINGS)
        os.remove(CACHE_AFFILIATIONS)
        embeddings = None
else:
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
K = 51          # fetch 51 neighbours, drop self → 50 candidates
N_ITER = 10_000
rng = random.Random(42)

records = []
skipped = 0

for _ in tqdm(range(N_ITER), desc="Sampling"):
    anchor_idx = rng.randrange(len(affiliations))
    anchor = affiliations[anchor_idx]
    anchor_ctries = aff_countries[anchor]

    sims, idxs = index.search(embeddings[anchor_idx: anchor_idx + 1], K)
    neighbours = [(int(i), float(s)) for i, s in zip(idxs[0], sims[0])
                  if int(i) != anchor_idx]

    same, diff = None, None
    for nidx, sim in neighbours:
        nbr = affiliations[nidx]
        nbr_ctries = aff_countries[nbr]
        if same is None and anchor_ctries & nbr_ctries:
            same = (nbr, sim)
        if diff is None and not (anchor_ctries & nbr_ctries):
            diff = (nbr, sim)
        if same and diff:
            break

    records.append(dict(
        anchor               = anchor,
        same_country_affil   = same[0] if same else None,
        diff_country_affil   = diff[0] if diff else None,
        same_country_cos_sim = same[1] if same else None,
        diff_country_cos_sim = diff[1] if diff else None,
    ))

print(f"Iterations with no usable neighbour (skipped): {skipped}")

# ── Write output ─────────────────────────────────────────────────────────────
out = pd.DataFrame(records)
out.to_csv("affiliation_pairs_by_country.csv", index=False)
print(f"Wrote {len(out):,} rows to robustness/affiliation_pairs_by_country.csv")
print(out[["same_country_cos_sim", "diff_country_cos_sim"]].describe().to_string())
