# ORCID Data Processing Pipeline

This repository contains a series of Jupyter notebooks that process ORCID researcher data to extract affiliations, classify fields as STEM/medicine, infer gender, and categorize academic roles.

## Pipeline Overview

The notebooks should be run in sequential order (00 through 05).

### 00_process_summaries_dump.ipynb

**Purpose:** Processes raw ORCID XML summary files and extracts affiliation data.

**Input:** Raw ORCID XML files from `/data/orcid/ORCID_2025_10_summaries/`, from the [2025 ORCID data dump](https://orcid.figshare.com/articles/dataset/ORCID_Public_Data_File_2025/30375589/1).

**Output:** Processed affiliations for use in the next series of preprocessing steps

---

### 01_data_cleaning.ipynb

**Purpose:** Cleans and enriches the affiliation data with translations and gender inference.

**Input:** Parquet files from step 00

**Output:** `data/final_cleaning_dataset.parquet`

**Key Operations:**
- Consolidates ~1100 affiliation parquet files (~21M records, ~10M unique ORCIDs)
- Drops invited positions, keeping only employment and education
- Handles department vs. organization name fallback for affiliations
- Applies country-specific rules (e.g., removes location names used as departments in CO, PE, SV, BO, UY, PY)
- Filters records without dates and cleans short/invalid affiliations
- **Translation:** Uses Google Translate API to translate non-English affiliations to English
- **Gender Inference:**
  - Primary: NomQuam library for name-based gender inference
  - Fallback: Namsor API for names not covered by NomQuam
- Removes individuals with >12 affiliations as outliers
- Final dataset: ~16.4M records, ~7M unique ORCIDs

---

### 02_build_stem_classifier.ipynb

**Purpose:** Trains a machine learning classifier to identify STEM fields from affiliation text.

**Input:**
- `data/larremore_field_data.csv` - Labeled field data from Larremore study
- `data/stem_training_data.csv` - Additional ORCID-specific training data
- `data/stem_validation_data.csv` - Validation dataset (250 samples)

**Output:** `data/final_stem_classifier.joblib`

**Key Operations:**
- Uses SimCSE embeddings (`princeton-nlp/sup-simcse-roberta-large`) for text representation
- Trains a calibrated logistic regression classifier
- Tests multiple training configurations combining different field name formats
- Best model: `simple_cleaned_manual_simcse_preproc` achieving ~89% accuracy on validation set
- Outputs calibrated probability scores for STEM classification

---

### 03_run_stem_med_role_classifier.ipynb

**Purpose:** Applies STEM/medicine classifiers and generates role categories.

**Input:**
- `data/final_cleaning_dataset.parquet`
- `data/final_stem_classifier.joblib`

**Output:**
- `data/stem_and_med_classifications.parquet`
- `data/roles.parquet`

**Key Operations:**
- **STEM Classification:** Runs trained classifier on all unique affiliations using parallel processing
- **Medicine Classification:** Regex-based classifier using 48 medical root terms (e.g., "cardio", "surgery", "oncol")
  - Excludes biomedical engineering and plant-related terms
  - Achieves 99% accuracy on validation set
- **Role Classification:** Categorizes roles into:
  - `bachelors` - Undergraduate degrees
  - `masters/postgrad` - Master's and postgraduate programs
  - `phd` - Doctoral students and candidates
  - `postdoc` - Postdoctoral researchers and fellows
  - `prof` - Professors, lecturers, faculty
  - `head` - Deans, directors, department heads
  - `research` - Research scientists and scholars
- Captures ~65% of non-null roles with these categories

---

### 04_final_writing_out.ipynb

**Purpose:** Merges all processed data and generates final output files.

**Input:**
- `data/final_cleaning_dataset.parquet`
- `data/roles.parquet`
- `data/stem_and_med_classifications.parquet`

**Output:**
- `data/full_affiliations_data.csv` - Complete merged dataset
- `data/full_affiliations_data_sample_50k.csv` - 50K sample for testing
- `orcid_res_10/` and `orcid_res_5/` - Per-ORCID CSV files with country/year data

**Key Operations:**
- Merges role categories and STEM/medicine classifications
- Applies additional MD role detection using multilingual regex (supports 25+ languages)
- Generates per-person longitudinal files with:
  - Country, ORCID ID, STEM probability, gender probability, medicine flag, year
- Handles missing end years based on role type:
  - Bachelors: 4 years
  - Masters: 2 years
  - Postdoc: 3 years
  - PhD: 5 years
  - Prof/default: 5 or 10 years (two output variants)

---

### 05_upload_to_dropbox.ipynb

**Purpose:** Uploads final data files to Dropbox for sharing.

**Input:** Generated CSV files from step 04

**Output:** Files uploaded to `/Kenny/orcid/` on Dropbox

**Key Operations:**
- Implements chunked upload for large files (>150MB)
- Uploads:
  - `full_affiliations_data.csv` (~5.5 GB)
  - `all_default10.csv` (~6.5 GB)
  - `all_default5.csv`

---

## Data Schema

### Final Affiliations Data (`full_affiliations_data.csv`)

| Column | Description |
|--------|-------------|
| `orcid` | ORCID identifier |
| `name` | Full name |
| `type` | employment or education |
| `org_name` | Organization name |
| `role` | Original role text |
| `country` | ISO country code |
| `department_name` | Department name |
| `start_year` | Start year of affiliation |
| `end_year` | End year (may be imputed) |
| `clean_affiliation` | Cleaned/translated affiliation text |
| `p(gf)` | Probability of female gender (0-1) |
| `stem_prob` | Probability of STEM field (0-1) |
| `med_clf` | Boolean medicine classification |
| `role_category` | Categorized role (bachelors/masters/phd/postdoc/prof/head/research) |

---

## Environment Setup

This project was developed using the `orcid2` conda environment. To replicate the environment:

```bash
# Create a new conda environment
conda create -n orcid2 python=3.10

# Activate the environment
conda activate orcid2

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_md

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('stopwords')"
```

### Key Dependencies

- pandas, numpy
- scikit-learn
- spacy (`en_core_web_md`)
- SimCSE (`princeton-nlp/sup-simcse-roberta-large`)
- nomquamgender
- langid
- google-cloud-translate
- dropbox
- BeautifulSoup4
- nltk
- joblib
- multiprocess

See `requirements.txt` for the complete list of pinned dependencies.

---

## External APIs Used

- **Google Translate API** - For translating non-English affiliations
- **Namsor API** - For gender inference on names not covered by NomQuam

---

## Data Sources

- ORCID 2025-10 data dump
- NSF field taxonomy (from archive.org)
- Wikipedia academic disciplines outline
- Larremore field classification dataset
- Hand-coded STEM affiliations data
