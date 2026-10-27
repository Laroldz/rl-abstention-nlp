# Run on a machine with internet, then copy the medmcqa_*.csv files into the repo on Grace.
# data.py reads them instead of downloading MedMCQA from Hugging Face.
from datasets import load_dataset

from constants import MEDMCQA_DATA, MEDMCQA_CSV

for split, ds in load_dataset(MEDMCQA_DATA).items():
    path = MEDMCQA_CSV.format(split=split)
    ds.to_csv(path, index=False)
    print(f"Saved {len(ds)} rows of {MEDMCQA_DATA} ({split}) to {path}")
