import pandas as pd
import logging
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from constants import LOGGING_FORMAT, DATE_FORMAT, TRAIN, VAL, TEST, SFT, IDK_PHRASES, MEDMCQA, MEDMCQA_DATA, MEDMCQA_CSV, GSM8K, GSM8K_DATA, MATH, MATH_DATA
import kagglehub
import re
from datasets import concatenate_datasets
import random
import os
from collections import Counter
from utils import extract_boxed_contents

# Set up logging
logging.basicConfig(
    format=LOGGING_FORMAT,
    level=logging.INFO,
    datefmt=DATE_FORMAT)

logger = logging.getLogger()

def process_example_medmcqa(sample, idk_enabled=False):
    TRAINING_TYPE = os.getenv("TRAINING_TYPE") or os.getenv("EVAL_TYPE")
    choices = [sample[f'op{x}'] for x in 'abcd']
    options_dict = {chr(65 + i): choice for i, choice in enumerate(choices)}
    options_str = "\n".join([f'{chr(65 + i)}: {choice}' for i, choice in enumerate(choices)])
    correct_option = chr(65 + sample['cop'])

    base_content = "Answer the following question. Provide your thoughts between <reasoning> and </reasoning> symbols. Provide the final answer option (letter only) between <answer> and </answer> symbols."

    if idk_enabled:
        base_content += " Answer only if you are certain, else choose I Don't Know."
        choices = choices + ["I Don't Know"]
        options_dict = {chr(65 + i): choice for i, choice in enumerate(choices)}
        options_str = "\n".join([f'{chr(65 + i)}: {choice}' for i, choice in enumerate(choices)])
        idk_option = chr(65 + len(choices) - 1)
        
        if TRAINING_TYPE is not None and TRAINING_TYPE == SFT:
            if sample['correct_answer'] != sample["model_answer"]:
                idk_phrase = random.choice(IDK_PHRASES)
                completion = f"<reasoning>{sample['exp'] + ' ' + idk_phrase}</reasoning><answer>{idk_option}</answer>"
            else:
                completion = f"<reasoning>{sample['exp']}</reasoning><answer>{correct_option}</answer>"

            COMPLETION_MESSAGES = [
                {
                    'role': 'assistant',
                    'content': completion
                }
            ]


    PROMPT_MESSAGES = [
        {
            'role': 'user',
            'content': base_content + \
                f"Question: {sample['question']}\n" \
                f"Options: \n{options_str}"
        } 
    ]

    result = {
        'prompt': PROMPT_MESSAGES,
        'correct_answer': correct_option
    }

    if idk_enabled:
        result['idk_answer'] = chr(65 + len(choices) - 1)
    if TRAINING_TYPE is not None and TRAINING_TYPE == SFT:
        result['completion'] = COMPLETION_MESSAGES

    return result

def process_example_gsm8k(sample, idk_enabled=False):
    match = re.search(r"####\s*(-?[\d,]+)", sample['answer'])
    if match:
        correct_numeric_answer = int(match.group(1).replace(",", ""))
    else:
        log.error("GSM8K data not clean")

    base_content = "Answer the following question. Provide your thoughts between <reasoning> and </reasoning> symbols. Provide the final numeric answer (number only) between <answer> and </answer> symbols."
    if idk_enabled:
        base_content += " Answer only if you are certain, else answer I Don't Know."

    PROMPT_MESSAGES = [
        {
            'role': 'user',
            'content': base_content + \
                f"Question: {sample['question']}"
        }
    ]
    result = {
        'prompt': PROMPT_MESSAGES,
        'correct_answer': correct_numeric_answer
    }
    if idk_enabled:
        result['idk_answer'] = "I Don't Know"
    return result


def process_example_math(sample, idk_enabled=False):
    correct_numeric_answer = extract_boxed_contents(sample['solution'])

    if not correct_numeric_answer:
        logger.error("MATH data not clean") # Made sure it is always clean

    base_content = """Answer the following question. Provide your reasoning between <reasoning> and </reasoning> tags. Then, provide the final answer between <answer> and </answer> tags.

CRITICAL FORMATTING REQUIREMENTS:
1. The <answer> section must contain ONLY the final answer in LaTeX \boxed{} format
2. NO additional text, explanations, code, or tags should appear inside <answer> tags
3. The answer must be mathematically correct and in simplest form"""

    if idk_enabled:
        base_content += " Answer only if you are certain, else answer \text{IDK}."

    PROMPT_MESSAGES = [
        {
            'role': 'user',
            'content': base_content + \
                f"Question: {sample['problem']}"
        }
    ]
    result = {
        'prompt': PROMPT_MESSAGES,
        'correct_answer': correct_numeric_answer
    }
    if idk_enabled:
        result['idk_answer'] = "\text{IDK}"
    return result

def load_medmcqa_split(split):
    """Load an official MedMCQA split from medmcqa_<split>.csv if present (offline), else from Hugging Face.
    Note: the official test split is unlabeled (cop == -1), so only train is used for training/eval."""
    path = MEDMCQA_CSV.format(split=split)
    if not os.path.isfile(path):
        return load_dataset(MEDMCQA_DATA, split=split)
    # read everything as str so options like "5" or "NA" aren't turned into numbers / NaN
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df['cop'] = df['cop'].astype(int)
    logger.info(f"Loaded {len(df)} MedMCQA {split} rows from {path}")
    return Dataset.from_pandas(df, preserve_index=False)

def get_medmcqa_data(idk_enabled=False):
    TRAINING_TYPE = os.getenv("TRAINING_TYPE") or os.getenv("EVAL_TYPE")
    if TRAINING_TYPE is not None and TRAINING_TYPE == SFT:
        ds = load_from_disk('./eval_outputs/baseline_ibm-granite/granite-3.3-2b-instruct_medmcqa_train_40000_4options')
        ds = ds.filter(
            lambda sample: (sample['exp'] is not None) and (len(sample['exp']) < 2250)
        )
        # Split sizes: train=58%, val=2%, test=40%
        # Train=146037, Validation=220, Test=36565
        return get_data(ds, lambda x: process_example_medmcqa(x, idk_enabled),
                        train_size=0.7988, val_size=0.0012, test_size=0.20)
    ds = load_medmcqa_split(TRAIN)
    # Split sizes: train=79.88%, val=0.12%, test=20%
    # Train=146037, Validation=220, Test=36565
    return get_data(ds, lambda x: process_example_medmcqa(x, idk_enabled),
                    train_size=0.7988, val_size=0.0012, test_size=0.20)

def get_gsm8k_data(idk_enabled=True):
    ds = load_dataset(GSM8K_DATA, "main")
    ds = concatenate_datasets([ds[TRAIN], ds[TEST]])
    # Split sizes: train=60%, val=10%, test=30%
    # Train=5275, Validation=880, Test=2638
    return get_data(ds, lambda x: process_example_gsm8k(x, idk_enabled),
                    train_size=0.60, val_size=0.10, test_size=0.30)

def get_math_data(idk_enabled=True):
    ds_algebra = load_dataset(MATH_DATA, 'algebra')
    ds_pnc = load_dataset(MATH_DATA, 'counting_and_probability')
    ds_geometry = load_dataset(MATH_DATA, 'geometry')
    ds_int_algebra = load_dataset(MATH_DATA, 'intermediate_algebra')
    ds_number_theory = load_dataset(MATH_DATA, 'number_theory')
    ds_pre_algebra = load_dataset(MATH_DATA, 'prealgebra')
    ds_pre_calculus = load_dataset(MATH_DATA, 'precalculus')
    ds = concatenate_datasets([ds_algebra[TRAIN], ds_algebra[TEST],
                               ds_pnc[TRAIN], ds_pnc[TEST],
                               ds_geometry[TRAIN], ds_geometry[TEST],
                               ds_int_algebra[TRAIN], ds_int_algebra[TEST],
                               ds_number_theory[TRAIN], ds_number_theory[TEST],
                               ds_pre_algebra[TRAIN], ds_pre_algebra[TEST],
                               ds_pre_calculus[TRAIN], ds_pre_calculus[TEST]])

    # filtering
    keep = {'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'}
    seq = r'\boxed{'
    ds = ds.filter(lambda sample: sample["level"] in keep)
    ds = ds.filter(lambda sample: sample["solution"].count(seq) != 0)

    # encoding
    ds = ds.class_encode_column("level")
    logger.info("Feature description: %s", ds.features)

    # Split sizes: train=68%, val=2%, test=30%
    # Train=8497, Validation=250, Test=3749
    return get_data(ds, lambda x: process_example_math(x, idk_enabled),
                    train_size=0.68, val_size=0.02, test_size=0.30, stratify="level")

def get_data(ds, process_example, train_size, val_size, test_size, stratify=None):
    ds.cleanup_cache_files()
    ds = ds.map(
        process_example,
        num_proc=8
    )

    # First split: separate test set (20%)
    ds = ds.train_test_split(test_size=test_size, seed=42, stratify_by_column=stratify)

    # Second split: separate validation from training
    # val_size as proportion of the training set
    val_proportion = val_size / (train_size + val_size)
    val = ds[TRAIN].train_test_split(test_size=val_proportion, seed=42, stratify_by_column=stratify)

    logger.info(f"Length of train: {len(val[TRAIN])}")
    logger.info(f"Length of validation: {len(val[TEST])}")
    logger.info(f"Length of test: {len(ds[TEST])}")

    modified_ds_split = DatasetDict({
        TRAIN: val[TRAIN],
        VAL: val[TEST],
        TEST: ds[TEST]})
    
    if stratify:
        train_count_group_by_level = dict(sorted(Counter(modified_ds_split[TRAIN][stratify]).items()))
        val_count_group_by_level = dict(sorted(Counter(modified_ds_split[VAL][stratify]).items()))
        test_count_group_by_level = dict(sorted(Counter(modified_ds_split[TEST][stratify]).items()))
        logger.info("Train dataset: %s", train_count_group_by_level)
        logger.info("Validation dataset: %s", val_count_group_by_level)
        logger.info("Test dataset: %s", test_count_group_by_level)

    return modified_ds_split
