from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from peft import PeftModelForCausalLM
from data import get_medmcqa_data, get_gsm8k_data, get_math_data
import datasets
import logging
from tqdm import tqdm
import re
from constants import *
import constants
from utils import chunked, extract_answer, DATASET_OPTIONS
import time
from datetime import datetime
import os
import sys


logging.basicConfig(
    format=LOGGING_FORMAT,
    level=logging.INFO,
    datefmt=DATE_FORMAT)

logger = logging.getLogger()

# Configuration flags
# Settings read with os.getenv can be overridden with an env var of the same name (used by grace_rl_sweep.slurm)

# Model settings
MODEL = os.getenv("MODEL", GRANITE) # Options: GRANITE | QWEN, or short names "granite" | "qwen"
MODEL = constants.MODELS.get(MODEL, MODEL)
LOAD_SPECIFIC_MODEL = os.getenv("LOAD_SPECIFIC_MODEL", "false").lower() == "true"
MODEL_CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT_PATH", "sft_rl_medmcqa_abstention_qwen_chk300_model_idk_plus_0/checkpoint-90") # only useful if loading specific model
MODEL_CHECKPOINT_PATH_1 = None     # Another checkpoint path Eg: RL over SFT
if LOAD_SPECIFIC_MODEL:
    EVAL_TYPE = os.getenv("EVAL_TYPE", SFT) # SFT | RL | SFT_RL. This will only affect the output file name
else:
    EVAL_TYPE = BASELINE
RUN_TAG = os.getenv("RUN_TAG", "") # appended to the output name so sweep runs don't overwrite each other

# Data settings
DATA = os.getenv("DATA", MATH) # Options: medmcqa | gsm8k | hendrycks_math
IDK_ENABLED = True  # Toggle IDK option in dataset
EVAL_ON = TEST # always keep this test dataset for eval unless really necessary
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 40000))
os.environ["DATA"] = DATA
os.environ["IDK_ENABLED"] = "true" if IDK_ENABLED else "false"
os.environ["EVAL_TYPE"] = EVAL_TYPE

# parallel processing and checkpointing
USE_BATCH_PROCESSING = True  # Toggle between batch and sequential processing
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
LOAD_CHECKPOINT = False
CHECKPOINT_INTERVAL_HOURS = 2  # Checkpoint interval in hours
CHECKPOINT_DIR = "eval_checkpoints"  # Directory to save checkpoints
CHECKPOINT_PATH = "eval_checkpoints/checkpoint_records_2336_time_8.18h_20251127_233002" # only useful if loading specific checkpoint



def save_checkpoint(records, num_processed, elapsed_time_hours):
    """Save checkpoint with processed records"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"{CHECKPOINT_DIR}/checkpoint_records_{num_processed}_time_{elapsed_time_hours:.2f}h_{timestamp}"

    checkpoint_ds = datasets.Dataset.from_list(records)
    checkpoint_ds.save_to_disk(checkpoint_name)
    logger.info(f"Checkpoint saved: {checkpoint_name} (Records: {num_processed}, Time: {elapsed_time_hours:.2f}h)")
    return checkpoint_name

logger.info("Using model: %s", MODEL)

tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side = "left")
logger.info("Tokenizer loaded")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto")
logger.info("Base model loaded")
if LOAD_SPECIFIC_MODEL:
    model = PeftModelForCausalLM.from_pretrained(model, MODEL_CHECKPOINT_PATH)
    logger.info("Peft model Loaded")
    if MODEL_CHECKPOINT_PATH_1:
        model = model.merge_and_unload()
        model = PeftModelForCausalLM.from_pretrained(model, MODEL_CHECKPOINT_PATH_1)

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
logger.info("Pipeline device: %s", pipe.device)

NUM_OPTIONS = DATASET_OPTIONS.get(DATA)
# Load dataset with IDK flag
match DATA:
    case constants.MEDMCQA:
        if IDK_ENABLED:
            NUM_OPTIONS += 1
        ds = get_medmcqa_data(idk_enabled=IDK_ENABLED)
    case constants.GSM8K:
        ds = get_gsm8k_data(idk_enabled=IDK_ENABLED)
    case constants.MATH:
        ds = get_math_data(idk_enabled=IDK_ENABLED)
    case _:
        logger.error("Please select valid dataset")
        raise ValueError("Invalid dataset selected")

logger.info("Dataset loaded: %s", ds[EVAL_ON])

test_ds = ds[EVAL_ON]
NUM_SAMPLES = min(NUM_SAMPLES, len(test_ds))
final_records = []

# Checkpoint tracking
start_time = time.time()
last_checkpoint_time = start_time
checkpoint_interval_seconds = CHECKPOINT_INTERVAL_HOURS * 3600
num_processed = 0

if LOAD_CHECKPOINT:
    logger.info(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    checkpoint_ds = datasets.load_from_disk(CHECKPOINT_PATH)
    final_records = checkpoint_ds.to_list()
    num_processed = len(final_records)
    logger.info(f"Resumed from checkpoint: {num_processed} records already processed")
    time_match = re.search(r'time_(\d+\.\d+)h', CHECKPOINT_PATH)
    if time_match:
        previous_elapsed_hours = float(time_match.group(1))
        start_time -= previous_elapsed_hours * 3600
        logger.info(f"Adjusted start time to account for {previous_elapsed_hours:.2f}h of previous processing")
remaining_samples = NUM_SAMPLES - num_processed
if remaining_samples <= 0:
    logger.info("All samples already processed!")
    sys.exit(0)
else:
    logger.info(f"Processing remaining {remaining_samples} samples")

if USE_BATCH_PROCESSING:
    logger.info("Using BATCH processing mode")

    for batch in tqdm(chunked(test_ds.select(range(num_processed, NUM_SAMPLES)), BATCH_SIZE), desc="Evaluation progress", initial=num_processed // BATCH_SIZE, total=(NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE):
        prompts = [s[PROMPT] for s in batch]
        outputs = pipe(prompts, max_new_tokens=2048, batch_size=BATCH_SIZE)

        for s, out in zip(batch, outputs):
            generated = out[0]['generated_text'][-1]['content']
            answer = extract_answer(generated)

            logger.info("Prompt: %s", s[PROMPT])
            logger.info("Model response: %s", generated)
            logger.info("Model Answer Extracted: %s", answer)
            logger.info("Correct Answer: %s", s[ANSWER])
            logger.info("="*100)

            s['model_response'] = generated
            s['model_answer'] = answer
            final_records.append(s)
            num_processed += 1

        # Check if checkpoint interval has passed
        current_time = time.time()
        if current_time - last_checkpoint_time >= checkpoint_interval_seconds:
            elapsed_time_hours = (current_time - start_time) / 3600
            save_checkpoint(final_records, num_processed, elapsed_time_hours)
            last_checkpoint_time = current_time

else:
    logger.info("Using SEQUENTIAL processing mode")

    for sample in tqdm(test_ds.select(range(num_processed, NUM_SAMPLES)), desc="Evaluation progress", initial=num_processed,total=NUM_SAMPLES):
        logger.info("Prompt: %s", sample[PROMPT])

        response = pipe(sample[PROMPT], max_new_tokens=2048)[0]['generated_text'][-1]['content']
        answer = extract_answer(response)

        logger.info("Model response: %s", response)
        logger.info("Model Answer Extracted: %s", answer)
        logger.info("Correct Answer: %s", sample[ANSWER])
        logger.info("="*100)

        sample['model_response'] = response
        sample['model_answer'] = answer
        final_records.append(sample)
        num_processed += 1

        # Check if checkpoint interval has passed
        current_time = time.time()
        if current_time - last_checkpoint_time >= checkpoint_interval_seconds:
            elapsed_time_hours = (current_time - start_time) / 3600
            save_checkpoint(final_records, num_processed, elapsed_time_hours)
            last_checkpoint_time = current_time


# Save final output
out_ds = datasets.Dataset.from_list(final_records)
idk_suffix = "_idk" if IDK_ENABLED else ""
EVAL_DATA_NAME = f"eval_outputs/{EVAL_TYPE}_{MODEL}_{DATA}_{NUM_SAMPLES}_{(f'{NUM_OPTIONS}option' if NUM_OPTIONS else 'numeric')}{idk_suffix}{f'_{RUN_TAG}' if RUN_TAG else ''}"

out_ds.save_to_disk(EVAL_DATA_NAME)
logger.info(f"Final evaluation results saved to: {EVAL_DATA_NAME}")

# Log total time
total_time_hours = (time.time() - start_time) / 3600
logger.info(f"Evaluation completed. Total time: {total_time_hours:.2f} hours, Records processed: {num_processed}")
