import logging
import os
import re
from constants import LOGGING_FORMAT, DATE_FORMAT
from utils import extract_answer

logging.basicConfig(
    format=LOGGING_FORMAT,
    level=logging.INFO,
    datefmt=DATE_FORMAT)
logger = logging.getLogger()

# Override via env vars for reward sweeps (see grace_rl_sweep.slurm)
CORRECT_ANSWER_REWARD = float(os.getenv("CORRECT_REWARD", 1))
IDK_ANSWER_REWARD = float(os.getenv("IDK_REWARD", 0))
INCORRECT_ANSWER_REWARD = float(os.getenv("INCORRECT_REWARD", -1))
logger.info("Rewards: correct=%s idk=%s incorrect=%s", CORRECT_ANSWER_REWARD, IDK_ANSWER_REWARD, INCORRECT_ANSWER_REWARD)


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>.*?</reasoning>\s*<answer>.*</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    rewards_list = [1.0 if match else -1.0 for match in matches]
    return rewards_list

# This method does not require to be modified even if idk_answer is not passed
def accuracy_reward(completions, correct_answer, idk_answer=None, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    answers = [extract_answer(comp) for comp in completion_contents]
    rewards = []

    for i, (ans, correct_ans) in enumerate(zip(answers, correct_answer)):
        if ans == correct_ans:
            rewards.append(CORRECT_ANSWER_REWARD)
        elif idk_answer and ans and ans[0] and (ans[0] == idk_answer[i].upper() or "IDK" in ans[0]):
            rewards.append(IDK_ANSWER_REWARD)
        else:
            rewards.append(INCORRECT_ANSWER_REWARD)

    format_rewards = format_reward(completions)
    for comp, model_answer, correct_ans, reward, form_reward in zip(completion_contents, answers, correct_answer, rewards, format_rewards):
        logger.info(comp)
        logger.info("Model's answer: %s", model_answer)
        logger.info("Correct answer: %s", correct_ans)
        logger.info("Accuracy Reward: %s", reward)
        logger.info("Format Reward: %s", form_reward)
        logger.info("="*50)
    logger.info("Accuracy Rewards: %s", rewards)
    logger.info("Format Rewards: %s", format_rewards)
    return rewards
