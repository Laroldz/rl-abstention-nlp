LOGGING_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


TRAIN = "train"
VAL = "validation"
TEST = "test"
BASELINE = "baseline"
SFT = "sft"
RL = "rl"
SFT_RL = "sft_rl"
GRANITE = "ibm-granite/granite-3.3-2b-instruct"
QWEN = "qwen/qwen3-4B-Instruct-2507"
MODELS = {"granite": GRANITE, "qwen": QWEN} # short names accepted by the BASE_MODEL / MODEL env vars
LORA = "lora"
FULL = "full"
IDK = "I Don't Know"
MEDMCQA = "medmcqa"
MEDMCQA_DATA = "openlifescienceai/medmcqa" # used in data.py only
MEDMCQA_CSV = "./medmcqa_{split}.csv" # offline copies of the official splits (train / validation / test), written by copy_local.py
GSM8K = "gsm8k"
GSM8K_DATA = "openai/gsm8k"  # used in data.py only
MATH = "hendrycks_math"
MATH_DATA = "EleutherAI/hendrycks_math"
PROMPT = "prompt"
ANSWER = "correct_answer"


IDK_PHRASES = [
    "I'm not entirely confident in my reasoning right now, so I'd rather not answer.",
    "My current line of thought feels uncertain — it's best I refrain from replying.",
    "I might be off track in my reasoning, so I'll hold back on giving an answer.",
    "I suspect there's a flaw in my logic, so I shouldn't respond just yet.",
    "I don't trust my reasoning enough at the moment to provide a reliable answer.",
    "My analysis feels shaky, so I prefer to stay silent instead of misleading.",
    "I haven't reasoned this through clearly enough to offer a proper response.",
    "There's too much uncertainty in my thought process for me to give a confident answer.",
    "I sense that my reasoning might be flawed, so I'd better not comment.",
    "I'm questioning my logic here, so I'll skip giving an answer for now.",
    "I'm unsure whether my reasoning holds up, so I'll avoid answering.",
    "I realize my thought process might not be sound, so I won't attempt an answer.",
    "I'm second-guessing my logic, so I'll refrain from responding.",
    "I don't think my reasoning is solid enough to justify a response.",
    "My reasoning feels incomplete, so it's wiser if I don't answer yet.",
    "I'm uncertain about my conclusions, so I'd prefer not to answer.",
    "Something about my reasoning doesn't add up, so I'll pause instead of replying.",
    "I'm in doubt about the validity of my logic, so I won't respond.",
    "My argument isn't stable enough right now to make a statement.",
    "Given my doubts about my reasoning, I'll refrain from giving an answer."
]