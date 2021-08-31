from prefect import Flow, Parameter, Task, tags, task
from prefect.engine.results import LocalResult
from prefect.engine.flow_runner import FlowRunner
from crossmodal_embedding.tasks.preprocessing import PreprocessStarTask
from crossmodal_embedding.tasks.crossmodal import TrainingTaskStar
from loguru import logger
import argparse
import json
from dynaconf import settings
import sys
import os
from os import path


parser = argparse.ArgumentParser(description="Training model")


parser.add_argument("--use_similar", help="use similar examples", action="store_true")
parser.add_argument("--num_negatives", type=int)
parser.add_argument("--use_random", help="use random examples", action="store_true")
parser.add_argument("--output_log", type=str, help=f"output log file", default="log_execution.json")
parser.add_argument("--output_model", type=str, help=f"output model file", default="trained_model.pt")
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--out_embedding", type=int, default=512)
parser.add_argument("--embedding", type=int, default=512)
parser.add_argument("--decay", type=float, default=0.01)
parser.add_argument("--max_len", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--att_head", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=2e-4)

args = parser.parse_args()

use_similar = args.use_similar 
use_random = args.use_random

if use_similar and use_random:
    logger.error(f"You can only use one of these options: use_similar or use_random, you are using both!")
    sys.exit()

if not use_similar and not use_random:
    logger.error(f"You didn't set use_similar or use_random, using random examples as default.")
    

num_negatives = args.num_negatives

if num_negatives not in [1,2,5,10]:
    logger.error(f"Number of negatives should be 1, 2, 5 or 10, you are selecting: {num_negatives}")
    sys.exit()




output_log = args.output_log
output_model = args.output_model
hidden_size = args.hidden_size

out_embedding = args.out_embedding
embedding = args.embedding
decay = args.decay
max_len = args.max_len
batch_size = args.batch_size
att_head = args.att_head
learning_rate = args.learning_rate

if use_similar:
    filename = f"./dataset/similar/neg_1_"
else:
    filename = f"./dataset/random/neg_1_"

with open(f"{filename}statements.json", "r") as f:
    statements = json.load(f)
with open(f"{filename}train.json", "r") as f:
    train = json.load(f)
with open(f"{filename}test.json", "r") as f:
    test = json.load(f)
with open(f"{filename}dev.json", "r") as f:
    dev = json.load(f)



CACHE_LOCATION = settings["cache_location"]
cache_args = dict(
    target="{task_name}--"+f"{num_negatives}--{use_random}.pkl",
    checkpoint=True,
    result=LocalResult(dir=f"{CACHE_LOCATION}"),
)

if not os.path.exists('./models'):
        os.mkdir('./models')

if not os.path.exists('./logs'):
        os.mkdir('./logs')

data_prep = PreprocessStarTask(**cache_args)
## Param
MAX_LEN = max_len

train_task = TrainingTaskStar()
# Param
BATCH_SIZE = batch_size
NUM_EPOCHS = 50
LEARNING_RATE = learning_rate
ATTENTION_HEAD = att_head



with Flow("Generating statement representation model - Negatives ") as flow1:
    dataset = data_prep(
        train, test, dev, statements, max_len=MAX_LEN
    )
    train_task(
        dataset["train"],
        dataset["test"],
        dataset["dev"],
        num_negatives=num_negatives,
        output_log=output_log,
        output_model=output_model,
        vocab_size=dataset["vocab"],
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        max_sequence_len=MAX_LEN,
        learning_rate=LEARNING_RATE,
        hidden_size=hidden_size,
        out_embedding=out_embedding,
        attention_heads=ATTENTION_HEAD,
        word_embedding=embedding,
        decay=decay,
    )

FlowRunner(flow=flow1).run()
