from prefect import Flow, task
from prefect.engine.flow_runner import FlowRunner
from crossmodal_embedding.tasks.preprocessing import PreprocessStarTask
from crossmodal_embedding.tasks.crossmodal import TrainingTaskStar
from loguru import logger
import argparse
import json
import sys

parser = argparse.ArgumentParser(description="Training model")


parser.add_argument("--train", type=str, help=f"train filename")
parser.add_argument("--test", type=str, help=f"test filename")
parser.add_argument("--dev", type=str, help=f"dev filename")
parser.add_argument("--encoder", type=str, help=f"encoder filename")
parser.add_argument("--output_log", type=str, help=f"output log")
parser.add_argument("--output_model", type=str, help=f"output model")
parser.add_argument("--num_negatives", type=int)
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--out_embedding", type=int, default=512)
parser.add_argument("--embedding", type=int, default=512)
parser.add_argument("--decay", type=float, default=0.01)
parser.add_argument("--max_len", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--use_cache", help="use cache", action="store_true")
parser.add_argument("--att_head", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=2e-4)

args = parser.parse_args()
train_file = args.train
test_file = args.test
dev_file = args.dev
encoder_file = args.encoder
output_log = args.output_log
output_model = args.output_model
hidden_size = args.hidden_size
num_negatives = args.num_negatives
out_embedding = args.out_embedding
embedding = args.embedding
use_cache = args.use_cache
decay = args.decay
max_len = args.max_len
batch_size = args.batch_size
att_head = args.att_head
learning_rate = args.learning_rate

with open(encoder_file, "r") as f:
    statements = json.load(f)
with open(train_file, "r") as f:
    train = json.load(f)
with open(test_file, "r") as f:
    test = json.load(f)
with open(dev_file, "r") as f:
    dev = json.load(f)

if not statements or not train or not test or not dev:
    logger.error(f"Make sure all the files exist!")
    sys.exit()


data_prep = PreprocessStarTask()
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
        train, test, dev, statements, use_cache=use_cache, max_len=MAX_LEN
    )
    # # outputs dict with [vocab] [num_neg][train/test/dev]

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
