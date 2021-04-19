from prefect import Task
from loguru import logger
import json
from tqdm import tqdm
from crossmodal_embedding.util import tokenizer
import pandas as pd
from crossmodal_embedding.util.caching import recover_cache, save_cache


class PreprocessStarTask(Task):
    def run(self, train, test, dev, statements, use_cache=False, max_len=100):

        padded_statements, padded_masking, vocabulary_size, leng = self.prepare_input(
            statements, max_len
        )
        dataset = dict()

        train_pairs = self.prepare_pairs_with_padding(
            train, padded_statements, padded_masking, leng
        )

        # pd.to_pickle(train_pairs, f"./cache/train_{n}.pickle")

        test_pairs = self.prepare_pairs_with_padding(
            test, padded_statements, padded_masking, leng
        )

        # pd.to_pickle(test_pairs, f"./cache/test_{n}.pickle")

        dev_pairs = self.prepare_pairs_with_padding(
            dev, padded_statements, padded_masking, leng
        )

        # pd.to_pickle(dev_pairs, f"./cache/dev_{n}.pickle")

        dataset = {"train": train_pairs, "test": test_pairs, "dev": dev_pairs}
        dataset["vocab"] = vocabulary_size + 1

        return dataset

    def prepare_pairs_with_padding(
        self, pairs, padded_statements, padded_masking, leng
    ):
        new_pairs = dict()
        i = 0
        for id_p, content in pairs.items():

            if (
                str(content["statement_1"]) in padded_statements
                and str(content["statement_2"]) in padded_statements
            ):
                new_pairs[i] = {
                    "e1": padded_statements[str(content["statement_1"])],
                    "e1_mask": padded_masking[str(content["statement_1"])],
                    "e1_len": leng[str(content["statement_1"])],
                    "e2": padded_statements[str(content["statement_2"])],
                    "e2_mask": padded_masking[str(content["statement_2"])],
                    "e2_len": leng[str(content["statement_2"])],
                    "score": content["score"],
                }
                i = i + 1
        pairs = pd.DataFrame.from_dict(new_pairs, "index")
        return pairs

    def prepare_input(self, statements, max_len):
        tokenized_statements, masking = self.tokenize_and_get_masking(statements)
        encoder, decoder, bow_statements = self.get_bow(tokenized_statements)
        vocabulary_size = len(encoder)
        padded_statements, padded_masking, leng = self.add_padding(
            bow_statements, masking, max_len
        )
        logger.info(f"Padded statements: {len(padded_statements)}")
        return padded_statements, padded_masking, vocabulary_size, leng

    def tokenize_and_get_masking(self, statements):
        logger.info("Running tokenisation and masking")
        tokenized_statements, masking = tokenizer.tokenize_exp_as_symbols_dict(
            statements
        )

        return tokenized_statements, masking

    def get_bow(self, statements):
        logger.info("Building vocabulary!")
        bow_statements = dict()
        encoder = dict()
        decoder = dict()
        encoder["[START]"] = 1
        encoder["[END]"] = 2
        decoder[1] = "[START]"
        decoder[2] = "[END]"

        for title, statement in statements.items():
            new_tokens = list()
            for token in statement:
                if token not in encoder:
                    encoder[token] = len(encoder) + 1
                    decoder[encoder[token]] = token
                new_tokens.append(encoder[token])
            new_tokens = new_tokens
            bow_statements[title] = new_tokens
        return encoder, decoder, bow_statements

    def add_padding(self, bow, masking, max_size=200):
        largest_expression = max_size
        average_expression = 0
        count = 0
        leng = dict()
        logger.info("Adding padding")
        padded_statements = dict()
        padded_masking = dict()
        logger.info("Checking largest statement.")

        for title, encoding in bow.items():
            if len(encoding) > max_size or len(masking[title]) > max_size:
                bow[title] = encoding[:max_size]
                masking[title] = masking[title][:max_size]

        for title, encoding in bow.items():
            if len(encoding) > largest_expression:
                largest_expression = len(encoding)
            padded_statements[title] = encoding

            average_expression = average_expression + len(encoding)

        logger.info(f"Largest expression: {largest_expression}")

        logger.info(f"Adding padding")
        for title, encoding in padded_statements.items():
            if len(encoding) < largest_expression:
                leng[title] = len(encoding)
                add_padding = largest_expression - len(encoding)
                padded_statements[title] = (add_padding * [0]) + encoding
                padded_masking[title] = (add_padding * [0]) + masking[title]

            else:
                padded_masking[title] = masking[title]
                leng[title] = max_size

        logger.info(f"Total before: {len(bow)}")
        logger.info(f"Total after: {len(padded_statements)}")

        return padded_statements, padded_masking, leng
