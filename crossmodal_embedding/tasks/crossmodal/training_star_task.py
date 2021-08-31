from prefect import Task
from loguru import logger
from tqdm import tqdm
from crossmodal_embedding.models import CrossModalEmbedding, SiameseNet
from crossmodal_embedding.models import InputData, InputDataTest
from sklearn.metrics import precision_recall_fscore_support, f1_score
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn as nn
from crossmodal_embedding.util.evaluation import (
    compute_map_basic,
    compute_map_with_unification,
)
from torch.utils.data import WeightedRandomSampler
import sys
import json

from torch.utils.tensorboard import SummaryWriter


class TrainingTaskStar(Task):
    def create_weights(self, df):
        positives = 0
        negatives = 0
        weights = list()
        for index, row in df.iterrows():
            if row["score"] == 0:
                negatives = negatives + 1
            else:
                positives = positives + 1

        weight_positive = 1.0 / float(positives)
        weight_negative = 1.0 / float(negatives)

        for index, row in df.iterrows():
            if row["score"] == 0:
                weights.append(weight_negative)
            else:
                weights.append(weight_positive)
        return torch.tensor(weights)

    def run(
        self,
        train,
        test,
        dev,
        num_negatives,
        output_log,
        output_model,
        vocab_size,
        batch_size=10,
        num_epochs=5,
        learning_rate=0.0001,
        max_sequence_len=100,
        hidden_size=10,
        out_embedding=128,
        attention_heads=5,
        word_embedding=50,
        decay=0.01,
    ):

        logger.info(f" Negative Examples: {num_negatives}")
        logger.info("Let's train the Cross-Modal Embedding ! (^・ω・^ )")
        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Check for multi_GPUS
        multiple_gpus = 0

        train_class_weight = self.create_weights(train)

        train_dataset = InputData(train)
        logger.info(f"TRAIN: {len(train_dataset)}")
        dev_dataset = InputData(dev)
        logger.info(f"DEV: {len(dev_dataset)}")
        test_dataset = InputDataTest(test, vocab_size)
        logger.info(f"TEST: {len(test_dataset)}")
        sampler_train = WeightedRandomSampler(
            train_class_weight, len(train_class_weight)
        )

        # Data loader

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, sampler=sampler_train,
        )

        dev_loader = torch.utils.data.DataLoader(
            dataset=dev_dataset, batch_size=batch_size, shuffle=False
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False
        )

        model = SiameseNet(
            out_embedding,
            batch_size,
            vocab_size,
            max_len=max_sequence_len,
            hidden_size=hidden_size,
            out_embedding=out_embedding,
            device=device,
            attention_heads=attention_heads,
            word_embedding=word_embedding,
        )

        if torch.cuda.device_count() > 1:
            logger.info(
                f"**********Let's use {torch.cuda.device_count()} GPUs!********"
            )
            multiple_gpus = 1
            model = nn.DataParallel(model)
        else:
            logger.info("********* Only one GPU *******")

        model = model.to(device)

        # Loss and optimizer
        criterion = nn.NLLLoss()

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", verbose=True, patience=1, cooldown=3
        )

        # Train the model
        best_value = 0
        all_best = dict()
        result_dict = dict()
        total_step = len(train_loader)
        for epoch in tqdm(range(num_epochs), desc=f"Epoch"):
            epoch_loss = 0.0
            running_loss = 0.0

            model.train()
            t = tqdm(iter(train_loader), leave=False, total=len(train_loader))
            for (
                i,
                (statement1, st1_mask, st1_len, statement2, st2_mask, st2_len, score),
            ) in enumerate(t):

                # Move tensors to the configured device
                statement1 = statement1.to(device)
                st1_mask = st1_mask.to(device)
                st1_len = st1_len.to(device)
                statement2 = statement2.to(device)
                st2_mask = st2_mask.to(device)
                st2_len = st2_len.to(device)

                score = score.to(device)
                optimizer.zero_grad()
                sim = model(
                    statement1, st1_mask, st1_len, statement2, st2_mask, st2_len
                )

                loss = criterion(sim, score)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 0:  
                    t.set_description("loss: {:.4f}".format(running_loss / 10))
                    running_loss = 0

            logger.info(
                f"********Epoch: {epoch+1} *****Loss: {epoch_loss / len(train_loader)}"
            )
            result_dict[epoch] = dict()
            result_dict[epoch]["train_loss"] = epoch_loss / len(train_loader)

            scheduler.step(epoch_loss / len(train_loader))
            if (epoch + 1) % 1 == 0:
                model.eval()
                with torch.no_grad():

                    logger.info("Evaluating on Train set!")
                    t = tqdm(iter(train_loader), leave=False, total=len(train_loader))
                    y_pred_list = []
                    y_real_list = []
                    for (
                        i,
                        (
                            statement1,
                            st1_mask,
                            st1_len,
                            statement2,
                            st2_mask,
                            st2_len,
                            score,
                        ),
                    ) in enumerate(t):

                        # Move tensors to the configured device
                        statement1 = statement1.to(device)
                        st1_mask = st1_mask.to(device)
                        st1_len = st1_len.to(device)
                        statement2 = statement2.to(device)
                        st2_mask = st2_mask.to(device)
                        st2_len = st2_len.to(device)

                        y_real_list.extend(score.cpu().tolist())
                        score = score.to(device)

                        sim = model(
                            statement1, st1_mask, st1_len, statement2, st2_mask, st2_len
                        )
                        y_dev_pred = torch.argmax(sim, dim=1)
                        # y_dev_pred = torch.argmax(sim, dim=1)
                        y_pred_list.extend(y_dev_pred.cpu().tolist())

                    f1_value = f1_score(y_real_list, y_pred_list)
                    (precision, recall, _, _,) = precision_recall_fscore_support(
                        y_real_list, y_pred_list, average="binary"
                    )
                    # logger.info("**** TRAINING SET **** ")
                    # logger.info(f"F1-value: {f1_value}")
                    # logger.info(f"Precision: {precision}")
                    # logger.info(f"Recall: {recall}")

                    logger.info("Evaluating on Dev set!")

                    t = tqdm(iter(dev_loader), leave=False, total=len(dev_loader))
                    y_pred_list = []
                    y_real_list = []
                    epoch_test_loss = 0.0
                    for (
                        i,
                        (
                            statement1,
                            st1_mask,
                            st1_len,
                            statement2,
                            st2_mask,
                            st2_len,
                            score,
                        ),
                    ) in enumerate(t):

                        statement1 = statement1.to(device)
                        st1_mask = st1_mask.to(device)
                        st1_len = st1_len.to(device)
                        statement2 = statement2.to(device)
                        st2_mask = st2_mask.to(device)
                        st2_len = st2_len.to(device)

                        y_real_list.extend(score.cpu().tolist())
                        score = score.to(device)

                        sim = model(
                            statement1, st1_mask, st2_len, statement2, st2_mask, st2_len
                        )
                        loss_test = criterion(sim, score)
                        epoch_test_loss += loss_test.item()
                        y_dev_pred = torch.argmax(sim, dim=1)
                        y_pred_list.extend(y_dev_pred.cpu().tolist())

                    logger.info(f"DEV LOSS: {epoch_test_loss / len(dev_loader)}")
                    # scheduler.step(epoch_test_loss / len(dev_loader))
                    f1_value = f1_score(y_real_list, y_pred_list)
                    (precision, recall, _, _,) = precision_recall_fscore_support(
                        y_real_list, y_pred_list, average="binary"
                    )
                    # logger.info("**** DEV SET **** ")
                    # logger.info(f"F1-value: {f1_value}")
                    # logger.info(f"Precision: {precision.tolist()}")
                    # logger.info(f"Recall: {recall.tolist()}")
                    result_dict[epoch]["f1"] = f1_value
                    result_dict[epoch]["precision"] = precision.tolist()
                    result_dict[epoch]["recall"] = recall.tolist()

                if f1_value > best_value:
                    best_value = f1_value
                    model = model.to("cpu")
                    if multiple_gpus:
                        torch.save(
                            model.module.state_dict(), f"./models/{output_model}",
                        )
                    else:
                        torch.save(
                            model.state_dict(), f"./models/{output_model}",
                        )

                    all_best["f1"] = f1_value
                    all_best["precision"] = precision.tolist()
                    all_best["recall"] = recall.tolist()
                    model = model.to(device)
                    best_model = model

        with torch.no_grad():
            best_model.eval()
            logger.info("Evaluating on Test set!")
            all_embeddings = dict()
            t = tqdm(iter(test_loader), leave=False, total=len(test_loader))
            y_pred_list = []
            y_real_list = []
            for (
                i,
                (statement1, st1_mask, st1_len, statement2, st2_mask, st2_len, score),
            ) in enumerate(t):

                # Move tensors to the configured device
                statement1 = statement1.to(device)
                st1_mask = st1_mask.to(device)
                st1_len = st1_len.to(device)
                statement2 = statement2.to(device)
                st2_mask = st2_mask.to(device)
                st2_len = st2_len.to(device)

                y_real_list.extend(score.cpu().tolist())
                score = score.to(device)

                sim = best_model(
                    statement1, st1_mask, st1_len, statement2, st2_mask, st2_len
                )
                # y_dev_pred = torch.round(sim)
                y_dev_pred = torch.argmax(sim, dim=1)
                y_pred_list.extend(y_dev_pred.cpu().tolist())

            f1_value = f1_score(y_real_list, y_pred_list)
            (precision, recall, _, _,) = precision_recall_fscore_support(
                y_real_list, y_pred_list, average="binary"
            )

            logger.info("****** PARAMETERS ********")
            logger.info(f"Num negatives: {num_negatives}")
            logger.info(f"Batch_size: {batch_size}")
            logger.info(f"Max len: {max_sequence_len}")
            logger.info(f"Word embedding: {word_embedding}")
            logger.info(f"Out embedding: {out_embedding}")
            logger.info(f"Hidden Size: {hidden_size}")
            logger.info(f"Decay: {decay}")
            logger.info(f"ATT heads: {attention_heads}")
            logger.info(f"Learning rate: {learning_rate}")
            logger.info("****** BEST RESULTS TEST******")
            logger.info(f"F1 SCORE {f1_value}")
            logger.info(f"PRECISION: {precision}")
            logger.info(f"RECALL: {recall}")
            all_best["f1_test"] = f1_value
            all_best["precision_test"] = precision.tolist()
            all_best["recall_test"] = recall.tolist()

            logger.info("******** BEST RESULTS DEV **********")
            logger.info(all_best)

        with open(f"./logs/{output_log}", "w") as f:
            json.dump(result_dict, f)
        with open(f"./logs/best_{output_log}", "w") as f:
            json.dump(result_dict, f)
