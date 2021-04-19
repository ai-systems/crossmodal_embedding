from torch.utils.data import Dataset
from collections import Counter
import torch
import numpy as np


class InputData(Dataset):
    def __init__(self, df):
        # Assign vocabularies.
        self.e1 = torch.from_numpy(np.asarray(df["e1"].tolist())).long()
        self.e1_mask = torch.from_numpy(np.asarray(df["e1_mask"].tolist())).long()
        self.e1_len = torch.from_numpy(np.asarray(df["e1_len"].tolist())).long()
        self.e2 = torch.from_numpy(np.asarray(df["e2"].tolist())).long()
        self.e2_mask = torch.from_numpy(np.asarray(df["e2_mask"].tolist())).long()
        self.e2_len = torch.from_numpy(np.asarray(df["e2_len"].tolist())).long()
        self.score = torch.from_numpy(np.asarray(df["score"].tolist())).long()

    def __len__(self):
        return len(self.score)

    def __getitem__(self, idx):
        # Split sentence into words.
        e1 = self.e1[idx]
        e1_mask = self.e1_mask[idx]
        e1_len = self.e1_len[idx]
        e2 = self.e2[idx]
        e2_mask = self.e2_mask[idx]
        e2_len = self.e2_len[idx]
        score = self.score[idx]

        return e1, e1_mask, e1_len, e2, e2_mask, e2_len, score


class InputDataBert(Dataset):
    def __init__(self, df):
        # Assign vocabularies.
        self.e1 = torch.from_numpy(np.asarray(df["e"].tolist())).long()
        self.types = torch.from_numpy(np.asarray(df["types"].tolist())).long()
        self.att = torch.from_numpy(np.asarray(df["att"].tolist())).long()

        self.score = torch.from_numpy(np.asarray(df["score"].tolist())).long()

    def __len__(self):
        return len(self.score)

    def __getitem__(self, idx):
        # Split sentence into words.
        e1 = self.e1[idx]
        types = self.types[idx]
        att = self.att[idx]
        score = self.score[idx]

        return e1, types, att, score


class InputDataTest(Dataset):
    def __init__(self, df, vocab_size):
        exp1 = df["e1"].tolist()
        exp2 = df["e2"].tolist()
        new_exp1 = list()
        for e1_list in exp1:
            new_element = list()
            for e in e1_list:
                if e >= vocab_size:
                    new_element.append(0)
                else:
                    new_element.append(e)
            new_exp1.append(new_element)

        new_exp2 = list()
        for e2_list in exp2:
            new_element = list()
            for e in e2_list:
                if e >= vocab_size:
                    new_element.append(0)
                else:
                    new_element.append(e)
            new_exp2.append(new_element)

        # Assign vocabularies.
        self.e1 = torch.from_numpy(np.asarray(new_exp1)).long()
        self.e1_mask = torch.from_numpy(np.asarray(df["e1_mask"].tolist())).long()
        self.e2 = torch.from_numpy(np.asarray(new_exp2)).long()
        self.e2_mask = torch.from_numpy(np.asarray(df["e2_mask"].tolist())).long()
        self.score = torch.from_numpy(np.asarray(df["score"].tolist())).long()
        self.e1_len = torch.from_numpy(np.asarray(df["e1_len"].tolist())).long()
        self.e2_len = torch.from_numpy(np.asarray(df["e2_len"].tolist())).long()

    def __len__(self):
        return len(self.e1)

    def __getitem__(self, idx):
        # Split sentence into words.
        e1 = self.e1[idx]
        e1_mask = self.e1_mask[idx]
        e1_len = self.e1_len[idx]
        e2 = self.e2[idx]
        e2_mask = self.e2_mask[idx]
        e2_len = self.e2_len[idx]
        score = self.score[idx]

        return e1, e1_mask, e1_len, e2, e2_mask, e2_len, score


class InputDataTestBert(Dataset):
    def __init__(self, df):
        self.e1 = torch.from_numpy(np.asarray(df["e"].tolist())).long()
        self.types = torch.from_numpy(np.asarray(df["types"].tolist())).long()
        self.att = torch.from_numpy(np.asarray(df["att"].tolist())).long()

        self.score = torch.from_numpy(np.asarray(df["score"].tolist())).long()

    def __len__(self):
        return len(self.e1)

    def __getitem__(self, idx):
        # Split sentence into words.
        e1 = self.e1[idx]
        types = self.types[idx]
        att = self.att[idx]
        score = self.score[idx]

        return e1, types, att, score


class InputDataBertMixed(Dataset):
    def __init__(self, df):
        # Assign vocabularies.

        self.e1_s = torch.from_numpy(np.asarray(df["e1_s"].tolist())).long()
        self.e1_x = torch.from_numpy(np.asarray(df["e1_x"].tolist())).long()
        self.types_1 = torch.from_numpy(np.asarray(df["types_1"].tolist())).long()
        self.att_1 = torch.from_numpy(np.asarray(df["att_1"].tolist())).long()
        self.e2_s = torch.from_numpy(np.asarray(df["e2_s"].tolist())).long()
        self.e2_x = torch.from_numpy(np.asarray(df["e2_x"].tolist())).long()
        self.types_2 = torch.from_numpy(np.asarray(df["types_2"].tolist())).long()
        self.att_2 = torch.from_numpy(np.asarray(df["att_2"].tolist())).long()

        self.score = torch.from_numpy(np.asarray(df["score"].tolist())).long()

    def __len__(self):
        return len(self.score)

    def __getitem__(self, idx):
        # Split sentence into words.
        e1_s = self.e1_s[idx]
        e1_x = self.e1_x[idx]
        types_1 = self.types_1[idx]
        att_1 = self.att_1[idx]

        e2_s = self.e2_s[idx]
        e2_x = self.e2_x[idx]
        types_2 = self.types_2[idx]
        att_2 = self.att_2[idx]

        score = self.score[idx]

        return e1_s, e1_x, types_1, att_1, e2_s, e2_x, types_2, att_2, score


class InputDataTestBertMixed(Dataset):
    def __init__(self, df):
        self.e1_s = torch.from_numpy(np.asarray(df["e1_s"].tolist())).long()
        self.e1_x = torch.from_numpy(np.asarray(df["e1_x"].tolist())).long()
        self.types_1 = torch.from_numpy(np.asarray(df["types_1"].tolist())).long()
        self.att_1 = torch.from_numpy(np.asarray(df["att_1"].tolist())).long()
        self.e2_s = torch.from_numpy(np.asarray(df["e2_s"].tolist())).long()
        self.e2_x = torch.from_numpy(np.asarray(df["e2_x"].tolist())).long()
        self.types_2 = torch.from_numpy(np.asarray(df["types_2"].tolist())).long()
        self.att_2 = torch.from_numpy(np.asarray(df["att_2"].tolist())).long()

        self.score = torch.from_numpy(np.asarray(df["score"].tolist())).long()

    def __len__(self):
        return len(self.e1_s)

    def __getitem__(self, idx):
        # Split sentence into words.
        e1_s = self.e1_s[idx]
        e1_x = self.e1_x[idx]
        types_1 = self.types_1[idx]
        att_1 = self.att_1[idx]

        e2_s = self.e2_s[idx]
        e2_x = self.e2_x[idx]
        types_2 = self.types_2[idx]
        att_2 = self.att_2[idx]

        score = self.score[idx]

        return e1_s, e1_x, types_1, att_1, e2_s, e2_x, types_2, att_2, score


class InputDataPair(Dataset):
    def __init__(self, df):
        # Assign vocabularies.
        self.e = torch.from_numpy(np.asarray(df["e"].tolist())).long()
        self.mask = torch.from_numpy(np.asarray(df["e_mask"].tolist())).long()
        self.score = torch.from_numpy(np.asarray(df["score"].tolist())).long()

    def __len__(self):
        return len(self.score)

    def __getitem__(self, idx):
        # Split sentence into words.
        e = self.e[idx]
        mask = self.mask[idx]

        score = self.score[idx]

        return e, mask, score


class InputDataTestPair(Dataset):
    def __init__(self, df):
        self.e = torch.from_numpy(np.asarray(df["e"].tolist())).long()
        self.mask = torch.from_numpy(np.asarray(df["e_mask"].tolist())).long()
        self.score = torch.from_numpy(np.asarray(df["score"].tolist())).long()

    def __len__(self):
        return len(self.e)

    def __getitem__(self, idx):
        # Split sentence into words.
        e = self.e[idx]
        mask = self.mask[idx]

        score = self.score[idx]

        return e, mask, score

