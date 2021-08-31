import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from .FFNet import FFNet
from loguru import logger
from torch.autograd import Variable

torch.utils.backcompat.broadcast_warning.enabled = True
torch.set_printoptions(threshold=5000)


class CrossModalEmbedding(nn.Module):
    def __init__(
        self,
        batch_size,
        vocabulary_size,
        max_len=100,
        hidden_size=200,
        out_embedding=128,
        device="cuda",
        attention_heads=2,
        word_embedding=100,
        glove=None,
    ):
        super(CrossModalEmbedding, self).__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.device = device
        self.out_embedding = out_embedding
        self.hidden_size = hidden_size
        self.num_att_heads_self_att = attention_heads
        self.word_embedding = word_embedding
        self.vocabulary_size = vocabulary_size
        self.bidirection = 1
        self.m = nn.BatchNorm1d(self.max_len)
        self.initial_embedding_w = nn.Embedding(
            vocabulary_size, self.word_embedding, padding_idx=0
        )

        self.initial_embedding_e = nn.Embedding(
            vocabulary_size, self.word_embedding, padding_idx=0
        )

        self.bilstm = nn.LSTM(
            self.word_embedding * 2,
            hidden_size,
            1,
            bidirectional=True,
            batch_first=True,
        )

        self.self_attention_expressions = nn.MultiheadAttention(
            self.word_embedding,
            self.num_att_heads_self_att,
            dropout=0.5,
            add_bias_kv=True,
        )
        self.self_attention_words = nn.MultiheadAttention(
            self.word_embedding,
            self.num_att_heads_self_att,
            dropout=0.5,
            add_bias_kv=True,
        )

        self.norm_e1 = nn.LayerNorm(self.word_embedding)
        self.norm_e2 = nn.LayerNorm(self.word_embedding)
        self.norm_w1 = nn.LayerNorm(self.word_embedding)
        self.norm_w2 = nn.LayerNorm(self.word_embedding)

        self.ff_e = FFNet(self.word_embedding, self.word_embedding)
        self.ff_w = FFNet(self.word_embedding, self.word_embedding)

        # self.linear_layer = nn.Linear(self.hidden_size // 2, out_embedding)
        self.linear_layer = nn.Linear(self.hidden_size * 2, out_embedding)

        self.maxpool = nn.MaxPool1d(4)
        self.maxpool_statement = nn.MaxPool1d(2)

        self.dropout = nn.Dropout()
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()

        self.m = nn.BatchNorm1d(self.hidden_size * 2)
        logger.info("****************PARAMETERS********************")
        logger.info(f"HIDDEN SIZE: {self.hidden_size}")
        logger.info(f"BATCH SIZE: {self.batch_size}")
        logger.info(f"ATTENTION HEADS: {self.num_att_heads_self_att}")
        logger.info(f"OUT EMBEDDING: {self.out_embedding}")
        logger.info(f"VOCABULARY SIZE: {self.vocabulary_size }")
        logger.info(f"WORD EMBEDDING: {self.word_embedding }")
        logger.info("*************************************")

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(2, self.batch_size, self.hidden_size)
        hidden_b = torch.randn(2, self.batch_size, self.hidden_size)

        hidden_a = hidden_a.cuda()
        hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, statement, masking, st_len):

        self.hidden = self.init_hidden()

        lots_zeros = torch.zeros_like(statement).to(self.device)

        selected_expressions = torch.where(masking == 1, statement, lots_zeros)
        selected_expressions = self.initial_embedding_e(selected_expressions)
        selected_expressions = self.dropout1(selected_expressions)

        selected_words = torch.where(masking == 0, statement, lots_zeros)
        selected_words = self.initial_embedding_w(selected_words)
        selected_words = self.dropout2(selected_words)

        expressions_statement = selected_expressions.transpose(0, 1)
        attention_expressions, _ = self.self_attention_expressions(
            expressions_statement, expressions_statement, expressions_statement
        )
        # ATTENDED EXPRESSIONS
        attended_expressions = self.norm_e1(
            expressions_statement + attention_expressions
        )
        # TRANSPOSE ATTENNDED EXPRESSIONS
        attended_expressions = attended_expressions.transpose(0, 1)
        # FF ATTENDED EXPRESSIONS
        attended_expressions_ff = self.ff_e(attended_expressions)
        # EXPRESSION STATEMENT
        expressions_statement = self.norm_e2(
            attended_expressions + attended_expressions_ff
        )

        ##### WORDS #####
        words_statement = selected_words.transpose(0, 1)
        # WORDS ATTENTION
        attention_words, _ = self.self_attention_words(
            words_statement, words_statement, words_statement,
        )

        # ATTENDED WORDS
        attended_words = self.norm_w1(words_statement + attention_words)
        # ATTENDED WORDS TRANSPSOED
        attended_words = attended_words.transpose(0, 1)
        # ATTENDED WORDS FF
        attended_words_ff = self.ff_w(attended_words)

        # WORDS
        words_statement = self.norm_w2(attended_words + attended_words_ff)

 
        statement = torch.cat([expressions_statement, words_statement], dim=2)
       
        statement_in = torch.nn.utils.rnn.pack_padded_sequence(
            statement, st_len, batch_first=True, enforce_sorted=False
        )

        self.bilstm.flatten_parameters()
        output_lstm, (hn, cn) = self.bilstm(statement_in, self.hidden)


        emb = torch.cat([hn[0], hn[1]], dim=1)

        emb = self.m(emb)
        embedding = self.linear_layer(emb)

        return (
            embedding,
            attention_expressions.transpose(0, 1),
            attention_words.transpose(0, 1),
        )


class SiameseNet(nn.Module):
    def __init__(
        self,
        out_embedding_size,
        batch_size,
        vocabulary_size,
        max_len=100,
        hidden_size=200,
        out_embedding=128,
        device="cuda",
        attention_heads=2,
        word_embedding=100,
        glove=None,
    ):
        super(SiameseNet, self).__init__()
        self.embedding_net = CrossModalEmbedding(
            batch_size,
            vocabulary_size,
            max_len=max_len,
            hidden_size=hidden_size,
            out_embedding=out_embedding,
            device=device,
            attention_heads=attention_heads,
            word_embedding=word_embedding,
            glove=glove,
        )

        self.embedding = out_embedding * 4
        self.linear_layer = nn.Linear(out_embedding_size * 4, 2)
        self.act = nn.LogSoftmax(dim=1)

        self.dropout = nn.Dropout()

    def forward(self, x1, x1_mask, x1_len, x2, x2_mask, x2_len):
        output1, _, _ = self.embedding_net(x1, x1_mask, x1_len)
        output2, _, _ = self.embedding_net(x2, x2_mask, x2_len)
        sim = torch.cat(
            [output1, output2, torch.abs(output1 - output2), output1 * output2], dim=1
        )

        sim = self.dropout(sim)
        output = self.linear_layer(sim)

        output = self.act(output)

        return output

    def get_embedding(self, x, x_mask, x_len):
        return self.embedding_net(x, x_mask, x_len)[0]

    def get_attentions(self, x, x_mask, x_len):
        output = self.embedding_net(x, x_mask, x_len)
        return {
            "att_exp": torch.mean(output[1], dim=2).cpu().tolist(),
            "att_words": torch.mean(output[2], dim=2).cpu().tolist(),
        }
