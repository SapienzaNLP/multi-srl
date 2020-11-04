import torch
import torch.nn as nn
import torch_scatter as scatter
from allennlp.modules.elmo import Elmo
from transformers import AutoModel, AutoConfig


class WordEncoder(nn.Module):

    def __init__(self, hparams):
        super(WordEncoder, self).__init__()

        self.input_representation = hparams.input_representation

        if hparams.input_representation == 'elmo_embeddings':
            self.word_embedding = ElmoEmbedding(fine_tune=hparams.language_model_fine_tuning)
            word_embedding_size = 1024

        elif hparams.input_representation == 'bert_embeddings':
            self.word_embedding = BertEmbedding(model_name=hparams.language_model, fine_tune=hparams.language_model_fine_tuning)
            if 'base' in hparams.language_model:
                word_embedding_size = 4*768
            else:
                word_embedding_size = 4*1024
        
        else:
            raise NotImplementedError('{} is not implemented.'.format(hparams.input_representation))

        self.batch_normalization = nn.BatchNorm1d(word_embedding_size)
        self.projection = nn.Linear(word_embedding_size, hparams.word_projection_size)
        self.word_dropout = nn.Dropout(hparams.word_dropout)

        self.word_embedding_size = hparams.word_projection_size

    def forward(self, word_ids, subword_indices=None, sequence_lengths=None):
        word_embeddings = self.word_embedding(word_ids, sequence_lengths=sequence_lengths)

        word_embeddings = word_embeddings.permute(0, 2, 1)
        word_embeddings = self.batch_normalization(word_embeddings)
        word_embeddings = word_embeddings.permute(0, 2, 1)

        word_embeddings = self.projection(word_embeddings)
        word_embeddings = word_embeddings * torch.sigmoid(word_embeddings)
        word_embeddings = self.word_dropout(word_embeddings)

        if self.input_representation == 'bert_embeddings':
            word_embeddings = scatter.scatter_mean(word_embeddings, subword_indices, dim=1)

        return word_embeddings


class ElmoEmbedding(nn.Module):

    _options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    _weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    def __init__(self, fine_tune=False):
        super(ElmoEmbedding, self).__init__()
        self.fine_tune = fine_tune
        self.word_embedding = Elmo(
            ElmoEmbedding._options_file,
            ElmoEmbedding._weight_file,
            num_output_representations=1,
            dropout=0.0,
            keep_sentence_boundaries=True,
            requires_grad=fine_tune)

    def forward(self, character_ids, sequence_lengths=None):
        if not self.fine_tune:
            with torch.no_grad():
                word_embeddings = self.word_embedding(character_ids)
        else:
            word_embeddings = self.word_embedding(character_ids)
        word_embeddings = word_embeddings['elmo_representations'][0]
        return word_embeddings


class BertEmbedding(nn.Module):

    def __init__(self, model_name='bert-base-multilingual-cased', fine_tune=False):
        super(BertEmbedding, self).__init__()
        self.fine_tune = fine_tune
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        if not fine_tune:
            self.bert.eval()

    def forward(self, word_ids, sequence_lengths=None):
        timesteps = word_ids.shape[1]
        attention_mask = torch.arange(timesteps).unsqueeze(0).cuda() < sequence_lengths.unsqueeze(1)

        if not self.fine_tune:
            with torch.no_grad():
                word_embeddings = self.bert(
                    input_ids=word_ids,
                    attention_mask=attention_mask)
        else:
            word_embeddings = self.bert(
                input_ids=word_ids,
                attention_mask=attention_mask)

        word_embeddings = torch.cat(word_embeddings[2][-4:], dim=-1)
        return word_embeddings
