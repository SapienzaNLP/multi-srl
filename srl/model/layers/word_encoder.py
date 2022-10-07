import torch
import torch.nn as nn
import torch_scatter as scatter
from transformers import AutoModel, AutoConfig

from srl.model.layers.swish import Swish


class WordEncoder(nn.Module):

    def __init__(
        self,
        language_model_name: str = 'bert-base-cased',
        language_model_fine_tuning: bool = False,
        output_size: int = 512,
        activation_type: str = 'swish',
        dropout_rate: float = 0.1,
    ):
        super(WordEncoder, self).__init__()
        self.language_model_name = language_model_name
        self.language_model_fine_tuning = language_model_fine_tuning
        self.output_size = output_size
        self.activation = activation_type
        self.dropout_rate = dropout_rate

        self.word_embedding_layer = BertEmbedding(model_name=self.language_model_name, fine_tune=self.language_model_fine_tuning)
        if 'base' in self.language_model_name:
            word_embedding_size = 4*768
        else:
            word_embedding_size = 4*1024

        self.normalization_layer = nn.LayerNorm(word_embedding_size)
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.projection_layer = nn.Linear(word_embedding_size, self.output_size, bias=False)

        if self.activation == 'identity':
            self.activation_layer = nn.Identity()
        elif self.activation == 'relu':
            self.activation_layer = nn.ReLU()
        elif self.activation == 'swish':
            self.activation_layer = Swish()

    def forward(self, lm_inputs, subword_indices):
        max_sequence_length = torch.max(subword_indices).item() + 1
        word_embeddings = self.word_embedding_layer(lm_inputs)
        word_embeddings = self.normalization_layer(word_embeddings)
        word_embeddings = self.projection_layer(word_embeddings)
        word_embeddings = self.activation_layer(word_embeddings)
        word_embeddings = scatter.scatter_mean(word_embeddings, subword_indices, dim=1)[:, :max_sequence_length, :]
        word_embeddings = self.dropout_layer(word_embeddings)
        return word_embeddings


class BertEmbedding(nn.Module):

    def __init__(self,
        model_name: str = 'bert-base-cased',
        fine_tune: bool = False
    ):
        super().__init__()
        self.fine_tune = fine_tune
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        if not fine_tune:
            self.bert.eval()

    def forward(self, lm_inputs):
        
        if not self.fine_tune:
            with torch.inference_mode():
                word_embeddings = self.bert(**lm_inputs)
        else:
            word_embeddings = self.bert(**lm_inputs)

        word_embeddings = torch.cat(word_embeddings.hidden_states[-4:], dim=-1)
        return word_embeddings
