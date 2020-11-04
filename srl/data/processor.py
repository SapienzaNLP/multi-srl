import json

import torch
from torch.nn.utils.rnn import pad_sequence
from allennlp.modules.elmo import batch_to_ids
from transformers import AutoTokenizer

from utils.decoding import viterbi_decode


class Processor(object):

    def __init__(
            self,
            dataset={},
            input_representation='bert_embeddings',
            model_name='bert-base-multilingual-cased',
            unknown_token='<unk>'):

        super(Processor, self).__init__()

        self.input_representation = input_representation
        self.padding_target_id = -1

        if input_representation == 'elmo_embeddings':
            self.padding_token_id = 0
            self.unknown_token_id = 0

        elif input_representation == 'bert_embeddings':
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.padding_token_id = self.tokenizer.pad_token_id
            self.unknown_token_id = self.tokenizer.unk_token_id

        else:
            raise ValueError('Unsupported value for input_representation: {}'.format(input_representation))

        output_maps = Processor._build_output_maps(dataset)

        self.predicate2id = output_maps['predicate2id']
        self.id2predicate = output_maps['id2predicate']
        self.unknown_predicate_id = self.predicate2id[unknown_token]
        self.num_senses = len(self.predicate2id)

        self.role2id = output_maps['role2id']
        self.id2role = output_maps['id2role']
        self.unknown_role_id = self.role2id[unknown_token]
        self.num_roles = len(self.role2id)

        self.role_transition_matrix = output_maps['role_transition_matrix']
        self.role_start_transitions = output_maps['role_start_transitions']

    def encode_sentence(self, sentence):
        '''
            Given a sentence object, returns an input sample.
        '''
        # word_ids is a list of integer ids, one for each (sub)token.
        word_ids = []
        # subword_indices is a list of incrementing indices, one for each (sub)token.
        # Each index indicates the index in the original sentence of the word the subtoken belongs to.
        # Ex: ['This', 'is', 'an', 'em', '##bed', '##ding', '.']
        #     [0,      1,     2,   3,    3,       3,        4]
        subword_indices = []

        # Length of the sentence in words.
        sequence_length = len(sentence['words'])
        # Add [CLS] and [SEP] to the original length.
        sequence_length += 2

        if self.input_representation == 'elmo_embeddings':
            word_ids = batch_to_ids([sentence['words']]).squeeze(0)

        elif self.input_representation == 'bert_embeddings':
            # List of subtokens.
            tokenized_sentence = []
            # List of the original word indices of each subtoken.
            # Initialized with the word index of the [CLS] token.
            subword_indices = [0]
            word_index = 1
            for word_index, word in enumerate(sentence['words'], 1):
                tokenized_word = self.tokenizer.tokenize(word)
                tokenized_sentence.extend(tokenized_word)
                subword_indices.extend([word_index]*len(tokenized_word))
                
                # Cut the tokenized sentence to at most 500 subtokens.
                # Most language models have a limit of 512 subtokens.
                if len(tokenized_sentence) > 500:
                    tokenized_sentence = tokenized_sentence[:500]
                    subword_indices = subword_indices[:500]
                    sequence_length = word_index
                    break
            # Add the word index of the [SEP] token.
            subword_indices.append(word_index + 1)
            # Map the tokenized sentence to word ids.
            # tokenizer.encode() automatically adds the [CLS] and [SEP] tokens.
            word_ids = self.tokenizer.encode(tokenized_sentence)
        
        else:
            raise ValueError('Unsupported value for input_representation: {}'.format(self.input_representation))

        return {
            'word_ids': torch.as_tensor(word_ids),
            'subword_indices': torch.as_tensor(subword_indices),
            'sequence_length': sequence_length,
            'tokenized_sequence_length': len(word_ids) - 1,
        }

    def encode_labels(self, sentence):
        '''
            Given a sentence object, returns its label ids (senses and roles).
        '''
        # List of 0s and 1s to indicate the predicates in the input sentence.
        predicates = []
        # List of sense ids.
        senses = []
        # List of predicate indices.
        predicate_indices = []

        for word_index, predicate in enumerate(sentence['predicates']):
            if predicate != '_':
                predicates.append(1)
                predicate_indices.append(word_index)
            else:
                predicates.append(0)

            if predicate in self.predicate2id:
                senses.append(self.predicate2id[predicate])
            else:
                senses.append(self.unknown_predicate_id)

        sentence_length = len(predicates)
        # Initialize all role ids to the padding target id (ignore).
        roles = [[self.padding_target_id] * sentence_length] * sentence_length

        for predicate_index, predicate_roles in sentence['roles'].items():
            predicate_role_ids = []
            for role in predicate_roles:
                if role in self.role2id:
                    predicate_role_ids.append(self.role2id[role])
                else:
                    predicate_role_ids.append(self.unknown_role_id)
            roles[predicate_index] = predicate_role_ids

        return {
            'predicate_indices': predicate_indices,
            'predicates': torch.as_tensor(predicates),
            'senses': torch.as_tensor(senses),
            'roles': roles,
        }

    def decode(self, x, y, viterbi_decoding=False):
        '''
            Given a sample and its label ids (in a batch), returns the labels.
        '''
        word_ids = x['word_ids']
        sentence_lengths = x['sequence_lengths']
        predicate_indices = list(map(list, zip(*x['predicate_indices'])))

        predicates = []
        predicate_ids = torch.argmax(y['predicates'], dim=-1).tolist()
        for sentence_predicate_ids, sentence_length in zip(predicate_ids, sentence_lengths):
            sentence_predicate_ids = sentence_predicate_ids[:sentence_length]
            predicates.append([p for p in sentence_predicate_ids])

        senses = [['_']*sentence_length for sentence_length in sentence_lengths]
        sense_ids = torch.argmax(y['senses'], dim=-1).tolist()
        for (sentence_index, predicate_index), sense_id in zip(predicate_indices, sense_ids):
            senses[sentence_index][predicate_index] = self.id2predicate[sense_id]

        roles = {i: {} for i in range(len(word_ids))}
        if not viterbi_decoding:
            role_ids = torch.argmax(y['roles'], dim=-1).tolist()
            for (sentence_index, predicate_index), predicate_role_ids in zip(predicate_indices, role_ids):
                sentence_length = sentence_lengths[sentence_index]
                predicate_role_ids = predicate_role_ids[:sentence_length]
                predicate_roles = [self.id2role[r] for r in predicate_role_ids]
                roles[sentence_index][predicate_index] = predicate_roles
        else:
            role_emissions = y['roles']
            for (sentence_index, predicate_index), predicate_role_emissions in zip(predicate_indices, role_emissions):
                sentence_length = sentence_lengths[sentence_index]
                predicate_role_emissions = predicate_role_emissions[:sentence_length]
                predicate_role_ids, _ = viterbi_decode(
                    predicate_role_emissions.to('cpu'),
                    torch.as_tensor(self.role_transition_matrix),
                    allowed_start_transitions=torch.as_tensor(self.role_start_transitions))
                predicate_roles = [self.id2role[r] for r in predicate_role_ids]
                roles[sentence_index][predicate_index] = predicate_roles

        return {
            'predicates': predicates,
            'senses': senses,
            'roles': roles,
        }

    def collate_sentences(self, sentences):
        batched_x = {
            'sentence_ids': [],
            'predicate_indices': [[], []],

            'word_ids': [],
            'subword_indices': [],
            'sequence_lengths': [],
            'tokenized_sequence_lengths': [],
        }

        batched_y = {
            'predicates': [],
            'senses': [],
            'roles': [],
        }

        max_sequence_length = 0
        for sentence_index, sentence in enumerate(sentences):
            encoded_sentence = self.encode_sentence(sentence)
            encoded_labels = self.encode_labels(sentence)

            max_sequence_length = max(encoded_sentence['sequence_length'], max_sequence_length)

            batched_x['sentence_ids'].append(sentence['sentence_id'])
            batched_x['predicate_indices'][0].extend([sentence_index]*len(encoded_labels['predicate_indices']))
            batched_x['predicate_indices'][1].extend(encoded_labels['predicate_indices'])

            batched_x['word_ids'].append(encoded_sentence['word_ids'])
            batched_x['subword_indices'].append(encoded_sentence['subword_indices'])
            batched_x['sequence_lengths'].append(encoded_sentence['sequence_length'])
            batched_x['tokenized_sequence_lengths'].append(encoded_sentence['tokenized_sequence_length'])

            batched_y['predicates'].append(encoded_labels['predicates'])
            batched_y['senses'].append(encoded_labels['senses'])
            batched_y['roles'].append(encoded_labels['roles'])

        if self.input_representation == 'elmo_embeddings':
            batched_x['word_ids'] = pad_sequence(batched_x['word_ids'], batch_first=True)
        elif self.input_representation == 'bert_embeddings':
            batched_x['word_ids'] = pad_sequence(batched_x['word_ids'], batch_first=True, padding_value=self.padding_token_id)

        batched_x['sequence_lengths'] = torch.as_tensor(batched_x['sequence_lengths'])
        batched_x['tokenized_sequence_lengths'] = torch.as_tensor(batched_x['tokenized_sequence_lengths'])

        batched_x['subword_indices'] = pad_sequence(
            batched_x['subword_indices'],
            batch_first=True,
            padding_value=max_sequence_length - 1)
        batched_y['predicates'] = pad_sequence(
            batched_y['predicates'],
            batch_first=True,
            padding_value=self.padding_target_id)
        batched_y['senses'] = pad_sequence(
            batched_y['senses'],
            batch_first=True,
            padding_value=self.padding_target_id)
        batched_y['roles'] = Processor._pad_bidimensional_sequences(
            batched_y['roles'],
            sequence_length=max_sequence_length - 2,
            padding_value=self.padding_target_id)

        return batched_x, batched_y

    def save_config(self, path):

        config = {
            'input_representation': self.input_representation,
            'padding_target_id': self.padding_target_id,
            'model_name': self.model_name if self.input_representation == 'bert_embeddings' else '',

            'padding_token_id': self.padding_token_id,
            'unknown_token_id': self.unknown_token_id,

            'predicate2id': self.predicate2id,
            'id2predicate': self.id2predicate,
            'unknown_predicate_id': self.unknown_predicate_id,
            'num_senses': self.num_senses,

            'role2id': self.role2id,
            'id2role': self.id2role,
            'unknown_role_id': self.unknown_role_id,
            'num_roles': self.num_roles,

            'role_transition_matrix': self.role_transition_matrix,
            'role_start_transitions': self.role_start_transitions,
        }

        with open(path, 'w') as f:
            json.dump(config, f, sort_keys=True, indent=4)

    @staticmethod
    def from_config(path):
        with open(path) as f:
            config = json.load(f)

        processor = Processor()
        processor.input_representation = config['input_representation']
        processor.padding_target_id = config['padding_target_id']
        processor.model_name = config['model_name']

        processor.padding_token_id = config['padding_token_id']
        processor.unknown_token_id = config['unknown_token_id']

        processor.predicate2id = config['predicate2id']
        processor.id2predicate = {int(id): predicate for id, predicate in config['id2predicate'].items()}
        processor.unknown_predicate_id = config['unknown_predicate_id']
        processor.num_senses = config['num_senses']

        processor.role2id = config['role2id']
        processor.id2role = {int(id): role for id, role in config['id2role'].items()}
        processor.unknown_role_id = config['unknown_role_id']
        processor.num_roles = config['num_roles']

        processor.role_transition_matrix = config['role_transition_matrix']
        processor.role_start_transitions = config['role_start_transitions']

        if processor.input_representation == 'bert_embeddings':
            processor.tokenizer = AutoTokenizer.from_pretrained(processor.model_name)

        return processor

    @staticmethod
    def _build_output_maps(dataset, bio_tags=False):
        predicate2id = {
            '_': 0,
            '<unk>': 1,
        }
        role2id = {
            '_': 0,
            '<unk>': 1
        }
        predicates = set()

        role_counts = {}
        total_role_count = 0
        for i in range(len(dataset)):
            sentence = dataset[i]

            for predicate in sentence['predicates']:
                if predicate not in predicate2id:
                    predicate2id[predicate] = len(predicate2id)

            for roles in sentence['roles'].values():
                for role in roles:
                    if role not in role2id:
                        role2id[role] = len(role2id)
                    role_id = role2id[role]
                    if role_id not in role_counts:
                        role_counts[role_id] = 0
                    role_counts[role_id] += 1
                    total_role_count += 1

        id2predicate = {id: predicate for predicate, id in predicate2id.items()}
        id2role = {id: role for role, id in role2id.items()}

        if bio_tags:
            role_transition_matrix = []
            for i in range(len(id2role)):
                previous_label = id2role[i]
                role_transitions = []
                for j in range(len(id2role)):
                    label = id2role[j]
                    if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                        role_transitions.append(float('-inf'))
                    else:
                        role_transitions.append(0.0)
                role_transition_matrix.append(role_transitions)
            
            role_start_transitions = []
            for i in range(len(id2role)):
                label = id2role[i]
                if label[0] == "I":
                    role_start_transitions.append(float("-inf"))
                else:
                    role_start_transitions.append(0.0)
        else:
            role_transition_matrix = []
            for i in range(len(id2role)):
                role_transitions = []
                for j in range(len(id2role)):
                    label = id2role[j]
                    if i == j and label != '_':
                        role_transitions.append(float('-inf'))
                    else:
                        role_transitions.append(0.0)
                role_transition_matrix.append(role_transitions)
            
            role_start_transitions = []
            for i in range(len(id2role)):
                label = id2role[i]
                role_start_transitions.append(0.0)

        return {
            'predicate2id': predicate2id,
            'id2predicate': id2predicate,
            'role2id': role2id,
            'id2role': id2role,
            'predicates': predicates,
            'role_transition_matrix': role_transition_matrix,
            'role_start_transitions': role_start_transitions,
        }

    @staticmethod
    def _pad_bidimensional_sequences(sequences, sequence_length, padding_value=0):
        padded_sequences = torch.full((len(sequences), sequence_length, sequence_length), padding_value, dtype=torch.long)
        for i, sequence in enumerate(sequences):
            for j, subsequence in enumerate(sequence):
                padded_sequences[i][j][:len(subsequence)] = torch.as_tensor(subsequence, dtype=torch.long)
        return padded_sequences
