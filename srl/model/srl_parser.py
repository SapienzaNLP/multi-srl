import json
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from srl.model.layers.word_encoder import WordEncoder
from srl.model.layers.sequence_encoder import SequenceEncoder
from srl.model.layers.state_encoder import StateEncoder
from srl.utils.decoding import viterbi_decode


class SrlParser(pl.LightningModule):

    def __init__(
        self,
        vocabulary_path: str,

        language_model_name: str = 'bert-base-cased',
        language_model_fine_tuning: bool = False,
        word_encoding_size: int = 512,
        word_encoding_activation: str = 'swish',
        word_encoding_dropout: float = 0.1,

        predicate_encoding_size: int = 128,
        predicate_encoding_activation: str = 'swish',
        predicate_encoding_dropout: float = 0.1,

        sense_encoding_size: int = 256,
        sense_encoding_activation: str = 'swish',
        sense_encoding_dropout: float = 0.2,

        role_encoding_size: int = 256,
        role_encoding_activation: str = 'swish',
        role_encoding_dropout: float = 0.2,

        predicate_timestep_encoding_size: int = 512,
        predicate_timestep_encoding_activation: str = 'swish',
        predicate_timestep_encoding_dropout: float = 0.1,

        argument_timestep_encoding_size: int = 512,
        argument_timestep_encoding_activation: str = 'swish',
        argument_timestep_encoding_dropout: float = 0.1,
    
        word_sequence_encoder_type: str = 'connected_lstm',
        word_sequence_encoder_hidden_size: int = 512,
        word_sequence_encoder_layers: int = 1,
        word_sequence_encoder_dropout: float = 0.2,

        argument_sequence_encoder_type: str = 'connected_lstm',
        argument_sequence_encoder_hidden_size: int = 512,
        argument_sequence_encoder_layers: int = 1,
        argument_sequence_encoder_dropout: float = 0.2,

        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        language_model_learning_rate: float = 5e-5,
        language_model_weight_decay: float = 1e-2,
        padding_label_id: int = -1,

        use_viterbi_decoding: bool = False,
        use_sense_candidates: bool = False,
        predictions_path: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._load_vocabulary(vocabulary_path)
        self._build_transition_matrices()

        self.use_viterbi_decoding = use_viterbi_decoding
        self.use_sense_candidates = use_sense_candidates
        self.predictions_path = predictions_path

        self.word_encoder = WordEncoder(
            language_model_name=language_model_name,
            language_model_fine_tuning=language_model_fine_tuning,
            output_size=word_encoding_size,
            activation_type=word_encoding_activation,
            dropout_rate=word_encoding_dropout)
        word_embedding_size = self.word_encoder.output_size

        self.sequence_encoder = SequenceEncoder(
            encoder_type=word_sequence_encoder_type,
            input_size=word_embedding_size,
            hidden_size=word_sequence_encoder_hidden_size,
            num_layers=word_sequence_encoder_layers,
            dropout=word_sequence_encoder_dropout)
        word_timestep_size = self.sequence_encoder.output_size

        self.predicate_encoder = StateEncoder(
            input_size=word_timestep_size,
            state_size=predicate_encoding_size,
            activation=predicate_encoding_activation,
            dropout_rate=predicate_encoding_dropout)

        self.sense_encoder = StateEncoder(
            input_size=word_timestep_size,
            state_size=sense_encoding_size,
            activation=sense_encoding_activation,
            dropout_rate=sense_encoding_dropout)

        self.predicate_timestep_encoder = StateEncoder(
            input_size=word_timestep_size,
            state_size=predicate_timestep_encoding_size,
            activation=predicate_timestep_encoding_activation,
            dropout_rate=predicate_timestep_encoding_dropout,
            bias=False)

        self.argument_timestep_encoder = StateEncoder(
            input_size=word_timestep_size,
            state_size=argument_timestep_encoding_size,
            activation=argument_timestep_encoding_activation,
            dropout_rate=argument_timestep_encoding_dropout,
            bias=False)

        self.argument_sequence_encoder = SequenceEncoder(
            encoder_type=argument_sequence_encoder_type,
            input_size=predicate_timestep_encoding_size + argument_timestep_encoding_size,
            hidden_size=argument_sequence_encoder_hidden_size,
            num_layers=argument_sequence_encoder_layers,
            dropout=argument_sequence_encoder_dropout)
        predicate_argument_timestep_size = self.argument_sequence_encoder.output_size

        self.argument_encoder = StateEncoder(
            input_size=predicate_argument_timestep_size,
            state_size=role_encoding_size,
            activation=role_encoding_activation,
            dropout_rate=role_encoding_dropout)

        self.predicate_scorer = nn.Linear(predicate_encoding_size, 2)
        self.sense_scorer = nn.Linear(sense_encoding_size, self.num_senses)
        self.role_scorer = nn.Linear(role_encoding_size, self.num_roles)


    def forward(self, x):
        lm_inputs = x['lm_inputs']
        subword_indices = x['subword_indices']
        sentence_lengths = x['sentence_lengths']

        word_embeddings = self.word_encoder(lm_inputs, subword_indices)
        sequence_states = self.sequence_encoder(word_embeddings, sentence_lengths)[:, 1:-1, :]

        predicate_encodings = self.predicate_encoder(sequence_states)
        predicate_scores = self.predicate_scorer(predicate_encodings)

        if 'predicates' in x:
            predicate_indices = x['predicates']
        else:
            predicate_indices = torch.argmax(predicate_scores, dim=-1)

        sense_encodings = sequence_states[predicate_indices == 1]
        sense_encodings = self.sense_encoder(sense_encodings)
        sense_scores = self.sense_scorer(sense_encodings)

        timesteps = sequence_states.shape[1]

        predicate_timestep_encodings = self.predicate_timestep_encoder(sequence_states)
        predicate_timestep_encodings = predicate_timestep_encodings.unsqueeze(2).expand(-1, -1, timesteps, -1)

        argument_timestep_encodings = self.argument_timestep_encoder(sequence_states)
        argument_timestep_encodings = argument_timestep_encodings.unsqueeze(1).expand(-1, timesteps, -1, -1)

        predicate_argument_states = torch.cat((predicate_timestep_encodings, argument_timestep_encodings), dim=-1)
        predicate_argument_states = predicate_argument_states[predicate_indices == 1]

        num_predicates = torch.sum(predicate_indices == 1, dim=1)
        argument_sequence_lengths = torch.repeat_interleave(sentence_lengths, num_predicates)
        max_argument_sequence_length = torch.max(argument_sequence_lengths)
        predicate_argument_states = predicate_argument_states[:, :max_argument_sequence_length, :]
        argument_encodings = self.argument_sequence_encoder(predicate_argument_states, argument_sequence_lengths)
        argument_encodings = self.argument_encoder(argument_encodings)
        role_scores = self.role_scorer(argument_encodings)

        return {
            'predicates': predicate_scores,
            'senses': sense_scores,
            'roles': role_scores,
        }


    def configure_optimizers(self):
        base_parameters = []
        base_no_weight_decay_parameters = []
        base_parameters.extend(list(self.predicate_encoder.parameters()))
        base_parameters.extend(list(self.sense_encoder.parameters()))
        base_parameters.extend(list(self.argument_encoder.parameters()))
        base_parameters.extend(list(self.predicate_timestep_encoder.parameters()))
        base_parameters.extend(list(self.argument_timestep_encoder.parameters()))
        base_parameters.extend(list(self.predicate_scorer.parameters()))
        base_parameters.extend(list(self.sense_scorer.parameters()))
        base_parameters.extend(list(self.role_scorer.parameters()))
        base_parameters.extend(list(self.sequence_encoder.parameters()))
        base_parameters.extend(list(self.argument_sequence_encoder.parameters()))

        lm_parameters = []
        lm_no_weight_decay_parameters = []
        for parameter_name, parameter in self.word_encoder.named_parameters():
            if 'word_embedding' not in parameter_name:
                base_parameters.append(parameter)
            elif self.hparams.language_model_fine_tuning:
                if 'bias' or 'LayerNorm.weight' in parameter_name:
                    lm_no_weight_decay_parameters.append(parameter)
                else:
                    lm_parameters.append(parameter)

        optimizer = torch.optim.AdamW(
            [
                {
                    'params': base_parameters
                },
                {
                    'params': base_no_weight_decay_parameters,
                    'weight_decay': 0.0,
                },
                {
                    'params': lm_parameters,
                    'weight_decay': self.hparams.language_model_weight_decay,
                    'correct_bias': False,
                    'lr': self.hparams.language_model_learning_rate,
                },
                {
                    'params': lm_no_weight_decay_parameters,
                    'weight_decay': 0.0,
                    'correct_bias': False,
                    'lr': self.hparams.language_model_learning_rate,
                },
            ],
            weight_decay=self.hparams.weight_decay,
            lr=self.hparams.learning_rate,
        )

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=1_000,
        )
        cooldown_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=20_000,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cooldown_scheduler],
            milestones=[1_000],
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


    def training_step(self, batch, batch_index):
        step_result = self._shared_step(batch)
        self.log('train/loss', step_result['loss'])
        self.log('train/loss/predicate_identification', step_result['predicate_identification_loss'])
        self.log('train/loss/sense_classification', step_result['sense_classification_loss'])
        self.log('train/loss/argument_classification', step_result['argument_classification_loss'])
        return step_result['loss']


    def validation_step(self, batch, batch_index):
        return self._shared_step(
            batch,
            compute_metrics=True,
            compute_predictions=True,
            use_sense_candidates=self.use_sense_candidates,
        )


    def test_step(self, batch, batch_index):
        return self._shared_step(
            batch,
            compute_metrics=True,
            compute_predictions=True,
            use_sense_candidates=self.use_sense_candidates,
        )


    def predict_step(self, batch, batch_index, write_to_file=True):
        step_output = self._shared_step(
            batch,
            compute_loss=False,
            compute_predictions=True,
            use_sense_candidates=self.use_sense_candidates,
        )

        predictions = step_output['predictions']

        if write_to_file:
            with open(self.predictions_path, 'a') as f:
                for sentence_id, annotations in predictions.items():
                    output = {
                        'sentence_id': sentence_id,
                        'annotations': annotations,
                    }
                    output_str = json.dumps(output, sort_keys=True)
                    f.write(output_str + '\n')
        else:
            return predictions


    def _shared_step(self, batch, compute_loss=True, compute_metrics=False, compute_predictions=False, use_preidentified_predicates=True, use_sense_candidates=False):
        step_output = {}
        sample, labels = batch
        scores = self(sample)

        if compute_loss:
            predicate_identification_loss = SrlParser._compute_classification_loss(
                scores['predicates'],
                labels['predicates'],
                2,
                ignore_index=self.hparams.padding_label_id,
            )
            sense_classification_loss = SrlParser._compute_classification_loss(
                scores['senses'],
                labels['sense_ids'],
                self.num_senses,
                ignore_index=self.hparams.padding_label_id,
            )
            argument_classification_loss = SrlParser._compute_classification_loss(
                scores['roles'],
                labels['role_ids'],
                self.num_roles,
                ignore_index=self.hparams.padding_label_id,
            )
            
            loss = predicate_identification_loss \
                + sense_classification_loss \
                + argument_classification_loss

            if torch.isnan(loss) or not torch.isfinite(loss):
                self.print('Loss:', loss)
                self.print('Predicate identification loss:', predicate_identification_loss)
                self.print('Predicate disambiguation loss:', sense_classification_loss)
                self.print('Argument classification loss:', argument_classification_loss)
                raise ValueError('NaN loss!')
            
            step_output['loss'] = loss
            step_output['predicate_identification_loss'] = predicate_identification_loss
            step_output['sense_classification_loss'] = sense_classification_loss
            step_output['argument_classification_loss'] = argument_classification_loss

        if use_sense_candidates:
            sense_candidates = sample['sense_candidates']
            sense_mask = torch.full(scores['senses'].shape, float('-inf')).to(self.device)
            sense_mask = sense_mask.scatter(1, sense_candidates, 0.0)
            scores['senses'] = scores['senses'] + sense_mask
        
        if compute_metrics:
            metrics = self.compute_step_metrics(scores, labels)
            step_output['metrics'] = metrics

        if compute_predictions:
            if use_preidentified_predicates:
                predicates = sample['predicates']
                predicate_indices = (predicates * (predicates >= 0)).nonzero().tolist()
            else:
                predicate_indices = self.get_predicate_indices(scores)
            
            step_output['predictions'] = self.get_senses_and_roles(predicate_indices, scores, sample['sentence_ids'], sample['sentence_lengths'])

        return step_output


    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'val', compute_metrics=True)


    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'test', compute_metrics=True)


    def _shared_epoch_end(self, outputs, stage, compute_metrics=False):
        if compute_metrics:
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
            predicate_identification_loss = torch.stack([x['predicate_identification_loss'] for x in outputs]).mean()
            sense_classification_loss = torch.stack([x['sense_classification_loss'] for x in outputs]).mean()
            argument_classification_loss = torch.stack([x['argument_classification_loss'] for x in outputs]).mean()
            metrics = SrlParser._compute_epoch_metrics(outputs)

            logs = {
                f'{stage}/loss': avg_loss,
                f'{stage}/loss/predicate_identification': predicate_identification_loss,
                f'{stage}/loss/sense_classification': sense_classification_loss,
                f'{stage}/loss/argument_classification': argument_classification_loss,
                f'{stage}/predicate_precision': metrics['predicates']['precision'],
                f'{stage}/predicate_recall': metrics['predicates']['recall'],
                f'{stage}/predicate_f1': metrics['predicates']['f1'],
                f'{stage}/sense_accuracy': metrics['senses']['accuracy'],
                f'{stage}/role_precision': metrics['roles']['precision'],
                f'{stage}/role_recall': metrics['roles']['recall'],
                f'{stage}/role_f1': metrics['roles']['f1'],
                f'{stage}/overall_precision': metrics['overall']['precision'],
                f'{stage}/overall_recall': metrics['overall']['recall'],
                f'{stage}/overall_f1': metrics['overall']['f1'],
            }
            self.log_dict(logs)

        predictions = {}
        for x in outputs:
            if 'predictions' not in x:
                continue
            
            for sentence_id in x['predictions']:
                if sentence_id not in predictions:
                    predictions[sentence_id] = {}
                predictions[sentence_id].update(x['predictions'][sentence_id])
        
        if predictions:
            if self.predictions_path is None:
                self.predictions_path = os.path.join(
                    self.trainer.logger.save_dir,
                    f'{stage}_predictions.json')
            
            with open(self.predictions_path, 'w') as f:
                json.dump(predictions, f, indent=2, sort_keys=True)


    def compute_step_metrics(self, scores, labels):
        predicates_g = labels['predicates']
        predicates_p = torch.argmax(scores['predicates'], dim=-1)
        predicate_tp = (predicates_p[torch.logical_and(predicates_g >= 0, predicates_p == 1)] == predicates_g[torch.logical_and(predicates_g >= 0, predicates_p == 1)]).sum()
        predicate_fp = (predicates_p[torch.logical_and(predicates_g >= 0, predicates_p == 1)] != predicates_g[torch.logical_and(predicates_g >= 0, predicates_p == 1)]).sum()
        predicate_fn = (predicates_p[predicates_g == 1] != predicates_g[predicates_g == 1]).sum()

        senses_g = labels['sense_ids']
        senses_p = torch.argmax(scores['senses'], dim=-1)
        sense_correct = (senses_p[senses_g >= 1] == senses_g[senses_g >= 1]).sum()
        sense_total = (senses_g >= 1).sum()

        roles_g = labels['role_ids']
        roles_p = torch.argmax(scores['roles'], dim=-1)
        role_tp = (roles_p[torch.logical_and(roles_g >= 0, roles_p >= 1)] == roles_g[torch.logical_and(roles_g >= 0, roles_p >= 1)]).sum()
        role_fp = (roles_p[torch.logical_and(roles_g >= 0, roles_p >= 1)] != roles_g[torch.logical_and(roles_g >= 0, roles_p >= 1)]).sum()
        role_fn = (roles_p[roles_g >= 1] != roles_g[roles_g >= 1]).sum()

        return {
            'predicate_tp': predicate_tp,
            'predicate_fp': predicate_fp,
            'predicate_fn': predicate_fn,
            'sense_correct': sense_correct,
            'sense_total': sense_total,
            'role_tp': role_tp,
            'role_fp': role_fp,
            'role_fn': role_fn,
        }


    def get_predicate_indices(self, scores):
        predicate_indices = torch.argmax(scores['predicates'], dim=-1)
        return predicate_indices


    def get_senses_and_roles(self, predicate_indices, scores, sentence_ids, sentence_lengths):
        output = {}
        sense_ids = torch.argmax(scores['senses'], dim=-1).tolist()
        role_scores = scores['roles']
        
        for (sentence_index, predicate_index), sense_id, predicate_role_scores in zip(predicate_indices, sense_ids, role_scores):
            sentence_id = sentence_ids[sentence_index]
            sentence_length = sentence_lengths[sentence_index]
            predicate_role_scores = predicate_role_scores[:sentence_length]
            
            if not self.use_viterbi_decoding:
                predicate_role_ids = torch.argmax(predicate_role_scores, dim=-1).tolist()
            else:
                predicate_role_ids, _ = viterbi_decode(
                    predicate_role_scores,
                    self.role_transition_matrix,
                    allowed_start_transitions=self.role_start_transitions,
                    device=self.device
                )

            if sentence_id not in output:
                output[sentence_id] = {}
            
            output[sentence_id][predicate_index] = {
                'predicate': self.id2sense[sense_id],
                'roles': [self.id2role[role_id] for role_id in predicate_role_ids[:sentence_length]]
            }
        
        return output


    def _load_vocabulary(self, vocabulary_path: str):
        with open(vocabulary_path, 'r') as f:
            vocabulary = json.load(f)
        
        self.sense2id = {k: v for k, v in vocabulary['senses'].items()}
        self.id2sense = {k: v for v, k in self.sense2id.items()}
        self.num_senses = len(self.sense2id)
        self.unknown_sense_id = self.sense2id['<UNK>']

        self.role2id = {k: v for k, v in vocabulary['roles'].items()}
        self.id2role = {k: v for v, k in self.role2id.items()}
        self.num_roles = len(self.role2id)
        self.unknown_role_id = self.role2id['<UNK>']


    def _build_transition_matrices(self):
        _role_transition_matrix = []
        for i in range(len(self.id2role)):
            previous_label = self.id2role[i]
            role_transitions = []
            for j in range(len(self.id2role)):
                label = self.id2role[j]
                if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                    role_transitions.append(float('-inf'))
                else:
                    role_transitions.append(0.0)
            _role_transition_matrix.append(role_transitions)
        
        _role_start_transitions = []
        for i in range(len(self.id2role)):
            label = self.id2role[i]
            if label[0] == 'I':
                _role_start_transitions.append(float('-inf'))
            else:
                _role_start_transitions.append(0.0)
        
        self.role_transition_matrix = nn.Parameter(torch.as_tensor(_role_transition_matrix), requires_grad=False)
        self.role_start_transitions = nn.Parameter(torch.as_tensor(_role_start_transitions), requires_grad=False)


    @staticmethod
    def _compute_classification_loss(scores, labels, num_classes, ignore_index=-1):
        classification_loss = F.cross_entropy(
            scores.view(-1, num_classes),
            labels.view(-1),
            reduction='sum',
            ignore_index=ignore_index)

        return classification_loss


    @staticmethod
    def _compute_epoch_metrics(outputs):
        predicate_tp = torch.stack([o['metrics']['predicate_tp'] for o in outputs]).sum()
        predicate_fp = torch.stack([o['metrics']['predicate_fp'] for o in outputs]).sum()
        predicate_fn = torch.stack([o['metrics']['predicate_fn'] for o in outputs]).sum()

        predicate_precision = torch.true_divide(predicate_tp, (predicate_tp + predicate_fp)) if predicate_tp + predicate_fp > 0 else torch.as_tensor(0)
        predicate_recall = torch.true_divide(predicate_tp, (predicate_tp + predicate_fn)) if predicate_tp + predicate_fn > 0 else torch.as_tensor(0)
        predicate_f1 = 2 * torch.true_divide(predicate_precision * predicate_recall, predicate_precision + predicate_recall) if predicate_precision + predicate_recall > 0 else torch.as_tensor(0.)

        sense_correct = torch.stack([o['metrics']['sense_correct'] for o in outputs]).sum()
        sense_total = torch.stack([o['metrics']['sense_total'] for o in outputs]).sum()
        sense_accuracy = torch.true_divide(sense_correct, sense_total) if sense_total > 0 else torch.as_tensor(0.)

        role_tp = torch.stack([o['metrics']['role_tp'] for o in outputs]).sum()
        role_fp = torch.stack([o['metrics']['role_fp'] for o in outputs]).sum()
        role_fn = torch.stack([o['metrics']['role_fn'] for o in outputs]).sum()

        role_precision = torch.true_divide(role_tp, (role_tp + role_fp)) if role_tp + role_fp > 0 else torch.as_tensor(0)
        role_recall = torch.true_divide(role_tp, (role_tp + role_fn)) if role_tp + role_fn > 0 else torch.as_tensor(0)
        role_f1 = 2 * torch.true_divide(role_precision * role_recall, role_precision + role_recall) if role_precision + role_recall > 0 else torch.as_tensor(0.)

        overall_tp = role_tp + sense_correct
        overall_fp = role_fp + (sense_total - sense_correct)
        overall_fn = role_fn + (sense_total - sense_correct)
        overall_precision = torch.true_divide(overall_tp, (overall_tp + overall_fp)) if overall_tp + overall_fp > 0 else torch.as_tensor(0.)
        overall_recall = torch.true_divide(overall_tp, (overall_tp + overall_fn)) if overall_tp + overall_fn > 0 else torch.as_tensor(0.)
        overall_f1 = 2 * torch.true_divide(overall_precision * overall_recall, overall_precision + overall_recall) if overall_precision + overall_recall > 0 else torch.as_tensor(0.)

        return {
            'predicates': {
                '_tp': predicate_tp,
                '_fp': predicate_fp,
                '_fn': predicate_fn,
                'precision': predicate_precision,
                'recall': predicate_recall,
                'f1': predicate_f1,
            },
            'senses': {
                '_correct': sense_correct,
                '_total': sense_total,
                'accuracy': sense_accuracy,
            },
            'roles': {
                '_tp': role_tp,
                '_fp': role_fp,
                '_fn': role_fn,
                'precision': role_precision,
                'recall': role_recall,
                'f1': role_f1,
            },
            'overall': {
                '_tp': overall_tp,
                '_fp': overall_fp,
                '_fn': overall_fn,
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
            },
        }

