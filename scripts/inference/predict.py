import argparse
import json
import logging

import stanza
import transformers

from srl.data.srl_data_module import SrlDataModule
from srl.data.srl_dataset import SrlDataset
from srl.model.srl_parser import SrlParser


def load_model(checkpoint_path:str, vocabulary_path:str) -> SrlParser:
    logging.info('Loading model from checkpoint {}', checkpoint_path)
    transformers.logging.set_verbosity_error()
    model = SrlParser.load_from_checkpoint(checkpoint_path, vocabulary_path=vocabulary_path)
    transformers.logging.set_verbosity_warning()
    model.eval()

    return model

def load_datamodule(model: SrlParser, vocabulary_path:str) -> SrlDataModule:
    logging.info('Loading data module from model')
    datamodule = SrlDataModule(
        vocabulary_path=vocabulary_path,
        language_model_name=model.hparams.language_model_name,
    )
    return datamodule

def convert_bio_to_spans(bio_tags:list) -> list:
    # Convert BIO tags to a dictionary of spans with key (start, end) and value (role).
    # Example: ['_', 'B-ARG0', 'I-ARG0', '_', 'B-ARG1', 'I-ARG1', '_'] -> {(1, 3): 'ARG0', (4, 6): 'ARG1'}
    spans = {}
    start = None
    end = None
    role = None

    for i, tag in enumerate(bio_tags):
        if tag == '_':
            # End of span.
            if start is not None:
                assert end is not None, f'End of span is None but start of span is {start}.'
                assert role is not None, f'Role of span is None but start of span is {start}.'
                if role != 'V':
                    spans[(start, end)] = role
                start = None
                end = None
                role = None
        else:
            # Start of span.
            if tag.startswith('B-'):
                if start is not None and role != 'V':
                    assert end is not None, f'End of span is None but start of span is {start}.'
                    assert role is not None, f'Role of span is None but start of span is {start}.'
                    spans[(start, end)] = role
                start = i
                end = i + 1
                role = tag[2:]
            # Continue span.
            elif tag.startswith('I-'):
                assert start is not None, f'Start of span is None but end of span is {end}.'
                assert end is not None, f'End of span is None but start of span is {start}.'
                assert role is not None, f'Role of span is None but start of span is {start}.'
                assert role == tag[2:], f'Role of span is {role} but tag is {tag}.'
                end += 1
            else:
                raise Exception(f'Invalid tag {tag}')
    
    # End of sentence.
    if start is not None:
        assert end is not None
        assert role is not None
        spans[(start, end)] = role
    
    return spans


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        dest='checkpoint_path',
        help='Path to the model checkpoint.')
    parser.add_argument(
        '--vocabulary',
        type=str,
        required=True,
        dest='vocabulary_path')
    parser.add_argument(
        '--input_file',
        type=str,
        required=False,
        dest='input_file_path',
        help='Path to the input file where to read the sentences to parse.')
    parser.add_argument(
        '--text',
        type=str,
        required=False,
        dest='text',
        help='Text to parse.')
    parser.add_argument(
        '--output_file',
        type=str,
        required=False,
        dest='output_file_path',
        help='Path to the output file where to write the parsed sentences.')
    parser.add_argument(
        '--language',
        type=str,
        required=False,
        default='en',
        dest='language',
        help='Language of the input text.')
    parser.add_argument(
        '--log',
        type=str,
        default='WARNING',
        dest='loglevel',
        help='Log level. Default = WARNING.')

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    logging.info('Loading modules...')
    nlp = stanza.Pipeline(lang=args.language, processors='tokenize,pos,lemma', logging_level='ERROR')
    model = load_model(args.checkpoint_path, args.vocabulary_path)

    if args.input_file_path:
        logging.info('Reading sentences from {}', args.input_file_path)
        with open(args.input_file_path, 'r') as input_file:
            text = input_file.read()
    elif args.text:
        text = args.text
    else:
        raise ValueError('Either --input_file or --text must be specified.')
    
    doc = nlp(text)
    datamodule = load_datamodule(model, args.vocabulary_path)
    datamodule.pred_data = SrlDataset.load_sentences(doc.sentences)

    predictions = {}
    for batch_idx, batch in enumerate(datamodule.predict_dataloader()):
        batch_predictions = model.predict_step(batch, batch_idx, write_to_file=False)
        predictions.update(batch_predictions)
    
    for sentence_id, sentence in predictions.items():
        for predicate_index, annotations in sentence.items():
            annotations['roles'] = convert_bio_to_spans(annotations['roles'])
    
    if args.output_file_path:
        logging.info('Writing predictions to {}', args.output_file_path)
        with open(args.output_file_path, 'w') as output_file:
            for sentence_id, sentence_predictions in predictions.items():
                words = [w.text for w in doc.sentences[sentence_id].words]
                output_object = {
                    'sentence_id': sentence_id,
                    'words': words,
                    'predictions': sentence_predictions,
                }
                output_file.write(json.dumps(output_object) + '\n')
    else:
        for sentence_id, sentence_predictions in predictions.items():
            words = [w.text for w in doc.sentences[sentence_id].words]
            output_object = {
                'sentence_id': sentence_id,
                'words': words,
                'predictions': sentence_predictions,
            }
            print(output_object)
    
    logging.info('Done!')
