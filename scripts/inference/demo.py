import argparse
import logging

import stanza
import transformers
import gradio as gr

from srl.data.srl_data_module import SrlDataModule
from srl.data.srl_dataset import SrlDataset
from srl.model.srl_parser import SrlParser


def load_model(checkpoint_path:str) -> SrlParser:
    logging.info('Loading model from checkpoint {}', checkpoint_path)
    transformers.logging.set_verbosity_error()
    model = SrlParser.load_from_checkpoint(checkpoint_path)
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

def group_spans(sentence:list, spans:dict, predicate_index:int, predicate_sense:str) -> dict:
    # Group spans by role.
    # Example:
    #   sentence = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.'],
    #   spans = {(0, 4): 'ARG0', (5, 9): 'ARGM-DIR'}
    # Returns:
    #   [('The quick brown fox', 'ARG0'), ('jumps', None), ('over the lazy dog', 'ARGM-DIR'), ('.', None)]] 
    
    groups = []
    flags = [False] * len(sentence)
    for (span_start, span_end), role in spans.items():
        flags[span_start:span_end] = [True] * (span_end - span_start)
        group = (span_start, span_end, sentence[span_start:span_end], role)
        groups.append(group)
    
    for i, word in enumerate(sentence):
        if not flags[i]:
            if i == predicate_index:
                group = (i, i + 1, [word], predicate_sense)
            else:
                group = (i, i + 1, [word], None)
            groups.append(group)
    
    groups.sort(key=lambda x: x[0])
    groups = [(f'{" ".join(group[2])}', group[3]) for group in groups]
    return groups

def srl(nlp: stanza.Pipeline, model: SrlParser, datamodule: SrlDataModule) -> list:
    # Run SRL on a sentence.
    # Example: 'The quick brown fox jumps over the lazy dog.'
    # Returns: [('The quick brown fox', 'ARG0'), ('jumps', None), ('over the lazy dog', 'ARGM-DIR'), ('.', None)]

    def _srl(text:str, index:int) -> list:
        predictions = {}

        doc = nlp(text)
        datamodule.pred_data = SrlDataset.load_sentences(doc.sentences)

        for batch_idx, batch in enumerate(datamodule.predict_dataloader()):
            batch_predictions = model.predict_step(batch, batch_idx, write_to_file=False)
            predictions.update(batch_predictions)
        
        output = []
        sorted_sentence_ids = sorted(predictions.keys())
        for sentence_id in sorted_sentence_ids:
            sentence = predictions[sentence_id]
            words = [w.text for w in doc.sentences[sentence_id].words]
            sorted_predicate_indices = sorted(sentence.keys())

            for predicate_index in sorted_predicate_indices:
                if predicate_index == index:
                    annotations = sentence[predicate_index]
                    annotations['roles'] = convert_bio_to_spans(annotations['roles'])
                    output = group_spans(words, annotations['roles'], predicate_index, annotations['predicate'])
        
        return output
    
    return _srl



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
        '--log',
        type=str,
        default='WARNING',
        dest='loglevel',
        help='Log level. Default = WARNING.')

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))
    logging.info('Loading modules...')

    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', logging_level='ERROR', tokenize_no_ssplit=True)
    model = load_model(args.checkpoint_path)
    datamodule = load_datamodule(model, args.vocabulary_path)

    with gr.Blocks() as demo:

        with gr.Row():
            text = gr.Textbox(placeholder="Enter sentence here...")

        with gr.Row():
            with gr.Column(scale=1):
                srl_index = gr.Number(0, precision=0, label="Predicate index")
            with gr.Column(scale=9):
                srl_output = gr.HighlightedText(label='SRL output')
                srl_examples = gr.Examples(
                    examples=[
                        ["The quick brown fox jumps over the lazy dog.", 4],
                        ["It was the best of times, it was the worst of times.", 1],
                        ["It was the best of times, it was the worst of times.", 8],
                    ],
                    inputs=[text, srl_index],
                    label='SRL examples',
                )
        
        srl_button = gr.Button("Run SRL")
        srl_button.click(fn=srl(nlp, model, datamodule), inputs=[text, srl_index], outputs=[srl_output])

        demo.launch()
    
    logging.info('Done!')
