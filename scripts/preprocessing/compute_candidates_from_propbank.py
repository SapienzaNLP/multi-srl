import argparse
import json
import logging
import os
import xml.etree.ElementTree as ET

def compute_vocabulary(input_dir: str, vocab_path: str, null_label: str = '_', unknown_label: str = '<UNK>'):
    """
    Compute the vocabulary from the PropBank frames.
    """

    # Compute the vocabulary.
    sense_vocabulary = {
        null_label: 0,
        unknown_label: 1,
    }
    candidates_vocabulary = {}

    # Load exisiting vocab.
    with open(vocab_path) as f:
        vocab = json.load(f)

    # Iterate over the xml files in input_dir.
    for filename in os.listdir(input_dir):
        if not filename.endswith('.xml'):
            continue

        # Load the root node from the xml file.
        tree = ET.parse(os.path.join(input_dir, filename))
        root = tree.getroot()

        # Iterate over the predicates.
        for predicate in root.iter('predicate'):

            # Iterate over the rolesets.
            for roleset in predicate.iter('roleset'):
                sense = roleset.attrib['id']
                if sense not in sense_vocabulary:
                    sense_vocabulary[sense] = len(sense_vocabulary)

                for alias in roleset.iter('alias'):
                    predicate_lemma = alias.text.strip().lower()
                    predicate_lemma = predicate_lemma.replace('_', ' ')
                    if predicate_lemma not in candidates_vocabulary:
                        candidates_vocabulary[predicate_lemma] = [sense]
                    elif sense not in candidates_vocabulary[predicate_lemma]:
                        candidates_vocabulary[predicate_lemma].append(sense)
                        candidates_vocabulary[predicate_lemma].sort()

    print('Number of senses: {} -> {}'.format(len(vocab['senses']), len(sense_vocabulary)))
    print('Number of candidates: {} -> {}'.format(len(vocab['candidates']), len(candidates_vocabulary)))
    vocab['senses'] = sense_vocabulary
    vocab['candidates'] = candidates_vocabulary

    return vocab


def write_vocabulary(output_path: str, vocabulary: dict):
    """
    Write the vocabulary to a file.
    """
    with open(output_path, 'w') as f:
        json.dump(vocabulary, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        dest='input_dir',
        help='Path to the directory that contains the PropBank frames.')
    parser.add_argument(
        '--existing_vocabulary',
        type=str,
        required=True,
        dest='vocab_path',
        help='Path to the existing vocabulary to expand.')
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        dest='output_path',
        help='Path to the output file for the vocabulary.')
    parser.add_argument(
        '--log',
        type=str,
        default='WARNING',
        dest='loglevel',
        help='Log level. Default = WARNING.')

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))
    logging.info('Parsing {}...'.format(args.input_dir))

    vocabulary = compute_vocabulary(args.input_dir, args.vocab_path)
    write_vocabulary(args.output_path, vocabulary)

    logging.info('Done!')
