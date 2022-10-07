import argparse
import json
import logging

def compute_vocabulary(input_path: str, null_label: str = '_', unknown_label: str = '<UNK>'):
    """
    Compute the vocabulary from data.
    """
    # Load the data.
    with open(input_path) as f:
        data = json.load(f)

    # Compute the vocabulary.
    sense_vocabulary = {
        null_label: 0,
        unknown_label: 1,
    }
    role_vocabulary = {
        null_label: 0,
        unknown_label: 1,
    }
    candidates_vocabulary = {}

    for sentence_id, sentence in data.items():
        lemmas = sentence['lemmas']
        
        for predicate_index, annotation in sentence['annotations'].items():
            sense = annotation['predicate']
            if sense not in sense_vocabulary:
                sense_vocabulary[sense] = len(sense_vocabulary)
            
            predicate_lemma = lemmas[int(predicate_index)].lower()
            if predicate_lemma not in candidates_vocabulary:
                candidates_vocabulary[predicate_lemma] = [sense]
            elif sense not in candidates_vocabulary[predicate_lemma]:
                candidates_vocabulary[predicate_lemma].append(sense)
                candidates_vocabulary[predicate_lemma].sort()

            for role in annotation['roles']:
                if role not in role_vocabulary:
                    role_vocabulary[role] = len(role_vocabulary)
    
    return {
        'senses': sense_vocabulary,
        'roles': role_vocabulary,
        'candidates': candidates_vocabulary,
    }


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
        dest='input_path',
        help='Path to the preprocessed data file to use for building the vocabularies.')
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
    logging.info('Parsing {}...'.format(args.input_path))

    vocabulary = compute_vocabulary(args.input_path)
    write_vocabulary(args.output_path, vocabulary)

    logging.info('Done!')
