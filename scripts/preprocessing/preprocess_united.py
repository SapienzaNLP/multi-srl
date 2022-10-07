import argparse
import json
import logging
import os


def parse(path: str):
    data = {}
    sense_vocabulary = {'_', '<UNK>'}
    role_vocabulary = {'_', '<UNK>'}

    with open(path) as f:
        word_index = 0
        sentence_words = []
        sentence_lemmas = []
        sentence_predicates = []
        sentence_predicate_indices = []
        sentence_roles = []

        for line in f:
            line = line.strip()
            if not line:
                sentence_roles = list(zip(*sentence_roles))
                
                annotations = {}
                for predicate_index, roles in zip(sentence_predicate_indices, sentence_roles):
                    predicate_sense = sentence_predicates[predicate_index]
                    sense_vocabulary.add(predicate_sense)
                    role_vocabulary.update(roles)

                    annotations[predicate_index] = {
                        'predicate': predicate_sense,
                        'roles': roles,
                    }

                sentence_data = {
                    'words': sentence_words,
                    'annotations': annotations,
                    'lemmas': sentence_lemmas,
                }

                data[len(data)] = sentence_data

                word_index = 0
                sentence_words = []
                sentence_lemmas = []
                sentence_predicates = []
                sentence_predicate_indices = []
                sentence_roles = []
                continue
            
            if line[0] == '#':
                continue

            parts = line.split('\t')

            word = parts[1].strip()
            sentence_words.append(word)

            lemma = parts[2].strip()
            sentence_lemmas.append(lemma)

            predicate = parts[3].strip()
            if predicate != '_':
                sentence_predicate_indices.append(word_index)
                sentence_predicates.append(predicate)
            else:
                sentence_predicates.append(predicate)

            roles = parts[5:]
            sentence_roles.append(roles)
            word_index += 1

    logging.info('Found {} senses'.format(len(sense_vocabulary)))
    logging.info('Found {} roles'.format(len(role_vocabulary)))
    return data


def write_parsed_data(data, path):
    output = json.dumps(data, indent=2, sort_keys=True)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'w') as f:
        f.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        dest='input_path',
        help='Path to the UniteD file to preprocess.')
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        dest='output_path',
        help='Path to the output file.')
    parser.add_argument(
        '--log',
        type=str,
        default='WARNING',
        dest='loglevel',
        help='Log level. Default = WARNING.')

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))
    logging.info('Parsing {}...'.format(args.input_path))

    parsed_data = parse(args.input_path)
    write_parsed_data(parsed_data, args.output_path)

    logging.info('Done!')