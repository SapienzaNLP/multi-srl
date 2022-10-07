import argparse
import json
import logging
import os


def parse(
    path,
    keep_pos_tags=False,
    keep_lemmas=False,
    add_predicate_pos=False,
    only_verbs=False
):
    verb_pos_set = {'VBD', 'VBZ', 'VBG', 'VBP', 'VBN', 'MD', 'VB', 'PRF'}
    data = {}
    sense_vocabulary = {'_', '<UNK>'}
    role_vocabulary = {'_', '<UNK>'}

    with open(path) as f:
        word_index = 0
        sentence_words = []
        sentence_lemmas = []
        sentence_pos_tags = []
        sentence_predicates = []
        sentence_predicate_indices = []
        sentence_roles = []

        for line in f:
            line = line.strip()
            if line and line[0] == '#':
                continue

            if not line:
                sentence_roles = list(map(list, zip(*sentence_roles)))
                sentence_roles = sentence_roles[:-1]
                
                annotations = {}
                for predicate_index, predicate_roles in zip(sentence_predicate_indices, sentence_roles):
                    if not only_verbs or sentence_pos_tags[predicate_index] in verb_pos_set:
                        bio_roles = []
                        current_role = '_'
                        inside_argument = False
                        for role in predicate_roles:
                            assert '*' in role, 'Error: found arg ({}) with no *.'.format(role)
                            assert role.count('(') < 2, '{}'.format(' '.join(sentence_words))
                            assert role.count(')') < 2, '{}'.format(' '.join(sentence_words))
                            if '(' in role and ')' in role:
                                current_role = role[1:-2]
                                inside_argument = False
                                bio_roles.append('B-{}'.format(current_role))
                                continue
                            if '(' in role:
                                current_role = role[1:-1]
                                inside_argument = True
                                bio_roles.append('B-{}'.format(current_role))
                                continue
                            if ')' in role:
                                inside_argument = False
                                bio_roles.append('I-{}'.format(current_role))
                                continue
                            if inside_argument:
                                bio_roles.append('I-{}'.format(current_role))
                            else:
                                bio_roles.append('_')
                        
                        annotations[predicate_index] = {
                            'predicate': sentence_predicates[predicate_index],
                            'roles': bio_roles,
                        }

                sentence_data = {
                    'words': sentence_words,
                    'annotations': annotations,
                }

                if keep_pos_tags:
                    sentence_data['pos_tags'] = sentence_pos_tags
                if keep_lemmas:
                    sentence_data['lemmas'] = sentence_lemmas
                
                data[len(data)] = sentence_data

                word_index = 0
                sentence_words = []
                sentence_lemmas = []
                sentence_pos_tags = []
                sentence_predicates = []
                sentence_predicate_indices = []
                sentence_roles = []
                continue

            parts = line.split()

            word = parts[3].strip()
            sentence_words.append(word)

            lemma = parts[6].strip()
            sentence_lemmas.append(lemma)

            pos_tag = parts[4].strip()
            sentence_pos_tags.append(pos_tag)

            predicate = parts[6].strip()
            predicate_sense = parts[7].strip()
            if predicate_sense != '-':
                sentence_predicate_indices.append(word_index)
                if add_predicate_pos and pos_tag.upper() in verb_pos_set:
                    sentence_predicates.append('{}-v.{}'.format(predicate, predicate_sense))
                else:
                    sentence_predicates.append('{}.{}'.format(predicate, predicate_sense))
            else:
                sentence_predicates.append('_')

            roles = parts[11:]
            sentence_roles.append(roles)
            word_index += 1

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
        help='Path to the data to preprocess.')
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        dest='output_path',
        help='Path to the output file.')
    parser.add_argument(
        '--keep_pos_tags',
        action='store_true',
        help='Keep the POS tags when parsing the dataset.')
    parser.add_argument(
        '--keep_lemmas',
        action='store_true',
        help='Keep the lemmas when parsing the dataset.')
    parser.add_argument(
        '--add_predicate_pos',
        action='store_true',
        help='Add a pos label to the predicate sense label (may be useful in English).')
    parser.add_argument(
        '--only_verbs',
        action='store_true',
        help='Only keep verbs in the dataset.')
    parser.add_argument(
        '--log',
        type=str,
        default='WARNING',
        dest='loglevel',
        help='Log level. Default = WARNING.')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    logging.info('Parsing {}...'.format(args.input_path))

    parsed_data = parse(
        args.input_path,
        keep_pos_tags=args.keep_pos_tags,
        keep_lemmas=args.keep_lemmas,
        add_predicate_pos=args.add_predicate_pos,
        only_verbs=args.only_verbs)
    
    write_parsed_data(parsed_data, args.output_path)

    logging.info('Done!')
