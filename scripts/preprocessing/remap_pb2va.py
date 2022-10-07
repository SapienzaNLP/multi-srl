import argparse
import json
import logging
import os

DEFAULT_ARGS_MAP = {
    "AM-EXT": "Extent",
    "AM-MNR": "Attribute",
    "AM-LOC": "Location",
    "AM-DIR": "Destination",
    "AM-PNC": "Purpose",
    "AM-MOD": "Modal",
    "AM-TMP": "Time",
    "AM-PRT": "Predicative",
    "AM-NEG": "Negation",
    "AM-ADV": "Adverbial",
    "AM-CAU": "Cause",
    "AM-DIS": "Connective",
    "AM-PRD": "Predicative",
    "AM-PRP": "Purpose",
    "AM-GOL": "Goal",
    "AM": "Modifier",
}


def read_mapping(pb2va_path:str, frame_info_path:str) -> dict:
    frame_names = {}
    predicate_mapping = {}
    role_mapping = {}

    with open(frame_info_path, 'r') as f:
        for line_no, line in enumerate(f):
            if line_no == 0:
                continue

            line = line.strip()
            if not line:
                continue

            frame_id, frame_name, *_ = line.split()
            frame_names[frame_id] = frame_name

    with open(pb2va_path, 'r') as f:
        for line_no, line in enumerate(f):
            if line_no == 0:
                continue
            
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            pb_sense, va_frame_id = parts[0].split('>')
            assert pb_sense not in predicate_mapping, f'Duplicate mapping for {pb_sense}'
            va_frame_name = frame_names[va_frame_id]
            predicate_mapping[pb_sense] = va_frame_name
            role_mapping[pb_sense] = {}

            for part in parts[1:]:
                pb_role, va_role = part.split('>')
                assert pb_role not in role_mapping[pb_sense]
                role_mapping[pb_sense][pb_role] = va_role

    return predicate_mapping, role_mapping


def remap_pb2va(input_path:str, output_path:str, predicate_mapping:dict, role_mapping:dict, is_propbank3:bool=False):

    with open(input_path, 'r') as f:
        data = json.load(f)

    remapped_data = {}
    skipped_predicates = []
    skipped_roles = []

    for sentence_id, sentence in data.items():
        remapped_data[sentence_id] = {
            'words': sentence['words'],
            'lemmas': sentence['lemmas'],
            'annotations': {}
        }

        annotations = sentence['annotations']
        predicate_indices = [int(i) for i in annotations.keys()]
        predicate_indices = sorted(predicate_indices)

        for predicate_index in predicate_indices:
            predicate_annotations = annotations[str(predicate_index)]
            pb_sense = predicate_annotations['predicate']
            if not is_propbank3 and '-v' not in pb_sense:
                continue

            pb_sense = pb_sense.replace('-v', '')
            if pb_sense not in predicate_mapping:
                skipped_predicates.append(pb_sense)
                continue

            va_frame = predicate_mapping[pb_sense]
            remapped_data[sentence_id]['annotations'][predicate_index] = {}
            remapped_data[sentence_id]['annotations'][predicate_index]['predicate'] = va_frame

            remapped_roles = []
            for pb_role in predicate_annotations['roles']:
                if pb_role.startswith('B-') or pb_role.startswith('I-'):
                    role_prefix, pb_role = pb_role[:2], pb_role[2:]
                else:
                    role_prefix = ''
                
                if pb_role.startswith('R-') or pb_role.startswith('C-'):
                    role_type, pb_role = pb_role[:2], pb_role[2:]
                else:
                    role_type = ''
                
                pb_role = pb_role.replace('ARG', 'A')

                if pb_role == '_':
                    role = '_'
                elif pb_role == 'V':
                    role = 'V'
                elif pb_role in role_mapping[pb_sense]:
                    role = role_mapping[pb_sense][pb_role]
                elif pb_role in DEFAULT_ARGS_MAP:
                    role = DEFAULT_ARGS_MAP[pb_role]
                else:
                    skipped_roles.append(f'{pb_sense}/{pb_role}')
                    role = '_'
                
                if role != '_':
                    role = f'{role_prefix}{role_type}{role}'
                remapped_roles.append(role)
            
            remapped_data[sentence_id]['annotations'][predicate_index]['roles'] = remapped_roles
    
    print(f'Skipped predicates: {len(skipped_predicates)}')
    print(f'Skipped roles: {len(skipped_roles)}')
    print()

    with open(output_path, 'w') as f:
        json.dump(remapped_data, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pb2va',
        type=str,
        default='data/resources/verbatlas-1.1/pb2va.tsv',
        dest='pb2va_path',
        help='Path to the pb2va file.')
    parser.add_argument(
        '--frame_info',
        type=str,
        default='data/resources/verbatlas-1.1/VA_frame_info.tsv',
        dest='frame_info_path',
        help='Path to the frame_info file.')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        dest='input_path',
        help='Path to the input file.')
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        dest='output_path',
        help='Path to the output file.')
    parser.add_argument(
        '--propbank-3',
        action='store_true'
    )
    parser.add_argument(
        '--log',
        type=str,
        default='WARNING',
        dest='loglevel',
        help='Log level. Default = WARNING.')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))
    logging.info('Parsing {}...'.format(args.input_path))

    predicate_mapping, role_mapping = read_mapping(args.pb2va_path, args.frame_info_path)
    remap_pb2va(args.input_path, args.output_path, predicate_mapping, role_mapping, is_propbank3=args.propbank_3)

    logging.info('Done!')
