import argparse
import logging
from typing import Dict


def read_file(path:str):
    data = {}
    token_ids = []
    tokens = []
    lemmas = []
    frames = {}
    synsets = {}
    roles = []

    with open(path, 'r') as f:

        for line_no, line in enumerate(f):
            line = line.strip()
            
            # End of sentence.
            if not line:
                assert len(token_ids) == len(tokens) == len(lemmas) == len(roles), \
                    f'Line {line_no}: Lengths of token_ids, tokens, lemmas and roles do not match.'
                assert len(frames) == len(synsets), \
                    f'Line {line_no}: Lengths of frames and synsets do not match.'
                assert united_srl_id not in data, f'Line {line_no}: Duplicate UniteD-SRL ID {united_srl_id}.'

                # Transpose roles.
                roles = list(map(list, zip(*roles)))
                assert len(roles) == len(frames), \
                    f'Line {line_no}: Lengths of roles and frames do not match.'
                
                roles_dict = {}
                for token_id, frame_roles in zip(frames.keys(), roles):
                    roles_dict[token_id] = frame_roles

                # Save the sentence.
                data[united_srl_id] = {
                    'token_ids': token_ids,
                    'tokens': tokens,
                    'lemmas': lemmas,
                    'frames': frames,
                    'synsets': synsets,
                    'roles': roles_dict,
                }

                # Reset the sentence.
                token_ids = []
                tokens = []
                lemmas = []
                frames = {}
                synsets = {}
                roles = []
                continue

            if line.startswith('#'):
                # Parse metadata (which starts with #).
                try:
                    key, value = line.split(' = ')
                except:
                    logging.error(f'Error in line {line_no}: {line}')
                    logging.error(f'Line {line_no} is not a valid key-value pair (separated by " = ")')
                    exit(1)
                
                key = key.replace('# ', '').strip()
                assert key in {'united_srl_id', 'document_id', 'sentence_id', 'domain', 'text'}, f'Error in line {line_no}: {line}\nKey {key} is not valid.'

                if key == 'united_srl_id':
                    united_srl_id = int(value)
                elif key == 'document_id':
                    document_id = value
                elif key == 'sentence_id':
                    sentence_id = value
                elif key == 'domain':
                    domain = value
                elif key == 'text':
                    text = value
                
            else:
                # Parse token.
                try:
                    token_id, token, lemma, frame, synset, *line_roles = line.split('\t')
                except:
                    logging.error(f'Error in line {line_no}: {line}')
                    logging.error(f'Line {line_no} is not a valid token line (separated by "\\t")')
                    exit(1)
                
                token_id = int(token_id)
                assert token_id not in token_ids, f'Error in line {line_no}: {line}\nToken ID {token_id} is not unique.'
                assert not token_ids or token_id == token_ids[-1] + 1, f'Error in line {line_no}: {line}\nToken ID {token_id} is not consecutive.'
                
                token_ids.append(token_id)
                tokens.append(token)
                lemmas.append(lemma)
                if frame != '_':
                    assert synset != '_', f'Error in line {line_no}: {line}\nFrame {frame} is not empty but synset {synset} is empty.'
                    frames[token_id] = frame
                    synsets[token_id] = synset
                roles.append(line_roles)

    return data


def evaluate(gold_data:dict, pred_data:dict) -> dict:
    # Compute the F1 score for each frame.
    frame_f1 = {}
    
    for united_srl_id, gold_sentence in gold_data.items():
        if united_srl_id in pred_data:
            pred_sentence = pred_data[united_srl_id]
        else:
            logging.warning(f'UniteD-SRL ID {united_srl_id} is not in the prediction data.')
            pred_sentence = {'frames': {}}
        
        for token_id, gold_frame in gold_sentence['frames'].items():
            if token_id in pred_sentence['frames']:
                pred_frame = pred_sentence['frames'][token_id]
            else:
                logging.warning(f'UniteD-SRL ID {united_srl_id}, token ID {token_id} is not in the prediction data.')
                pred_frame = '_'
            
            if gold_frame not in frame_f1:
                frame_f1[gold_frame] = {'tp': 0, 'fp': 0, 'fn': 0}
            if pred_frame != '_' and pred_frame not in frame_f1:
                frame_f1[pred_frame] = {'tp': 0, 'fp': 0, 'fn': 0}
            
            if pred_frame == gold_frame:
                frame_f1[gold_frame]['tp'] += 1
            else:
                frame_f1[gold_frame]['fn'] += 1
                if pred_frame != '_':
                    frame_f1[pred_frame]['fp'] += 1
        
        for token_id, pred_frame in pred_sentence['frames'].items():
            if token_id not in gold_sentence['frames']:
                if pred_frame == '_':
                    continue
                if pred_frame not in frame_f1:
                    frame_f1[pred_frame] = {'tp': 0, 'fp': 0, 'fn': 0}
                frame_f1[pred_frame]['fp'] += 1
    
    # Compute the F1 score for each synset.
    synset_f1 = {}
    for united_srl_id, gold_sentence in gold_data.items():
        if united_srl_id in pred_data:
            pred_sentence = pred_data[united_srl_id]
        else:
            logging.warning(f'UniteD-SRL ID {united_srl_id} is not in the prediction data.')
            pred_sentence = {'synsets': {}}
        
        for token_id, gold_synset in gold_sentence['synsets'].items():
            if token_id in pred_sentence['synsets']:
                pred_synset = pred_sentence['synsets'][token_id]
            else:
                logging.warning(f'UniteD-SRL ID {united_srl_id}, token ID {token_id} is not in the prediction data.')
                pred_synset = '_'
            
            if gold_synset not in synset_f1:
                synset_f1[gold_synset] = {'tp': 0, 'fp': 0, 'fn': 0}
            if pred_synset != '_' and pred_synset not in synset_f1:
                synset_f1[pred_synset] = {'tp': 0, 'fp': 0, 'fn': 0}
            
            if pred_synset == gold_synset:
                synset_f1[gold_synset]['tp'] += 1
            else:
                synset_f1[gold_synset]['fn'] += 1
                if pred_synset != '_':
                    synset_f1[pred_synset]['fp'] += 1
        
        for token_id, pred_synset in pred_sentence['synsets'].items():
            if token_id not in gold_sentence['synsets']:
                if pred_synset == '_':
                    continue
                if pred_synset not in synset_f1:
                    synset_f1[pred_synset] = {'tp': 0, 'fp': 0, 'fn': 0}
                synset_f1[pred_synset]['fp'] += 1
    
    # Compute the F1 score for each role.
    role_f1 = {}

    for united_srl_id, gold_sentence in gold_data.items():
        if united_srl_id in pred_data:
            pred_sentence = pred_data[united_srl_id]
        else:
            logging.warning(f'UniteD-SRL ID {united_srl_id} is not in the prediction data.')
            pred_sentence = {'roles': {}}
        
        for token_id, gold_roles in gold_sentence['roles'].items():
            if token_id in pred_sentence['roles']:
                pred_roles = pred_sentence['roles'][token_id]
            else:
                logging.warning(f'UniteD-SRL ID {united_srl_id}, token ID {token_id} is not in the prediction data.')
                pred_roles = ['_' for _ in gold_roles]
            
            for gold_role, pred_role in zip(gold_roles, pred_roles):
                if gold_role == pred_role:
                    if gold_role == '_' or gold_role == 'V':
                        continue
                    if gold_role not in role_f1:
                        role_f1[gold_role] = {'tp': 0, 'fp': 0, 'fn': 0}
                    role_f1[gold_role]['tp'] += 1
                else:
                    if gold_role != '_' and gold_role != 'V':
                        if gold_role not in role_f1:
                            role_f1[gold_role] = {'tp': 0, 'fp': 0, 'fn': 0}
                        role_f1[gold_role]['fn'] += 1
                    if pred_role != '_' and pred_role != 'V':
                        if pred_role not in role_f1:
                            role_f1[pred_role] = {'tp': 0, 'fp': 0, 'fn': 0}
                        role_f1[pred_role]['fp'] += 1
        
        for token_id, pred_roles in pred_sentence['roles'].items():
            if token_id not in gold_sentence['roles']:
                for pred_role in pred_roles:
                    if pred_role == '_' or pred_role == 'V':
                        continue
                    if pred_role not in role_f1:
                        role_f1[pred_role] = {'tp': 0, 'fp': 0, 'fn': 0}
                    role_f1[pred_role]['fp'] += 1
    
    coarse_semantics_f1 = compute_combined_f1(frame_f1, role_f1)
    fine_semantics_f1 = compute_combined_f1(synset_f1, role_f1)

    metrics = {
        'frames': {
            'micro-F1': compute_micro_f1(frame_f1),
            'macro-F1': compute_macro_f1(frame_f1),
        },
        'synsets': {
            'micro-F1': compute_micro_f1(synset_f1),
            'macro-F1': compute_macro_f1(synset_f1),
        },
        'roles': {
            'micro-F1': compute_micro_f1(role_f1),
            'macro-F1': compute_macro_f1(role_f1),
        },
        'overall-semantics': {
            'coarse-grained': coarse_semantics_f1,
            'fine-grained': fine_semantics_f1,
        },
    }

    return metrics


def compute_micro_f1(data:dict) -> Dict[str, float]:
    # Compute the micro-averaged F1 score from the true positives, false positives and false negatives in data.
    tp, fp, fn = 0, 0, 0

    for item, values in data.items():
        tp += values['tp']
        fp += values['fp']
        fn += values['fn']
    
    precision = float(tp) / (float(tp) + float(fp)) if tp + fp > 0 else 0.
    recall = float(tp) / (float(tp) + float(fn)) if tp + fn > 0 else 0.
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0. else 0.

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def compute_macro_f1(data:dict) -> Dict[str, float]:
    # Compute the macro-averaged F1 score from the true positives, false positives and false negatives in data.
    result = {
        'precision': {},
        'recall': {},
        'f1': {},
    }

    for item, values in data.items():
        tp = values['tp']
        fp = values['fp']
        fn = values['fn']
        precision = float(tp) / (float(tp) + float(fp)) if tp + fp > 0 else 0.
        recall = float(tp) / (float(tp) + float(fn)) if tp + fn > 0 else 0.
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0. else 0.
        if tp > 0 or fp > 0:
            result['precision'][item] = precision
        if tp > 0 or fn > 0:
            result['recall'][item] = recall
        result['f1'][item] = f1
    
    precision, recall, f1 = 0., 0., 0.

    for item, value in result['precision'].items():
        precision += value
    
    for item, value in result['recall'].items():
        recall += value
    
    for item, value in result['f1'].items():
        f1 += value
    
    precision /= len(result['precision']) if len(result['precision']) > 0 else 1.
    recall /= len(result['recall']) if len(result['recall']) > 0 else 1.
    f1 /= len(result['f1']) if len(result['f1']) > 0 else 1.

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def compute_combined_f1(data_1:dict, data_2:dict) -> Dict[str, float]:
    # Compute the micro-averaged F1 score from the true positives, false positives and false negatives in data.
    tp, fp, fn = 0, 0, 0

    for item, values in data_1.items():
        tp += values['tp']
        fp += values['fp']
        fn += values['fn']
    
    for item, values in data_2.items():
        tp += values['tp']
        fp += values['fp']
        fn += values['fn']

    precision = float(tp) / (float(tp) + float(fp)) if tp + fp > 0 else 0.
    recall = float(tp) / (float(tp) + float(fn)) if tp + fn > 0 else 0.
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0. else 0.

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def display_metrics(metrics:dict) -> None:
    # Print the metrics in a nice format.
    print()
    print('=' * 80)
    # Print 'Evaluation Report' in the middle of the line.
    print('Evaluation Report'.center(80))
    print('-' * 80)

    print('FRAMES'.center(80))
    print(('-' * 12).center(80))
    print()
    # Display center-aligned column names (micro-precision, micro-recall, micro-F1).
    print('{:^15} | {:^15} | {:^15}'.format('micro-P', 'micro-R', 'micro-F1').center(80))
    # Display the micro-averaged precision, recall and F1 score.
    print('{:^15.2f} | {:^15.2f} | {:^15.2f}'.format(
        metrics['frames']['micro-F1']['precision'] * 100,
        metrics['frames']['micro-F1']['recall'] * 100,
        metrics['frames']['micro-F1']['f1'] * 100,
    ).center(80))
    print(('-' * 51).center(80))
    # Display column names (macro-precision, macro-recall, macro-F1).
    print('{:^15} | {:^15} | {:^15}'.format('macro-P', 'macro-R', 'macro-F1').center(80))
    # Display the macro-averaged precision, recall and F1 score.
    print('{:^15.2f} | {:^15.2f} | {:^15.2f}'.format(
        metrics['frames']['macro-F1']['precision'] * 100,
        metrics['frames']['macro-F1']['recall'] * 100,
        metrics['frames']['macro-F1']['f1'] * 100,
    ).center(80))
    print()

    print('-' * 80)

    print('SYNSETS'.center(80))
    print(('-' * 12).center(80))
    print()
    # Display column names (micro-precision, micro-recall, micro-F1).
    print('{:^15} | {:^15} | {:^15}'.format('micro-P', 'micro-R', 'micro-F1').center(80))
    print('{:^15} | {:^15} | {:^15}'.format('-' * 15, '-' * 15, '-' * 15).center(80))
    # Display the micro-averaged precision, recall and F1 score.
    print('{:^15.2f} | {:^15.2f} | {:^15.2f}'.format(
        metrics['synsets']['micro-F1']['precision'] * 100,
        metrics['synsets']['micro-F1']['recall'] * 100,
        metrics['synsets']['micro-F1']['f1'] * 100,
    ).center(80))
    print()
    # Display column names (macro-precision, macro-recall, macro-F1).
    print('{:^15} | {:^15} | {:^15}'.format('macro-P', 'macro-R', 'macro-F1').center(80))
    print('{:^15} | {:^15} | {:^15}'.format('-' * 15, '-' * 15, '-' * 15).center(80))
    # Display the macro-averaged precision, recall and F1 score.
    print('{:^15.2f} | {:^15.2f} | {:^15.2f}'.format(
        metrics['synsets']['macro-F1']['precision'] * 100,
        metrics['synsets']['macro-F1']['recall'] * 100,
        metrics['synsets']['macro-F1']['f1'] * 100,
    ).center(80))
    print()

    print('-' * 80)

    print('SEMANTIC ROLES'.center(80))
    print(('-' * 12).center(80))
    print()
    # Display column names (micro-precision, micro-recall, micro-F1).
    print('{:^15} | {:^15} | {:^15}'.format('micro-P', 'micro-R', 'micro-F1').center(80))
    print('{:^15} | {:^15} | {:^15}'.format('-' * 15, '-' * 15, '-' * 15).center(80))
    # Display the micro-averaged precision, recall and F1 score.
    print('{:^15.2f} | {:^15.2f} | {:^15.2f}'.format(
        metrics['roles']['micro-F1']['precision'] * 100,
        metrics['roles']['micro-F1']['recall'] * 100,
        metrics['roles']['micro-F1']['f1'] * 100,
    ).center(80))
    print()
    # Display column names (macro-precision, macro-recall, macro-F1).
    print('{:^15} | {:^15} | {:^15}'.format('macro-P', 'macro-R', 'macro-F1').center(80))
    print('{:^15} | {:^15} | {:^15}'.format('-' * 15, '-' * 15, '-' * 15).center(80))
    # Display the macro-averaged precision, recall and F1 score.
    print('{:^15.2f} | {:^15.2f} | {:^15.2f}'.format(
        metrics['roles']['macro-F1']['precision'] * 100,
        metrics['roles']['macro-F1']['recall'] * 100,
        metrics['roles']['macro-F1']['f1'] * 100,
    ).center(80))
    print()

    print('-' * 80)

    print('OVERALL SEMANTIC SCORE'.center(80))
    print(('-' * 12).center(80))
    print()
    # Display column names (coarse-precision, coarse-recall, coarse-F1).
    print('{:^15} | {:^15} | {:^15}'.format('coarse-P', 'coarse-R', 'coarse-F1').center(80))
    print('{:^15} | {:^15} | {:^15}'.format('-' * 15, '-' * 15, '-' * 15).center(80))
    # Display the coarse-grained precision, recall and F1 score.
    print('{:^15.2f} | {:^15.2f} | {:^15.2f}'.format(
        metrics['overall-semantics']['coarse-grained']['precision'] * 100,
        metrics['overall-semantics']['coarse-grained']['recall'] * 100,
        metrics['overall-semantics']['coarse-grained']['f1'] * 100,
    ).center(80))
    print()
    # Display column names (fine-precision, fine-recall, fine-F1).
    print('{:^15} | {:^15} | {:^15}'.format('fine-P', 'fine-R', 'fine-F1').center(80))
    print('{:^15} | {:^15} | {:^15}'.format('-' * 15, '-' * 15, '-' * 15).center(80))
    # Display the fine-grained precision, recall and F1 score.
    print('{:^15.2f} | {:^15.2f} | {:^15.2f}'.format(
        metrics['overall-semantics']['fine-grained']['precision'] * 100,
        metrics['overall-semantics']['fine-grained']['recall'] * 100,
        metrics['overall-semantics']['fine-grained']['f1'] * 100,
    ).center(80))
    print()

    print('=' * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gold',
        type=str,
        required=True,
        dest='gold_path',
        help='Path to the preprocessed data file to use for building the vocabularies.')
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        dest='predictions_path',
        help='Path to the output file for the vocabulary.')
    parser.add_argument(
        '--log',
        type=str,
        default='INFO',
        dest='loglevel',
        help='Log level. Default = INFO.')

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))
    logging.info('Evaluating {}...'.format(args.predictions_path))

    gold_data = read_file(args.gold_path)
    pred_data = read_file(args.predictions_path)
    logging.info('Gold data: {} sentences'.format(len(gold_data)))
    logging.info('Pred data: {} sentences'.format(len(pred_data)))

    metrics = evaluate(gold_data, pred_data)
    display_metrics(metrics)

    logging.info('Done!')
