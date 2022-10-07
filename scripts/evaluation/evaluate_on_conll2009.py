import argparse
import json
import logging
import subprocess


def load_predictions(predictions_path:str) -> dict:
    with open(predictions_path) as f:
        predictions = json.load(f)
    
    predictions = {
        int(sentence_idx): {
            int(predicate_idx): predicate_v for predicate_idx, predicate_v in sentence_v.items()
        } for sentence_idx, sentence_v in predictions.items()
    }

    return predictions


def write_predictions_to_conll(predictions_path:str, gold_path:str, czech: bool = False):
    output_path = predictions_path.replace('.json', '.conll')
    predictions = load_predictions(predictions_path)

    with open(gold_path) as f_gold, open(output_path, 'w') as f_pred:
        sentence_id = 0
        sentence_parts = []

        for line in f_gold:
            line = line.strip()

            if not line:
                sentence_roles = []

                if sentence_id in predictions:
                    predicate_indices = sorted(predictions[sentence_id].keys())

                    for predicate_idx in predicate_indices:
                        predicate_predictions = predictions[sentence_id][predicate_idx]
                        sentence_parts[predicate_idx][12] = 'Y'
                        predicted_predicate = predicate_predictions['predicate'].replace('-v.', '.')
                        if czech and predicted_predicate == '[no-sense]':
                            predicted_predicate = sentence_parts[predicate_idx][3]
                        sentence_parts[predicate_idx][13] = predicted_predicate if predicted_predicate != '<UNK>' and predicted_predicate != '_' else 'unknown.01'
                        sentence_roles.append(predicate_predictions['roles'])
                
                if sentence_roles:
                    sentence_roles = list(zip(*sentence_roles))
                    for line, line_roles in zip(sentence_parts, sentence_roles):
                        line.extend(line_roles)

                for line in sentence_parts:
                    f_pred.write('\t'.join(line) + '\n')
                f_pred.write('\n')
                
                sentence_id += 1
                sentence_parts = []
                continue

            parts = line.split('\t')[:12] + ['_', '_']
            sentence_parts.append(parts)

    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gold',
        type=str,
        required=True,
        dest='gold_path',
        help='Path to original gold file.')
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        dest='predictions_path',
        help='Path to the json file containing the predictions.')
    parser.add_argument(
        '--scorer',
        default='scripts/evaluation/scorer_conll2009.pl',
        dest='scorer_path',
    )
    parser.add_argument(
        '--czech',
        action='store_true',
    )
    parser.add_argument(
        '--log',
        type=str,
        default='WARNING',
        dest='loglevel',
        help='Log level. Default = WARNING.')

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))
    logging.info('Evaluating {}...'.format(args.predictions_path))

    output_path = write_predictions_to_conll(args.predictions_path, args.gold_path, czech=args.czech)

    arguments = ['perl', args.scorer_path, '-q', '-g', args.gold_path, '-s', output_path]
    scorer_output = subprocess.run(arguments, capture_output=True, text=True).stdout
    print(scorer_output)

    logging.info('Done!')
