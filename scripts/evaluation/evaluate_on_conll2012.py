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


def write_predictions_to_conll(predictions_path:str, gold_path:str):
    pred_output_path = predictions_path.replace('.json', '.conll')
    gold_output_path = predictions_path.replace('.json', '.gold.conll')
    predictions = load_predictions(predictions_path)

    with open(gold_path) as fin_gold, open(pred_output_path, 'w') as fout_pred, open(gold_output_path, 'w') as fout_gold:
        sentence_id = 0
        sentence_lemmas = []
        sentence_gold_roles = []

        for line in fin_gold:
            line = line.strip()

            if line.startswith('#'):
                continue

            if not line:
                sentence_roles = []

                if sentence_id in predictions:
                    predicate_indices = sorted(predictions[sentence_id].keys())

                    for predicate_idx in predicate_indices:
                        predicate_predictions = predictions[sentence_id][predicate_idx]
                        predicate_roles = convert_bio2conll(predicate_predictions['roles'])
                        sentence_roles.append(predicate_roles)
                
                if sentence_roles:
                    sentence_roles = list(zip(*sentence_roles))

                    for line_no, predicate in enumerate(sentence_lemmas):
                        pred_roles = sentence_roles[line_no]
                        gold_roles = sentence_gold_roles[line_no]
                        fout_pred.write(predicate + '\t' + '\t'.join(pred_roles) + '\n')
                        fout_gold.write(predicate + '\t' + '\t'.join(gold_roles) + '\n')
                    fout_pred.write('\n')
                    fout_gold.write('\n')
                
                sentence_id += 1
                sentence_lemmas = []
                sentence_gold_roles = []
                continue

            sentence_parts = line.strip().split()
            predicate = sentence_parts[6] if sentence_parts[7] != '-' else '-'
            sentence_lemmas.append(predicate)
            sentence_gold_roles.append(sentence_parts[11:-1])

    return gold_output_path, pred_output_path


def convert_bio2conll(bio_tags:list):
    conll_sequence = []

    for tag_index, tag in enumerate(bio_tags):
        conll_tag = tag[2:]
        if tag.startswith('B-'):
            if tag_index == len(bio_tags) - 1 or not bio_tags[tag_index + 1].startswith('I-'):
                conll_sequence.append(f'({conll_tag}*)')
            else:
                conll_sequence.append(f'({conll_tag}*')
        elif tag.startswith('I-'):
            if tag_index == len(bio_tags) - 1 or not bio_tags[tag_index + 1].startswith('I-'):
                conll_sequence.append(f'*)')
            else:
                conll_sequence.append(f'*')
        else:
            conll_sequence.append('*')
    
    return conll_sequence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gold',
        type=str,
        required=True,
        dest='gold_path',
        help='Path to the original gold file.')
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        dest='predictions_path',
        help='Path to the json file containing the predictions.')
    parser.add_argument(
        '--scorer',
        default='scripts/evaluation/scorer_conll2005.pl',
        dest='scorer_path',
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

    gold_output_path, pred_output_path = write_predictions_to_conll(args.predictions_path, args.gold_path)

    arguments = ['perl', args.scorer_path, gold_output_path, pred_output_path]
    scorer_output = subprocess.run(arguments, capture_output=True, text=True).stdout
    print(scorer_output)

    logging.info('Done!')
