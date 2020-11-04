import argparse
import json

from collections import Counter


def compute_stats(path, czech=False, conll_2012=False):
    stats = {}

    with open(path) as f:
        data = json.load(f)
    
    stats['sentences'] = len(data)

    stats['non-empty_sentences'] = 0
    stats['predicate_instances'] = 0
    stats['role_instances'] = 0
    stats['avg_sentence_length'] = 0
    stats['sentence_lengths'] = {
        '<10': 0,
        '<25': 0,
        '<50': 0,
        '<100': 0,
        '<150': 0,
        '<200': 0,
        '+200': 0,
    }
    unique_predicates = set()
    unique_roles = set()
    for sentence in data.values():
        predicates = [p for p in sentence['predicates'] if p != '_']
        if predicates:
            stats['non-empty_sentences'] += 1
            stats['avg_sentence_length'] += len(sentence['words'])
            stats['predicate_instances'] += len(predicates)
            sentence_length = len(sentence['words'])
            if sentence_length >= 200:
                stats['sentence_lengths']['+200'] += 1
            elif sentence_length >= 150:
                stats['sentence_lengths']['<200'] += 1
            elif sentence_length >= 100:
                stats['sentence_lengths']['<150'] += 1
            elif sentence_length >= 50:
                stats['sentence_lengths']['<100'] += 1
            elif sentence_length >= 25:
                stats['sentence_lengths']['<50'] += 1
            elif sentence_length >= 10:
                stats['sentence_lengths']['<25'] += 1
            else:
                stats['sentence_lengths']['<10'] += 1

            unique_predicates.update(predicates)
        
        roles = [r for predicate_roles in sentence['roles'].values() for r in predicate_roles if r != '_']
        if roles:
            if conll_2012:
                roles = [r[2:] for r in roles if r[0] == 'B']
            stats['role_instances'] += len(roles)
            unique_roles.update(roles)
    
    stats['unique_predicates'] = len(unique_predicates)
    if czech:
        stats['unique_predicates'] -= 1

    stats['unique_roles'] = len(unique_roles)
    stats['avg_roles_per_predicate'] = stats['role_instances'] / stats['predicate_instances']
    stats['avg_sentence_length'] /= stats['non-empty_sentences']

    predicate_types = Counter([p.split('.')[-1] for p in unique_predicates])
    print(predicate_types.most_common(10))
    return stats


def print_stats(path, stats):
    print('  Stats for {}'.format(path))
    for key, value in stats.items():
        if isinstance(value, dict):
            print(value)
        elif isinstance(value, float):
            print('    {}: {:0.2f}'.format(key, value))
        else:
            print('    {}: {}'.format(key, value))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        dest='input_path',
        help='Path to the data to preprocess.')
    parser.add_argument(
        '--conll_2012',
        action='store_true')
    parser.add_argument(
        '--czech',
        action='store_true')
    args = parser.parse_args()

    stats = compute_stats(args.input_path, conll_2012=args.conll_2012, czech=args.czech)
    print_stats(args.input_path, stats)

