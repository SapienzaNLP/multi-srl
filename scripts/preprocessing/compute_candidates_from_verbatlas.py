import argparse
import json
import logging


def load_lemma2va(wn2lemma_path:str, bn2wn_path:str, bn2va_path:str, frame_info_path:str) -> dict:
    id2frame = {}
    with open(frame_info_path) as f:
        for line_no, line in enumerate(f):
            if line_no == 0:
                continue
            frame_id, frame_name, *_ = line.strip().split('\t')
            id2frame[frame_id] = frame_name
    
    bn2va = {}
    with open(bn2va_path) as f:
        for line_no, line in enumerate(f):
            if line_no == 0:
                continue
            bn, va_id = line.strip().split('\t')
            assert bn not in bn2va
            va = id2frame[va_id]
            bn2va[bn] = va
    
    wn2va = {}
    with open(bn2wn_path) as f:
        for line_no, line in enumerate(f):
            if line_no == 0:
                continue
            bn, wn = line.strip().split('\t')
            assert wn not in wn2va
            va = bn2va[bn]
            wn2va[wn] = va
    
    lemma2va = {}
    with open(wn2lemma_path) as f:
        for line_no, line in enumerate(f):
            wn, lemma = line.strip().split('\t')
            lemma = lemma.lower().split('-')[0]
            if lemma not in lemma2va:
                lemma2va[lemma] = set()
            va = wn2va[wn]
            lemma2va[lemma].add(va)
    
    return lemma2va


def compute_vocabulary(lemma2va: dict, vocab_path: str, null_label: str = '_', unknown_label: str = '<UNK>'):
    """
    Compute the vocabulary from the PropBank frames.
    """

    # Compute the vocabulary.
    sense_vocabulary = {
        null_label: 0,
        unknown_label: 1,
    }
    candidates_vocabulary = {}

    for lemma, va_set in lemma2va.items():
        for va in va_set:
            if va not in sense_vocabulary:
                sense_vocabulary[va] = len(sense_vocabulary)
        
        if lemma not in candidates_vocabulary:
            candidates_vocabulary[lemma] = [frame for frame in va_set]
        else:
            candidates_vocabulary[lemma].extend(va_set)

    # Load exisiting vocab.
    with open(vocab_path) as f:
        vocab = json.load(f)
    
    for lemma, candidates in vocab['candidates'].items():
        if lemma not in candidates_vocabulary:
            candidates_vocabulary[lemma] = candidates
        else:
            candidates_vocabulary[lemma].extend(candidates)
    
    for frame, idx in vocab['senses'].items():
        if frame not in sense_vocabulary:
            sense_vocabulary[frame] = len(sense_vocabulary)
    
    # Sort all candidates.
    for lemma, candidates in candidates_vocabulary.items():
        candidates_vocabulary[lemma] = sorted(list(set(candidates)))

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
        '--wn2lemma',
        type=str,
        default='data/resources/verbatlas-1.1/wn2lemma.tsv',
        help='Path to the wn2lemma.tsv file.')
    parser.add_argument(
        '--bn2wn',
        type=str,
        default='data/resources/verbatlas-1.1/bn2wn.tsv',
        help='Path to the bn2wn.tsv file.')
    parser.add_argument(
        '--bn2va',
        type=str,
        default='data/resources/verbatlas-1.1/VA_bn2va.tsv',
        help='Path to the VA_bn2va.tsv file.')
    parser.add_argument(
        '--frame_info',
        type=str,
        default='data/resources/verbatlas-1.1/VA_frame_info.tsv',
        help='Path to VA_frame_info.tsv file.')
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
    logging.info('Computing...')

    lemma2va = load_lemma2va(args.wn2lemma, args.bn2wn, args.bn2va, args.frame_info)
    vocabulary = compute_vocabulary(lemma2va, args.vocab_path)
    write_vocabulary(args.output_path, vocabulary)

    logging.info('Done!')
