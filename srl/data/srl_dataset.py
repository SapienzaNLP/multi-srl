import json
from typing import List

from torch.utils.data import Dataset


class SrlDataset(Dataset):

    def __init__(self, path_to_data: str = None, sentences: List = None):
        super().__init__()
        if path_to_data is not None:
            self.sentences = SrlDataset.load_sentences_from_file(path_to_data)
        else:
            self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

    @staticmethod
    def load_sentences_from_file(path: str) -> List[dict]:
        sentences = []

        with open(path) as json_file:

            for i, sentence in json.load(json_file).items():
                if 'predicate_indices' not in sentence and 'annotations' not in sentence:
                    continue
                
                if 'predicate_indices' in sentence and not sentence['predicate_indices']:
                    continue

                if 'annotations' in sentence and not sentence['annotations']:
                    continue

                if len(sentence['words']) > 128:
                    continue

                sample = {
                    'sentence_id': i,
                    'words': SrlDataset.process_words(sentence['words']),
                    'lemmas': sentence['lemmas'],
                }

                if 'annotations' in sentence:
                    sample['annotations'] = { int(idx): a for idx, a in sentence['annotations'].items() }
                elif 'predicate_indices' in sentence:
                    sample['predicate_indices'] = sentence['predicate_indices']

                sentences.append(sample)

        return sentences
    
    @staticmethod
    def load_sentences(data: list) -> List[dict]:
        sentences = []

        for i, sentence in enumerate(data):
            sentence_id = i
            words = SrlDataset.process_words([w.text for w in sentence.words])
            lemmas = [w.lemma for w in sentence.words]
            pos_tags = [w.upos for w in sentence.words]

            sample = {
                'sentence_id': sentence_id,
                'words': words,
                'lemmas': lemmas,
            }
            sample['predicate_indices'] = [i for i, pos_tag in enumerate(pos_tags) if pos_tag in ['VERB']]
            sentences.append(sample)
        
        return sentences

    @staticmethod
    def process_words(words:List[str]) -> List[str]:
        processed_words = []
        for word in words:
            processed_word = SrlDataset.clean_word(word)
            processed_words.append(processed_word)
        return processed_words
    
    @staticmethod
    def clean_word(word:str) -> str:
        if word == "n\'t":
            return 'not'
        if word == 'wo':
            return 'will'
        if word == "'ll":
            return 'will'
        if word == "'m":
            return 'am'
        if word == '``':
            return '"'
        if word == "''":
            return '"'
        if word == '/.':
            return '.'
        if word == '/-':
            return '...'
        if word == '-LRB-' or word == '-LSB-' or word == '-LCB-':
            return '('
        if word == '-RRB-' or word == '-RSB-' or word == '-RCB-':
            return ')'
        
        if '\\/' in word:
            word = word.replace('\\/', '/')

        return word
