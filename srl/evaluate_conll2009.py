import subprocess
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from data.dataset import CoNLL
from data.processor import Processor
from models.model import MultilingualSrlModel

if __name__ == '__main__':
    parser = ArgumentParser()

    # Add data args.
    parser.add_argument('--scorer', type=str, default='scripts/evaluation/scorer_conll2009.pl')
    parser.add_argument('--processor', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--test_json', type=str, required=True)
    parser.add_argument('--test_txt', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--czech', action='store_true')

    # Add dataloader args.
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)

    # Store the arguments in hparams.
    args = parser.parse_args()

    processor = Processor.from_config(args.processor, viterbi_decoding=False)

    test_dataset = CoNLL(args.test_json)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=processor.collate_sentences)

    model = MultilingualSrlModel.load_from_checkpoint(args.model)
    model.eval()

    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0)
    trainer.test(model=model, test_dataloaders=test_dataloader)

    predictions = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    with torch.no_grad():
        for x, _ in test_dataloader:
            x = {k: v.to(device) if not isinstance(v, list) else v for k, v in x.items()}
            y = model(x)
            batch_predictions = processor.decode(x, y)

            for i, sentence_id in enumerate(x['sentence_ids']):
                predictions[int(sentence_id)] = {
                    'predicates': batch_predictions['predicates'][i],
                    'senses': batch_predictions['senses'][i],
                    'roles': batch_predictions['roles'][i],
                }

    sentence_id = 0
    sentence_output = []
    sentence_senses = []
    with open(args.test_txt) as f_in, open(args.output, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                if sentence_id not in predictions:
                    for i in range(len(sentence_output)):
                        output_line = '\t'.join(sentence_output[i])
                        f_out.write('{}\t_\n'.format(output_line))
                    f_out.write('\n')
                    sentence_id += 1
                    sentence_output = []
                    sentence_senses = []
                    continue

                predicted_senses = predictions[sentence_id]['senses']
                output_senses = []
                for predicate_index, (gold, predicted) in enumerate(zip(sentence_senses, predicted_senses)):
                    if gold != '_' and predicted != '_':
                        lemma = sentence_output[predicate_index][3]
                        if args.czech and predicted == 'lemma':
                            output_senses.append(lemma)
                        else:
                            output_senses.append(predicted)
                    elif gold != '_' and predicted == '_':
                        if args.czech:
                            output_senses.append(sentence_output[predicate_index][3])
                        else:
                            output_senses.append('unk.01')
                    else:
                        output_senses.append('_')

                predicted_roles = predictions[sentence_id]['roles']
                output_roles = []
                for i in range(len(output_senses)):
                    if output_senses[i] != '_':
                        output_roles.append(predicted_roles[i])

                output_roles = list(map(list, zip(*output_roles)))
                for i in range(len(sentence_output)):
                    if output_roles:
                        line_parts = sentence_output[i] + [output_senses[i]] + output_roles[i]
                    else:
                        line_parts = sentence_output[i] + [output_senses[i]]
                    output_line = '\t'.join(line_parts)
                    f_out.write('{}\n'.format(output_line))
                f_out.write('\n')

                sentence_id += 1
                sentence_output = []
                sentence_senses = []
                continue

            parts = line.split('\t')
            sentence_output.append(parts[:13])
            sentence_senses.append(parts[13])

    subprocess.run(['perl', args.scorer, '-g', args.test_txt, '-s', args.output, '-q'])
