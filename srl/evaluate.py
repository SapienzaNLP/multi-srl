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
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--processor', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--viterbi_decoding', action='store_true')

    # Add dataloader args.
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)

    # Store the arguments in hparams.
    args = parser.parse_args()

    processor = Processor.from_config(args.processor, viterbi_decoding=args.viterbi_decoding)

    test_dataset = CoNLL(args.input)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=processor.collate_sentences)

    model = MultilingualSrlModel.load_from_checkpoint(args.model)
    model.eval()

    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0)
    trainer.test(model=model, test_dataloaders=test_dataloader)
