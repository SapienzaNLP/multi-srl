import os
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from data.dataset import CoNLL
from data.processor import Processor
from models.model import MultilingualSrlModel

if __name__ == '__main__':
    parser = ArgumentParser()

    # Add trial name.
    parser.add_argument('--name', type=str, required=True)

    # Add seed arg.
    parser.add_argument('--seed', type=int, default=313)

    # Add data args.
    parser.add_argument('--train_path', type=str, default='data/json/en/CoNLL2009_train.json')
    parser.add_argument('--dev_path', type=str, default='data/json/en/CoNLL2009_dev.json')

    # Add dataloader args.
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=8)

    # Add checkpoint args.
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    # Add model-specific args.
    parser = MultilingualSrlModel.add_model_specific_args(parser)

    # Add all the available trainer options to argparse.
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(
        min_epochs=3,
        max_epochs=30,
        gpus=1,
        precision=16,
        gradient_clip_val=1.0,
        row_log_interval=128,
        deterministic=True,
    )

    # Store the arguments in hparams.
    hparams = parser.parse_args()

    # Initialize random generators with the given seed.
    seed_everything(hparams.seed)

    # Load the train dataset.
    train_dataset = CoNLL(hparams.train_path)

    # Load the validation dataset.
    dev_dataset = CoNLL(hparams.dev_path)

    # The processor takes care of building i/o maps, encode a sentence, and decode the output.
    processor = Processor(
        train_dataset,
        input_representation=hparams.input_representation,
        model_name=hparams.language_model)

    # Create the dataloader for the training set.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=hparams.shuffle,
        num_workers=hparams.num_workers,
        collate_fn=processor.collate_sentences)

    # Create the dataloader for the validation set.
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        collate_fn=processor.collate_sentences)

    # Additional hparams.
    # steps_per_epoch: Number of steps for each epoch. Used for learning rate warmup and cooldown.
    # num_roles: Number of unique roles in the output layer.
    # num_senses: Number of unique senses in the output layer.
    hparams.steps_per_epoch = int(len(train_dataset) / (hparams.batch_size * hparams.accumulate_grad_batches)) + 1
    hparams.num_roles = processor.num_roles
    hparams.num_senses = processor.num_senses

    # Create the model with the given hparams.
    model = MultilingualSrlModel(hparams, padding_token_id=processor.padding_token_id)

    # Set the dir where the processor config (with the i/o maps) and the best checkpoints will be saved.
    model_dir = os.path.join(hparams.checkpoint_dir, hparams.name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the processor config to a JSON file.
    processor_config_path = os.path.join(model_dir, 'processor_config.json')
    processor.save_config(processor_config_path)

    # Set the dir where the model checkpoints will be saved.
    model_checkpoint_path = os.path.join(model_dir, 'checkpoint_{epoch:03d}-{val_f1:0.4f}')
    checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint_path,
        monitor='val_f1',
        mode='max')

    # Create the Trainer which will be taking care of training the model.
    trainer = Trainer.from_argparse_args(
        hparams,
        checkpoint_callback=checkpoint_callback)

    # Train the model.
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=dev_dataloader)
