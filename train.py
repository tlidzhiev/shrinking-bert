import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, Trainer, TrainingArguments

from src.dataset import NerDataCollator, get_tokenized_datasets
from src.logger import setup_comet
from src.metrics import compute_metrics
from src.model import build_model

logger = logging.getLogger(__name__)


@hydra.main(version_base='1.3', config_path='src/configs', config_name='train')
def main(cfg: DictConfig) -> None:
    """
    Main script for training. Instantiates the model, tokenizer, dataset,
    logger, and metrics from the Hydra config. Runs Trainer to train
    and evaluate the model.

    Parameters
    ----------
    cfg : DictConfig
        Hydra experiment config.
    """
    logger.info(f'Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}')
    setup_comet(cfg)

    training_args = TrainingArguments(**cfg.training_args)
    logger.info(f'Device: {training_args.device}')

    datasets = get_tokenized_datasets(**cfg.dataset)
    logger.info(f'Datasets:\n{datasets}')

    # Sequences are tokenized without padding, so batches have to be padded on collation.
    tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.tokenizer)
    data_collator = NerDataCollator(tokenizer)  # ty:ignore[invalid-argument-type]

    model = build_model(**cfg.model)
    logger.info(
        f'Model:\n{model}\n'
        f'Parameters: {model.num_parameters() / 1e6:.2f}M total, '
        f'{model.num_parameters(exclude_embeddings=True) / 1e6:.2f}M non-embedding'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model()
    trainer.save_state()

    test_metrics = trainer.evaluate(
        eval_dataset=datasets['test'],  # ty:ignore[invalid-argument-type]
        metric_key_prefix='test',
    )
    trainer.log_metrics('test', test_metrics)
    trainer.save_metrics('test', test_metrics)


if __name__ == '__main__':
    main()
