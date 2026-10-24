import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.dataset import get_tokenized_datasets
from src.logger.cometml import log_final_model, setup_comet
from src.metrics import compute_metrics
from src.model import get_model
from src.utils.init import setup_saving_and_logging


@hydra.main(version_base='1.3', config_path='src/configs', config_name='train')
def main(config: DictConfig) -> None:
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Parameters
    ----------
    config : DictConfig
        Hydra experiment config.
    """
    set_seed(config.training_args.seed)

    logger = setup_saving_and_logging(config)
    setup_comet(config, logger)

    logger.info(f'Config:\n{OmegaConf.to_yaml(config, resolve=True)}')

    training_args = TrainingArguments(**config.training_args)
    logger.info(f'Device: {training_args.device}')

    datasets = get_tokenized_datasets(**config.dataset)
    logger.info(f'Datasets:\n{datasets}')

    tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer)
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)  # ty:ignore[invalid-argument-type]

    model = get_model(**config.model)
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
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if config.resume_from_checkpoint is None:
        trainer.evaluate()

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    trainer.save_model()
    trainer.save_state()

    test_metrics = trainer.evaluate(
        eval_dataset=datasets['test'],  # ty:ignore[invalid-argument-type]
        metric_key_prefix='test',
    )
    trainer.log_metrics('test', test_metrics)
    trainer.save_metrics('test', test_metrics)

    log_final_model(training_args.output_dir, config.writer.run_name, logger)  # ty:ignore[invalid-argument-type]


if __name__ == '__main__':
    main()
