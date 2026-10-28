from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.dataset import get_tokenized_datasets
from src.distillation import DataCollatorWithTeacherLogits, DistillationTrainer, add_teacher_logits
from src.logger.cometml import log_final_model, setup_comet
from src.metrics import compute_metrics
from src.model import get_model, load_ner_model
from src.utils.init import setup_saving_and_logging
from src.utils.io import get_root


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
    if config.distillation.enabled:
        # The 'teacher_logits' column is not a model input, so the Trainer would drop it.
        training_args.remove_unused_columns = False
    logger.info(f'Device: {training_args.device}')

    datasets = get_tokenized_datasets(**config.dataset)
    logger.info(f'Datasets:\n{datasets}')

    tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer)
    data_collator = DataCollatorWithTeacherLogits(tokenizer, pad_to_multiple_of=8)  # ty:ignore[invalid-argument-type]

    if config.distillation.enabled:
        teacher_path = get_root() / config.distillation.teacher_path
        teacher = load_ner_model(str(teacher_path))
        logger.info(f'Teacher: {teacher.num_parameters() / 1e6:.2f}M parameters')
        datasets['train'] = add_teacher_logits(
            datasets['train'],
            teacher,
            teacher_path,
            data_collator,
            batch_size=config.distillation.precompute_batch_size,
            device=training_args.device,
        )
        del teacher  # the teacher is not needed during training

    model = get_model(**config.model)
    logger.info(
        f'Model:\n{model}\n'
        f'Parameters: {model.num_parameters() / 1e6:.2f}M total, '
        f'{model.num_parameters(exclude_embeddings=True) / 1e6:.2f}M non-embedding'
    )

    trainer_cls = (
        partial(DistillationTrainer, temperature=config.distillation.temperature)
        if config.distillation.enabled
        else Trainer
    )
    trainer = trainer_cls(
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
