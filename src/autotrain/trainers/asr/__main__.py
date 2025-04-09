import argparse
import json
from functools import partial

from accelerate.state import PartialState
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoModelForCTC,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_callback import PrinterCallback

from autotrain import logger
from autotrain.trainers.common import (
    ALLOW_REMOTE_CODE,
    LossLoggingCallback,
    TrainStartCallback,
    UploadLogs,
    monitor,
    pause_space,
    remove_autotrain_data,
    save_training_params,
)
from autotrain.trainers.asr.dataset import ASRDataset
from autotrain.trainers.asr.params import ASRParams
from autotrain.trainers.asr.utils import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


@monitor
def train(config):
    if isinstance(config, dict):
        config = ASRParams(**config)

    train_data = None
    valid_data = None

    if config.data_path == f"{config.project_name}/autotrain-data":
        train_data = load_from_disk(config.data_path)[config.train_split]
    else:
        train_data = load_dataset(
            config.data_path,
            split=config.train_split,
            token=config.token,
            trust_remote_code=ALLOW_REMOTE_CODE,
        )

    if config.valid_split is not None:
        valid_data = load_dataset(
            config.data_path,
            split=config.valid_split,
            token=config.token,
            trust_remote_code=ALLOW_REMOTE_CODE,
        )

    processor = AutoProcessor.from_pretrained(config.model, token=config.token, trust_remote_code=ALLOW_REMOTE_CODE)
    train_data = ASRDataset(data=train_data, processor=processor, config=config)
    if valid_data:
        valid_data = ASRDataset(data=valid_data, processor=processor, config=config)

    training_args = TrainingArguments(
        output_dir=config.project_name,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        evaluation_strategy="steps" if valid_data else "no",
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        gradient_accumulation_steps=config.gradient_accumulation,
        report_to=config.log,
        push_to_hub=config.push_to_hub,
        load_best_model_at_end=True if valid_data else False,
    )

    model = AutoModelForCTC.from_pretrained(config.model, trust_remote_code=ALLOW_REMOTE_CODE, token=config.token)

    callbacks = [LossLoggingCallback(), TrainStartCallback(), UploadLogs(config=config)]
    if valid_data:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=processor,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(config.project_name)
    processor.save_pretrained(config.project_name)

    if config.push_to_hub:
        remove_autotrain_data(config)
        save_training_params(config)
        api = HfApi(token=config.token)
        api.create_repo(repo_id=f"{config.username}/{config.project_name}", repo_type="model", private=True)
        api.upload_folder(folder_path=config.project_name, repo_id=f"{config.username}/{config.project_name}")

    if PartialState().process_index == 0:
        pause_space(config)


if __name__ == "__main__":
    args = parse_args()
    training_config = json.load(open(args.training_config))
    config = ASRParams(**training_config)
    train(config)