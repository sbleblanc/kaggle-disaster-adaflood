import click
import mlflow
import evaluate
import torch
import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_dataset
from mlflow.transformers import log_model
from sklearn.model_selection import KFold
from trainers.flood import FloodingTrainer

@click.group()
def cli():
    pass


@cli.command()
@click.option("-s", "--seed", type=click.INT, default=42)
@click.option("-m", "--model-name", type=click.STRING, default="bert-base-cased")
@click.option("-r", "--run-name", type=click.STRING, required=True)
@click.option("-d", "--train-dataset-path", type=click.Path(dir_okay=False, path_type=Path), default="data/train.csv")
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default="model_outputs/")
@click.option("-lr", "--learning-rate", type=click.FLOAT, default=1e-4)
@click.option("-bs", "--batch-size", type=click.INT, default=32)
@click.option("-e", "--epochs", type=click.INT, default=10)
@click.option("-wd", "--weight-decay", type=click.FLOAT, default=1e-4)
def train(
        seed: int,
        model_name: str,
        run_name: str,
        train_dataset_path: Path,
        output_dir: Path,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        weight_decay: float
):
    set_seed(seed)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.enable_system_metrics_logging()

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    ds = load_dataset("csv", data_files=str(train_dataset_path))
    ds = ds.remove_columns(["id", "keyword", "location"]).rename_column("target", "labels")
    ds = ds.map(tok, batched=True, input_columns="text", remove_columns="text")
    ds = ds['train'].train_test_split(test_size=0.2, shuffle=True)

    data_collator = DataCollatorWithPadding(tokenizer=tok)
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return f1_metric.compute(predictions=predictions, references=labels)

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        load_best_model_at_end=True,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        weight_decay=weight_decay,
        num_train_epochs=epochs,
        metric_for_best_model="loss",
        remove_unused_columns=False
    )

    trainer = FloodingTrainer(
        b=0.0,
        model=model,
        args=train_args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name=run_name):
        trainer.train()

        log_model(
            transformers_model={
                "model": model,
                "tokenizer": tok
            },
            artifact_path="model",
            task="text-classification",
            pip_requirements="requirements.txt"
        )


@cli.command()
@click.option("-s", "--seed", type=click.INT, default=42)
@click.option("-m", "--model-name", type=click.STRING, default="bert-base-cased")
@click.option("-d", "--train-dataset-path", type=click.Path(dir_okay=False, path_type=Path), default="data/train.csv")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default="model_outputs/")
@click.option("-lr", "--learning-rate", type=click.FLOAT, default=1e-4)
@click.option("-bs", "--batch-size", type=click.INT, default=32)
@click.option("-e", "--epochs", type=click.INT, default=10)
@click.option("-wd", "--weight-decay", type=click.FLOAT, default=1e-4)
@click.option("-o", "--output-train-csv", type=click.Path(dir_okay=False, path_type=Path), default="data/train_flood.csv")
@click.option("-g", "--gamma", type=click.FloatRange(0.0, 1.0), required=True)
@click.option("-f", "--num-folds", type=click.INT, default=5)
def adaflood_train(
        seed: int,
        model_name: str,
        train_dataset_path: Path,
        output_dir: Path,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        weight_decay: float,
        output_train_csv: Path,
        gamma: float,
        num_folds: int
):
    set_seed(seed)

    tok = AutoTokenizer.from_pretrained(model_name)

    ds = load_dataset("csv", data_files=str(train_dataset_path), split='train')
    ds = ds.remove_columns(["keyword", "location"]).rename_column("target", "labels")
    ds = ds.map(tok, batched=True, input_columns="text", remove_columns="text")
    ds = ds.shuffle()

    data_collator = DataCollatorWithPadding(tokenizer=tok)
    kf = KFold(n_splits=num_folds)

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        load_best_model_at_end=True,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        weight_decay=weight_decay,
        num_train_epochs=epochs,
        metric_for_best_model="loss"
    )

    flood_levels_per_id = []

    for train_idx, eval_idx in kf.split(ds):
        train_ds = ds.select(train_idx)
        eval_ds = ds.select(eval_idx)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tok,
            data_collator=data_collator
        )
        trainer.train()
        model.eval()

        dl = DataLoader(
            eval_ds.with_format("torch").remove_columns('id'),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator
        )
        with torch.no_grad():
            flood_levels = []
            for b in tqdm.tqdm(dl, desc="Computing flood weights", unit="batch"):
                cuda_inputs = {k: v.cuda() for k, v in b.items()}
                res = model(**cuda_inputs)
                probs = torch.nn.functional.softmax(res['logits'], dim=1)
                fx = torch.gather(probs, dim=1, index=b['labels'].unsqueeze(dim=-1).cuda()).squeeze(dim=-1)
                phi = (1-gamma) * fx + gamma
                theta = -torch.log(phi)
                flood_levels.append(theta.cpu())

            df = eval_ds.select_columns("id").to_pandas()
            df['flood'] = torch.cat(flood_levels)
            flood_levels_per_id.append(df)

    train_df = pd.read_csv(str(train_dataset_path))
    new_df = train_df.set_index('id').join(pd.concat(flood_levels_per_id).set_index('id'))
    new_df.to_csv(output_train_csv)


if __name__ == "__main__":
    cli()
