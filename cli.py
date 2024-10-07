import click
import mlflow
import evaluate
import torch
import tqdm
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_dataset
from mlflow.transformers import log_model, load_model
from sklearn.model_selection import KFold, StratifiedKFold
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
@click.option("-f", "--num-folds", type=click.INT, default=5)
def train(
        seed: int,
        model_name: str,
        run_name: str,
        train_dataset_path: Path,
        output_dir: Path,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        weight_decay: float,
        num_folds: int
):
    set_seed(seed)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.enable_system_metrics_logging()

    tok = AutoTokenizer.from_pretrained(model_name)

    ds = load_dataset("csv", data_files=str(train_dataset_path), split='train')
    ds = ds.remove_columns(["id", "keyword", "location"]).rename_column("target", "labels")
    ds = ds.map(tok, batched=True, input_columns="text", remove_columns="text")
    # ds = ds['train'].train_test_split(test_size=0.2, shuffle=True)

    skfold = StratifiedKFold(n_splits=num_folds, shuffle=True)

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
    with mlflow.start_run(run_name=run_name) as parent_run:

        mlflow.log_param("num_folds", num_folds)
        mlflow.log_param("input_dataset", str(train_dataset_path))
        # mlf_ds = mlflow.data.from_pandas(ds.to_pandas(), source=train_dataset_path, targets='labels')
        # mlflow.log_input(mlf_ds, "training")

        for i, (train_idx, eval_idx) in enumerate(skfold.split(ds, ds['labels'])):
            train_ds = ds.select(train_idx)
            eval_ds = ds.select(eval_idx)

            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

            trainer = FloodingTrainer(
                b=0.0,
                model=model,
                args=train_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=tok,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            with mlflow.start_run(run_name=f"{run_name}_f{i}", parent_run_id=parent_run.info.run_id, nested=True):
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
def adaflood_hp_optim(
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
    ds = ds.train_test_split(test_size=0.2, shuffle=True)
    data_collator = DataCollatorWithPadding(tokenizer=tok)

    def objective(trial: optuna.Trial):
        num_epochs = trial.suggest_int("num_epochs", 1, 5)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2)
        schedule = trial.suggest_categorical("schedule", ["linear", "constant", "constant_with_warmup"])

        if schedule != "constant":
            num_warmup_steps = trial.suggest_int("num_warmup_steps", 0, 150)
        else:
            num_warmup_steps = 0

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
            warmup_steps=num_warmup_steps,
            lr_scheduler_type=schedule
        )
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=ds['train'],
            eval_dataset=ds['eval'],
            tokenizer=tok,
            data_collator=data_collator
        )
        trainer.train()
        trainer.evaluate()
        model.eval()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)


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


@cli.command()
@click.option("-i", "--input-csv", type=click.Path(dir_okay=False), required=True)
@click.option("-c", "--text-col", type=click.STRING, required=True)
@click.option("-m", "--model-uri", type=click.STRING, required=True)
@click.option("-o", "--output-csv", type=click.Path(dir_okay=False), default="submission.csv")
def infer(
        input_csv: str,
        text_col: str,
        model_uri: str,
        output_csv: str
):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    test_df = pd.read_csv(input_csv)
    model = load_model(model_uri, device="cuda:0")
    inferred = model(test_df[text_col].tolist())
    test_df['target'] = [
        1 if res['label'] == "LABEL_1" else 0
        for res in inferred
    ]
    test_df.loc[:, ["id", "target"]].to_csv(output_csv, index=False)



if __name__ == "__main__":
    cli()
