import os
import logging
import librosa

import wandb
import numpy as np

from datasets import DatasetDict, load_dataset, load_metric
from transformers import (
    HubertForSequenceClassification,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
)
from utils import collator

logging.basicConfig(
    format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
NUM_LABELS = 6


USER = "XXXX" # TODO: replace with your username
WANDB_PROJECT = "XXXXX" # TODO: replace with your project name
wandb.init(entity=USER, project=WANDB_PROJECT)


# PROCESS THE DATASET TO THE FORMAT EXPECTED BY THE MODEL FOR TRAINING
PreTrainedFeatureExtractor = "SequenceFeatureExtractor"  # noqa: F821

INPUT_FIELD = "input_values"
LABEL_FIELD = "labels"


def prepare_dataset(batch, feature_extractor: PreTrainedFeatureExtractor):
    audio_arr = batch["array"]
    input = feature_extractor(
        audio_arr, sampling_rate=16000, padding=True, return_tensors="pt"
    )

    batch[INPUT_FIELD] = input.input_values[0]
    batch[LABEL_FIELD] = batch[
        "label"
    ]  # colname MUST be labels as Trainer will look for it by default

    return batch


model_id = "facebook/hubert-base-ls960"
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

extractor_path = (
    model_id
    if len(os.listdir(MODELS_DIR)) == 0
    else os.path.join(MODELS_DIR, "feature_extractor")
)
model_path = (
    model_id
    if len(os.listdir(MODELS_DIR)) == 0
    else os.path.join(MODELS_DIR, "pretrained_model")
)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(extractor_path)

config = PretrainedConfig.from_pretrained(model_path, num_labels=NUM_LABELS)
hubert_model = HubertForSequenceClassification.from_pretrained(
    model_path,
    config=config,  # because we need to update num_labels as per our dataset
    ignore_mismatched_sizes=True,  # to avoid classifier size mismatch from from_pretrained.
)


# FREEZE LAYERS

# freeze all layers to begin with
for param in hubert_model.parameters():
    param.requires_grad = False

layers_freeze_num = 2
n_layers = (
    4 + layers_freeze_num * 16
)  # 4 refers to projector and classifier's weights and biases.
for name, param in list(hubert_model.named_parameters())[-n_layers:]:
    param.requires_grad = True

# # freeze model weights for all layers except projector and classifier
# for name, param in hubert_model.named_parameters():
#     if any(ext in name for ext in ["projector", "classifier"]):
#         param.requires_grad = True


trainer_config = {
    "OUTPUT_DIR": "results",
    "TRAIN_EPOCHS": 5,
    "TRAIN_BATCH_SIZE": 32,
    "EVAL_BATCH_SIZE": 32,
    "GRADIENT_ACCUMULATION_STEPS": 4,
    "WARMUP_STEPS": 500,
    "DECAY": 0.01,
    "LOGGING_STEPS": 10,
    "MODEL_DIR": "models/audio-model",
    "LR": 1e-3,
}


dataset_config = {
    "LOADING_SCRIPT_FILES": os.path.join(PROJECT_ROOT, "src/data/crema.py"),
    "CONFIG_NAME": "clean",
    "DATA_DIR": os.path.join(PROJECT_ROOT, "data/archive.zip"),
    "CACHE_DIR": os.path.join(PROJECT_ROOT, "cache_crema"),
}


ds = load_dataset(
    dataset_config["LOADING_SCRIPT_FILES"],
    dataset_config["CONFIG_NAME"],
    cache_dir=dataset_config["CACHE_DIR"],
    data_dir=dataset_config["DATA_DIR"],
)


# CONVERING RAW AUDIO TO ARRAYS
ds = ds.map(
    lambda x: {"array": librosa.load(x["file"], sr=16000, mono=False)[0]},
    num_proc=2,
)


# LABEL TO ID
ds = ds.class_encode_column("label")


# ds["train"] = ds["train"].select(range(2500))
wandb.log({"dataset_size": len(ds["train"])})


# APPLY THE DATA PREP USING FEATURE EXTRACTOR TO ALL EXAMPLES
ds = ds.map(
    prepare_dataset,
    fn_kwargs={"feature_extractor": feature_extractor},
    # num_proc=4,
)
logging.info("Finished extracting features from audio arrays.")


# INTRODUCE TRAIN TEST VAL SPLITS

# 90% train, 10% test + validation
train_testvalid = ds["train"].train_test_split(shuffle=True, test_size=0.1)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
ds = DatasetDict(
    {
        "train": train_testvalid["train"],
        "test": test_valid["test"],
        "val": test_valid["train"],
    }
)


# DEFINE DATA COLLATOR - TO PAD TRAINING BATCHES DYNAMICALLY
data_collator = collator.DataCollatorCTCWithPadding(
    processor=feature_extractor, padding=True
)


# Fine-Tuning with Trainer
training_args = TrainingArguments(
    output_dir=os.path.join(
        PROJECT_ROOT, trainer_config["OUTPUT_DIR"]
    ),  # output directory
    gradient_accumulation_steps=trainer_config[
        "GRADIENT_ACCUMULATION_STEPS"
    ],  # accumulate the gradients before running optimization step
    num_train_epochs=trainer_config["TRAIN_EPOCHS"],  # total number of training epochs
    per_device_train_batch_size=trainer_config[
        "TRAIN_BATCH_SIZE"
    ],  # batch size per device during training
    per_device_eval_batch_size=trainer_config[
        "EVAL_BATCH_SIZE"
    ],  # batch size for evaluation
    warmup_steps=trainer_config[
        "WARMUP_STEPS"
    ],  # number of warmup steps for learning rate scheduler
    weight_decay=trainer_config["DECAY"],  # strength of weight decay
    logging_steps=trainer_config["LOGGING_STEPS"],
    evaluation_strategy="epoch",  # report metric at end of each epoch
    report_to="wandb",  # enable logging to W&B
    learning_rate=trainer_config["LR"],  # default = 5e-5
)


def compute_metrics(eval_pred):
    # DEFINE EVALUATION METRIC
    compute_accuracy_metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return compute_accuracy_metric.compute(predictions=predictions, references=labels)


# START TRAINING
trainer = Trainer(
    model=hubert_model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    data_collator=data_collator,
    train_dataset=ds["train"],  # training dataset
    eval_dataset=ds["val"],  # evaluation dataset
    compute_metrics=compute_metrics,
)


trainer.train()

# TO RESUME TRAINING FROM CHECKPOINT
# trainer.train("results/checkpoint-2000")

# VALIDATION SET RESULTS
logging.info("Eval Set Result: {}".format(trainer.evaluate()))

# TEST RESULTS
test_results = trainer.predict(ds["test"])
logging.info("Test Set Result: {}".format(test_results.metrics))
wandb.log({"test_accuracy": test_results.metrics["test_accuracy"]})

trainer.save_model(os.path.join(PROJECT_ROOT, trainer_config["MODEL_DIR"]))

# logging trained models to wandb
wandb.save(
    os.path.join(PROJECT_ROOT, trainer_config["MODEL_DIR"], "*"),
    base_path=os.path.dirname(trainer_config["MODEL_DIR"]),
    policy="end",
)
