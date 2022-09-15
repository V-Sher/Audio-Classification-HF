# Lint as: python3
"""CREMA-D dataset."""

import os
from typing import Union

import datasets
import pandas as pd

_DESCRIPTION = """\
    CREMA-D is a data set of 7,442 original clips from 91 actors.
    These clips were from 48 male and 43 female actors between the ages of 20 and 74
    coming from a variety of races and ethnicities (African America, Asian,
    Caucasian, Hispanic, and Unspecified). Actors spoke from a selection of 12
    sentences. The sentences were presented using one of six different emotions
    (Anger, Disgust, Fear, Happy, Neutral, and Sad) and four different emotion
    levels (Low, Medium, High, and Unspecified).
"""

_HOMEPAGE = "https://github.com/CheyneyComputerScience/CREMA-D"

DATA_DIR = {"train": "AudioWAV"}


class Crema(datasets.GeneratorBasedBuilder):
    """Crema-D dataset."""

    DEFAULT_WRITER_BATCH_SIZE = 256
    BUILDER_CONFIGS = [datasets.BuilderConfig(name="clean", description="Train Set.")]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {"file": datasets.Value("string"), "label": datasets.Value("string")}
            ),
            supervised_keys=("file", "label"),
            homepage=_HOMEPAGE,
        )

    def _split_generators(
        self, dl_manager: datasets.utils.download_manager.DownloadManager
    ):
        data_dir = dl_manager.extract(self.config.data_dir)
        if self.config.name == "clean":
            train_splits = [
                datasets.SplitGenerator(
                    name="train", gen_kwargs={"files": data_dir, "name": "train"}
                )
            ]

        return train_splits

    def _generate_examples(self, files: Union[str, os.PathLike], name: str):
        """Generate examples from a Crema unzipped directory."""
        key = 0
        examples = list()

        audio_dir = os.path.join(files, DATA_DIR[name])

        if not os.path.exists(audio_dir):
            raise FileNotFoundError
        else:
            for file in os.listdir(audio_dir):
                res = dict()
                res["file"] = "{}".format(os.path.join(audio_dir, file))
                res["label"] = file.split("_")[-2]
                examples.append(res)

        for example in examples:
            yield key, {**example}
            key += 1
        examples = []
