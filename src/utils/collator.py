from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from transformers import Wav2Vec2Processor

INPUT_FIELD = "input_values"
LABEL_FIELD = "labels"


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, examples: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        input_features = [
            {INPUT_FIELD: example[INPUT_FIELD]} for example in examples
        ]  # example is basically row0, row1, etc...
        labels = [example[LABEL_FIELD] for example in examples]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch[LABEL_FIELD] = torch.tensor(labels)

        return batch
