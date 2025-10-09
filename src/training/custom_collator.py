
import torch
from dataclasses import dataclass
from typing import Dict, List, Union

@dataclass
class CustomDataCollatorForCTC:
    """
    A custom data collator for CTC tasks that handles padding for both input features and labels.
    This is a simplified, standalone replacement for the transformers.DataCollatorForCTC
    to bypass import issues.
    """
    processor: any
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split features into input_features and labels.
        # The features are lists of dicts from the dataset.
        input_features_list = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad the input features (audio).
        # The processor's feature extractor handles the padding of input_features.
        batch = self.processor.feature_extractor.pad(
            input_features_list,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad the labels (text).
        # The processor's tokenizer handles the padding of labels.
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Replace the tokenizer's pad_token_id with -100 for the labels,
        # as this is the ignore index for CTC loss.
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch
