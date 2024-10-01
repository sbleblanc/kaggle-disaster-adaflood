import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer


class FloodingTrainer(Trainer):

    def __init__(self, b: float = 0.03, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.b = b
        self.loss_fct = CrossEntropyLoss(reduction='none')

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        if "flood" in inputs:
            flood_levels = inputs.pop("flood")
        else:
            flood_levels = torch.full(labels.shape, self.b).to(model.device)

        outputs = model(**inputs)
        logits = outputs.get('logits')
        losses = self.loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
        loss = ((losses - flood_levels).abs() + flood_levels).mean()
        return (loss, outputs) if return_outputs else loss
