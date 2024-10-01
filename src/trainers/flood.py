from torch.nn import CrossEntropyLoss
from transformers import Trainer


class FloodingTrainer(Trainer):

    def __init__(self, b: float = 0.03, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.b = b
        self.loss_fct = CrossEntropyLoss(reduction='none')

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        losses = self.loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
        loss = ((losses - self.b).abs() + self.b).mean()
        return (loss, outputs) if return_outputs else loss