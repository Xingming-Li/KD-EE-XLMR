import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutput

class XLMRWithEarlyExit(PreTrainedModel):
    def __init__(self, model_name_or_path, num_labels, is_token_classification=False, exit_layers=[4, 8], threshold=0.5):
        config = AutoConfig.from_pretrained(model_name_or_path, output_hidden_states=True)
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.config = config

        self.num_labels = num_labels
        self.exit_layers = exit_layers
        self.threshold = threshold
        self.is_token_classification = is_token_classification  # True for NER (WikiANN), False for classification (XNLI)

        hidden_size = config.hidden_size

        # Exit heads: one linear layer per exit layer
        self.exit_heads = nn.ModuleDict({
            str(layer): nn.Linear(hidden_size, num_labels) for layer in exit_layers
        })

    def compute_entropy(self, logits):
        """Compute entropy of the output logits."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # List of all hidden layers

        final_logits = None
        exit_layer = None
        loss = None

        # For auxiliary training
        all_logits = []
        all_losses = []

        for layer in self.exit_layers:
            hidden = hidden_states[layer]  # (batch_size, seq_len, hidden_size)

            if self.is_token_classification:
                logits = self.exit_heads[str(layer)](hidden)  # (batch_size, seq_len, num_labels)
                mask = attention_mask.unsqueeze(-1).expand(-1, -1, self.num_labels)
                logits = logits.masked_fill(mask == 0, -1e9)
                entropy = self.compute_entropy(logits).mean(dim=1)  # (batch_size,)
            else:
                cls_hidden = hidden[:, 0, :]
                logits = self.exit_heads[str(layer)](cls_hidden)  # (batch_size, num_labels)
                entropy = self.compute_entropy(logits)  # (batch_size,)

            all_logits.append(logits)

            # Exit decision logic
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                if self.is_token_classification:
                    layer_loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
                else:
                    layer_loss = loss_fn(logits, labels)
                all_losses.append(layer_loss)

        # If no early exit was taken
        if final_logits is None:
            final_logits = all_logits[-1]
            exit_layer = self.exit_layers[-1]

        # Use auxiliary loss: sum over all heads
        if labels is not None:
            loss = sum(all_losses) / len(all_losses)

        output_cls = TokenClassifierOutput if self.is_token_classification else SequenceClassifierOutput
        return output_cls(
            loss=loss,
            logits=final_logits,
            hidden_states=None,
            attentions=None,
        )
