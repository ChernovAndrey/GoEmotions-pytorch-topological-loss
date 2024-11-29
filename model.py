import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
import pandas as pd


def get_distance_matrix(file_path_correlations, label2id):
    distance = 1. - pd.read_csv(file_path_correlations, index_col=0)
    distance_indexes = distance.rename(index=label2id, columns=label2id)
    return distance_indexes, distance


class WassersteinLoss(nn.Module):
    def __init__(self, dist_matrix: torch.tensor, agg_type: str,
                 weight=None):  # agg_type in ('min', 'max', 'mean')
        super(WassersteinLoss, self).__init__()
        self.weight = weight
        self.dist_matrix = dist_matrix
        self.agg_type = agg_type
        print(f'agg type: {self.agg_type}')

    def forward(self, logits, targets):
        # Apply sigmoid activation to logits
        # predictions = torch.sigmoid(logits)
        self.dist_matrix = self.dist_matrix.to(targets.device)
        predictions = torch.softmax(logits, dim=-1)

        # Compute binary cross-entropy

        if self.agg_type == 'min':
            # Create a mask of the same shape as targets, where True indicates non-zero elements
            # Create a mask of the same shape as targets, where True indicates non-zero elements
            # Create a mask of the same shape as targets, where True indicates non-zero elements
            mask = targets != 0  # Shape [batch_size, n_labels]

            # Expand the mask to gather relevant rows from the distance matrix
            # This step creates a 3D tensor of shape [batch_size, n_labels, n_labels]
            gathered_distances = self.dist_matrix.unsqueeze(0).expand(targets.size(0), -1, -1)
            gathered_distances = gathered_distances.masked_fill(~mask.unsqueeze(-1), float('inf'))

            # Compute the minimum along the last dimension (corresponding to n_labels in dist_matrix)
            cost, _ = gathered_distances.min(dim=-2)

            # cost, _ = self.dist_matrix[torch.nonzero(targets).reshape(-1)].min(axis=0)
        elif self.agg_type == 'max':
            # Create a mask of the same shape as targets, where True indicates non-zero elements
            mask = targets != 0  # Shape [batch_size, n_labels]

            # Expand the mask to gather relevant rows from the distance matrix
            # This step creates a 3D tensor of shape [batch_size, n_labels, n_labels]
            gathered_distances = self.dist_matrix.unsqueeze(0).expand(targets.size(0), -1, -1)
            gathered_distances = gathered_distances.masked_fill(~mask.unsqueeze(-1), float('-inf'))

            # Compute the minimum along the last dimension (corresponding to n_labels in dist_matrix)
            cost, _ = gathered_distances.max(dim=-2)

            # cost, _ = self.dist_matrix[torch.nonzero(targets).reshape(-1)].max(axis=0)
        elif self.agg_type == 'mean':
            # Assuming self.dist_matrix is a 2D tensor of shape [n_labels, n_labels]
            # and targets is a 2D tensor of shape [batch_size, n_labels]

            # Create a mask of the same shape as targets, where True indicates non-zero elements
            mask = targets != 0  # Shape [batch_size, n_labels]

            # Expand the mask to gather relevant rows from the distance matrix
            # This step creates a 3D tensor of shape [batch_size, n_labels, n_labels]
            gathered_distances = self.dist_matrix.unsqueeze(0).expand(targets.size(0), -1, -1)
            gathered_distances = gathered_distances.masked_fill(~mask.unsqueeze(-1), float('nan'))

            # Compute the mean along the last dimension, ignoring NaN
            cost = torch.nanmean(gathered_distances, dim=-2)  # Mean

            # cost_mean will have shape [batch_size, n_labels]
            # cost = self.dist_matrix[torch.nonzero(targets).reshape(-1)].mean(axis=0)
        else:
            assert False, f"aggregation type: {self.agg_type} is not supported"
        loss = torch.sum(predictions * cost, dim=-1)

        # Apply weights if provided
        if self.weight is not None:
            loss *= self.weight

        return loss.mean()


class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config, use_wasserstein_loss=True):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.use_wasserstein_loss = use_wasserstein_loss
        if self.use_wasserstein_loss:
            dist, _ = get_distance_matrix('correlation_train.csv', config.label2id)
            self.loss_fct = WassersteinLoss(torch.tensor(dist.values), 'max')
        else:
            self.loss_fct = nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
