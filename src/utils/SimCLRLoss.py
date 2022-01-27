import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def __call__(self, proj_features_combined: torch.Tensor, *args) -> torch.Tensor:

        """
        custom_simclr_contrastive_loss(proj_feat1, proj_feat2)
        Returns contrastive loss, given sets of projected features, with positive
        pairs matched along the batch dimension.
        Required args:
        - proj_features_combined (2D torch Tensor): projected features for both images
            augmentations (2*batch_size x feat_size)
        Returns:
        - loss (float): mean contrastive loss
        """

        device = proj_features_combined.device

        batch_size = int(len(proj_features_combined) / 2)  # N
        proj_features = F.normalize(proj_features_combined, dim=1)

        similarity_matrix = F.cosine_similarity(
            proj_features.unsqueeze(1), proj_features.unsqueeze(0), dim=2
        )  # dim: 2N x 2N

        # initialize arrays to identify sets of positive and negative examples, of
        # shape (batch_size * 2, batch_size * 2), and where
        # 0 indicates that 2 images are NOT a pair (either positive or negative, depending on the indicator type)
        # 1 indices that 2 images ARE a pair (either positive or negative, depending on the indicator type)
        pos_sample_indicators = torch.roll(torch.eye(2 * batch_size), batch_size, 1).to(device)

        # have all values but the diagonal components
        all_sample_indicators = (torch.ones(2 * batch_size) - torch.eye(2 * batch_size)).to(device)

        # Calculate the numerator of the Loss expression by selecting the appropriate elements from similarity_matrix.
        # Use the pos_sample_indicators tensor
        numerator = torch.exp(similarity_matrix / self.temperature)[pos_sample_indicators.bool()]

        # Calculate the denominator of the Loss expression by selecting the appropriate elements from similarity_matrix,
        # and summing over pairs for each item.
        # Use the neg_sample_indicators tensor
        denominator = torch.sum(torch.exp(similarity_matrix / self.temperature) * all_sample_indicators, dim=1)

        if (denominator < 1e-8).any():  # clamp to avoid division by 0
            denominator = torch.clamp(denominator, 1e-8)

        loss = torch.mean(-torch.log(numerator / denominator))

        return loss
