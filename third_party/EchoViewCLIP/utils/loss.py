import torch
import torch.nn as nn

class TextSemanticOppositeLoss(nn.Module):
    def __init__(self, mode="L2"):
        super(TextSemanticOppositeLoss, self).__init__()
        assert mode in ["L2", "cosine"], "mode must be 'L2' or 'cosine'"
        self.mode = mode

    def forward(self, all_text_features, all_text_features_no):
        if self.mode == "L2":
            l2_distance = 2 - 2 * (all_text_features * all_text_features_no).sum(dim=-1) + 1e-4
            loss = 2 - torch.sqrt(l2_distance) 

        elif self.mode == "cosine":
            loss = (all_text_features * all_text_features_no).sum(dim=-1) + 1.0 

        return loss.mean()
