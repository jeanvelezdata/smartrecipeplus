"""
DINO model components: projection head, student-teacher model, and loss.

References:
    Caron et al. (2021) — "Emerging Properties in Self-Supervised Vision Transformers"
    https://arxiv.org/abs/2104.14294
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """
    DINO projection head.

    Architecture:
        Linear(embed_dim → hidden_dim) → GELU
        Linear(hidden_dim → hidden_dim) → GELU
        Linear(hidden_dim → bottleneck_dim)          ← bottleneck
        WeightNorm Linear(bottleneck_dim → output_dim, bias=False)  ← last layer

    The last linear layer uses weight normalisation (L2-normalised weight rows)
    and does not receive gradients through its weight norm component —
    it is fixed as a normalised projection via torch.nn.utils.weight_norm.

    Args:
        embed_dim:       Input feature dimension from the backbone.
        hidden_dim:      Width of the hidden MLP layers.
        bottleneck_dim:  Output dimension of the bottleneck layer.
        output_dim:      Final projection dimension (e.g. 65536).
    """

    def __init__(self, embed_dim, hidden_dim, bottleneck_dim, output_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        # Last layer: no bias, weight-normalised
        last_layer = nn.Linear(bottleneck_dim, output_dim, bias=False)
        self.last_layer = nn.utils.weight_norm(last_layer)
        # Initialise the weight norm scale to 1 and freeze it
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, embed_dim]

        Returns:
            Tensor of shape [B, output_dim]
        """
        x = self.mlp(x)
        # L2-normalise the bottleneck features before the last layer
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOModel(nn.Module):
    """
    DINO student-teacher model.

    The teacher is an exponential moving average (EMA) of the student.
    Teacher parameters never receive gradients.

    Args:
        backbone_name:    Config string passed to get_backbone().
        pretrained:       Whether to load ImageNet weights for the backbone.
        proj_hidden_dim:  Hidden dim for DINOHead MLP.
        proj_bottleneck_dim: Bottleneck dim for DINOHead.
        proj_output_dim:  Output dim for DINOHead (e.g. 65536).
    """

    def __init__(
        self,
        backbone_name,
        pretrained=True,
        proj_hidden_dim=2048,
        proj_bottleneck_dim=256,
        proj_output_dim=65536,
    ):
        super().__init__()
        from models.backbones import get_backbone

        # Student
        self.student_backbone, embed_dim = get_backbone(backbone_name, pretrained)
        self.student_head = DINOHead(
            embed_dim, proj_hidden_dim, proj_bottleneck_dim, proj_output_dim
        )

        # Teacher — constructed independently, then initialised from student weights
        self.teacher_backbone, _ = get_backbone(backbone_name, pretrained)
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head = DINOHead(embed_dim, proj_hidden_dim, proj_bottleneck_dim, proj_output_dim)
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False

        # Center buffer
        self.register_buffer("center", torch.zeros(1, proj_output_dim))

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward_student(self, images_list):
        """
        Batch all crops into a single forward pass per spatial size.

        Global (224px) and local (96px) crops have different spatial dims, so
        they are grouped by size, each group is cat'd and run through the
        backbone+head in one shot, then the outputs are reassembled in the
        original crop order.
        """
        # Group crop indices by spatial resolution
        size_to_indices = {}
        for i, img in enumerate(images_list):
            size_to_indices.setdefault(img.shape[-1], []).append(i)

        outputs = [None] * len(images_list)
        for indices in size_to_indices.values():
            batch = torch.cat([images_list[i] for i in indices], dim=0)
            feats = self.student_backbone(batch)
            out = self.student_head(feats)
            for chunk, idx in zip(out.chunk(len(indices)), indices):
                outputs[idx] = chunk

        return outputs

    @torch.no_grad()
    def forward_teacher(self, global_images_list):
        """
        Batch the 2 global crops.
        """
        num_crops = len(global_images_list)

        all_imgs = torch.cat(global_images_list, dim=0)  # [2 * B, 3, H, W]

        feats = self.teacher_backbone(all_imgs)
        out = self.teacher_head(feats)

        return out.chunk(num_crops)

    # ------------------------------------------------------------------
    # EMA & center updates
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_teacher(self, momentum):
        """
        Exponential moving average update of teacher weights.

        teacher_param = momentum * teacher_param + (1 - momentum) * student_param

        Args:
            momentum: EMA coefficient (e.g. 0.996 → 1.0 over training).
        """
        student_params = (
            list(self.student_backbone.parameters())
            + list(self.student_head.parameters())
        )
        teacher_params = (
            list(self.teacher_backbone.parameters())
            + list(self.teacher_head.parameters())
        )
        for t_param, s_param in zip(teacher_params, student_params):
            t_param.data.mul_(momentum).add_((1.0 - momentum) * s_param.data)

    @torch.no_grad()
    def update_center(self, teacher_output, momentum):
        """
        Update the center buffer used for teacher output centering.

        center = momentum * center + (1 - momentum) * mean(teacher_output)

        Args:
            teacher_output: List of teacher output tensors, each [B, output_dim].
            momentum:       Center EMA coefficient (e.g. 0.9).
        """
        # Stack all teacher outputs and compute the global mean across batch
        all_outputs = torch.cat(teacher_output, dim=0)  # [num_global * B, output_dim]
        batch_center = all_outputs.mean(dim=0, keepdim=True)  # [1, output_dim]
        self.center = momentum * self.center + (1.0 - momentum) * batch_center


class DINOLoss(nn.Module):
    """
    DINO self-distillation loss.

    For each teacher global crop t and every student crop s where s ≠ t:
        loss += -softmax((t - center) / teacher_temp) · log_softmax(s / student_temp)

    The total loss is averaged over all valid (t, s) pairs.

    Args:
        teacher_temp:  Temperature for teacher softmax (e.g. 0.04).
        student_temp:  Temperature for student log-softmax (e.g. 0.1).
    """

    def __init__(self, teacher_temp=0.04, student_temp=0.1):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

    def forward(self, student_output, teacher_output, center):
        """
        Compute the DINO loss.

        Args:
            student_output: List of tensors [B, D] — all crops (global + local).
            teacher_output: List of tensors [B, D] — global crops only (length 2).
            center:         Center buffer tensor [1, D] from DINOModel.

        Returns:
            Scalar loss tensor.
        """
        # Teacher: centre → temperature scale → softmax
        teacher_probs = [
            F.softmax((t - center) / self.teacher_temp, dim=-1)
            for t in teacher_output
        ]

        # Student: temperature scale → log-softmax
        student_log_probs = [
            F.log_softmax(s / self.student_temp, dim=-1)
            for s in student_output
        ]

        total_loss = 0.0
        num_pairs = 0

        for t_idx, t_prob in enumerate(teacher_probs):
            for s_idx, s_log_prob in enumerate(student_log_probs):
                # Skip same-crop pairs 
                if s_idx == t_idx:
                    continue
                # Cross-entropy: -sum(p_teacher * log p_student), averaged over batch
                loss = -(t_prob * s_log_prob).sum(dim=-1).mean()
                total_loss += loss
                num_pairs += 1

        return total_loss / num_pairs
