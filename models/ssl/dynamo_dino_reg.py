from typing import Tuple, Dict, Optional

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator

from .dynamo import DynaMoSSL
from ..ema import EMA
from ..transformer_encoder import TransformerEncoder, TransformerEncoderConfig

accelerator = Accelerator()


class DINOLoss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        assert student_output.size() == teacher_output.size()
        assert len(student_output.size()) == 3

        student_output = einops.rearrange(student_output, "b t e -> (b t) e")
        teacher_output = einops.rearrange(teacher_output, "b t e -> (b t) e")

        student_out = student_output / self.student_temp
        student_log_dists = F.log_softmax(student_out, dim=-1)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_dists = F.softmax((teacher_output - self.center) / temp, dim=-1).detach()

        total_loss = torch.sum(-teacher_dists * student_log_dists, dim=-1).mean()
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0)
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DynaMoDinoSSL(DynaMoSSL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _forward_dyn_loss_one_pair(
            self,
            obs_enc: torch.Tensor,
            obs_proj: torch.Tensor,
            obs_target: torch.Tensor,
            i: int,
            j: int,
    ):
        forward_dyn_input = torch.cat([obs_enc[:, :-1, j], obs_proj[:, 1:, i]], dim=-1)
        obs_enc_pred = self.forward_dynamics(forward_dyn_input)  # (N, T-1, E)
        loss = (
                1
                - torch.nn.functional.cosine_similarity(
            obs_enc_pred, obs_target[:, 1:, j].detach(), dim=-1
        ).mean()
        )
        return loss
