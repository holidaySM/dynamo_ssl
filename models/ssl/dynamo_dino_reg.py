import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator

from .dynamo import DynaMoSSL
from ..ema import EMA, copy_weights
from ..encoder.vision_transformer import trunc_normal_

accelerator = Accelerator()


class DINOLoss(nn.Module):
    def __init__(self, out_dim,
                 teacher_temp,
                 # warmup_teacher_temp,
                 # warmup_teacher_temp_epochs, num_epochs,
                 student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        # self.teacher_temp_schedule = np.concatenate((
        #     np.linspace(warmup_teacher_temp,
        #                 teacher_temp, warmup_teacher_temp_epochs),
        #     np.ones(num_epochs - warmup_teacher_temp_epochs) * teacher_temp
        # ))
        self.teacher_temp = teacher_temp

    def forward(self, student_output, teacher_output):
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
        # temp = self.teacher_temp_schedule[epoch]
        temp = self.teacher_temp
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


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nlayers = nlayers

        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

    @classmethod
    def create_copy_from(cls, src_model):
        src_device = next(src_model.parameters()).device
        model = DINOHead(in_dim=src_model.in_dim, out_dim=src_model.out_dim, nlayers=src_model.nlayers)
        model = copy_weights(src_model, model)
        model = model.to(src_device)
        return model


class DynaMoDinoSSL(DynaMoSSL):
    def __init__(self, teacher_temp, student_temp, center_momentum, dino_head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dino_loss = DINOLoss(out_dim=dino_head.out_dim,
                                  teacher_temp=teacher_temp,
                                  student_temp=student_temp,
                                  center_momentum=center_momentum).cuda()
        self.__dict__["dino_head"] = dino_head
        self.teacher_dino_head = EMA(DINOHead.create_copy_from(dino_head), self.ema_beta, copy=False)

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

        predicted_features = self.dino_head(obs_enc_pred)
        target_features = self.teacher_dino_head(obs_target)
        loss = self.dino_loss(predicted_features, target_features[:, 1:, j].detach())
        return loss

    def step(self):
        super().step()
        self.teacher_dino_head.step(self.dino_head)


class DynaMoHeadlessDinoSSL(DynaMoSSL):
    def __init__(self, teacher_temp, student_temp, center_momentum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dino_loss = DINOLoss(out_dim=self.forward_dynamics.config.output_dim,
                                  teacher_temp=teacher_temp,
                                  student_temp=student_temp,
                                  center_momentum=center_momentum).cuda()

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

        loss = self.dino_loss(obs_enc_pred, obs_target[:, 1:, j].detach())
        return loss
