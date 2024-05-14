import os
import json
from typing import Optional

from dataclasses import dataclass

import einops
import torch
from torch import nn
from transformer_lens.hook_points import HookedRootModule, HookPoint

from config import SAEConfig


@dataclass
class SAEOutput:
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    loss: torch.Tensor
    mse_loss: torch.Tensor
    l1_loss: torch.Tensor
    l1_attr: Optional[torch.Tensor]
    reconstr_attr: Optional[torch.Tensor]


class SparseAutoencoder(HookedRootModule):

    def __init__(
        self,
        cfg: SAEConfig,
        init_weights: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        self.d_sae = cfg.d_sae
        self.l1_coefficient = cfg.l1_coefficient
        self.lp_norm = cfg.lp_norm
        self.dtype = torch.float32
        # self.dtype = torch.float16 if cfg.half else torch.float32
        self.device = cfg.device

        # Sparse Attribution
        self.attrib_sparsity_coeff = cfg.attrib_sparsity_coeff
        self.unexplained_attrib_coeff = cfg.unexplained_attrib_coeff

        self.hook_sae_in = HookPoint()
        self.hook_hidden_pre = HookPoint()
        self.hook_hidden_post = HookPoint()
        self.hook_sae_out = HookPoint()

        if init_weights:
            self.W_enc = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(self.d_in, self.d_sae, dtype=self.dtype, device=self.device)
                )
            )
            self.b_enc = nn.Parameter(
                torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
            )
            self.W_dec = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(self.d_sae, self.d_in, dtype=self.dtype, device=self.device)
                )
            )
            with torch.no_grad():
                # Anthropic normalize this to have unit columns
                self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
            self.b_dec = nn.Parameter(
                torch.zeros(self.d_in, dtype=self.dtype, device=self.device)
            )

            self.setup()  # Required for `HookedRootModule`
    
    def _forward(self, x: torch.Tensor):
        # move x to correct dtype
        x = x.to(self.dtype)
        sae_in = self.hook_sae_in(
            x - self.b_dec
        )  # Remove decoder bias as per Anthropic

        hidden_pre = self.hook_hidden_pre(
            einops.einsum(
                sae_in,
                self.W_enc,
                "... d_in, d_in d_sae -> ... d_sae",
            )
            + self.b_enc
        )
        feature_acts = self.hook_hidden_post(torch.nn.functional.relu(hidden_pre))

        sae_out = self.hook_sae_out(
            einops.einsum(
                feature_acts,
                self.W_dec,
                "... d_sae, d_sae d_in -> ... d_in",
            )
            + self.b_dec
        )
        return sae_out, feature_acts

    def forward(
            self,
            x: torch.Tensor,
            l1_factor: float = 1.0,
            grad_wrt_x: Optional[torch.Tensor] = None,
        ):
        sae_out, feature_acts = self._forward(x)

        # add config for whether l2 is normalized:
        x_centred = x - x.mean(dim=0, keepdim=True)
        mse_loss = (
            torch.pow((sae_out - x.float()), 2)
            / (x_centred**2).sum(dim=-1, keepdim=True).sqrt()
        )

        mse_loss = mse_loss.mean()

        sparsity = feature_acts.norm(p=self.lp_norm, dim=1)
        sparsity = sparsity.mean(dim=(0,))
        l1_loss = self.l1_coefficient * l1_factor * sparsity

        if self.cfg.norm_grad:
            grad_wrt_x = grad_wrt_x / grad_wrt_x.norm(p=2, dim=-1, keepdim=True)

        grad_y = grad_wrt_x @ self.W_dec.T
        l1_attr = (grad_y * feature_acts).abs().sum(dim=-1).mean()

        e = x - sae_out
        if self.cfg.unexplained_attrib_method == "anthropic":
            reconstr_attr = einops.einsum(e, grad_wrt_x,
                                        "batch d, batch d -> batch").abs().mean()
        elif self.cfg.unexplained_attrib_method == "l2":
            reconstr_attr = torch.norm(e * grad_wrt_x, dim=-1).pow(2).mean()
        else:
            raise ValueError(f"Invalid attribution method: {self.cfg.unexplained_attrib_method}")
        
        l1_attr = self.attrib_sparsity_coeff * l1_attr
        reconstr_attr = self.unexplained_attrib_coeff * reconstr_attr

        mse_loss = self.cfg.mse_coefficient * mse_loss
        
        loss = mse_loss + l1_loss + l1_attr + reconstr_attr

        return SAEOutput(
            sae_out=sae_out,
            feature_acts=feature_acts,
            loss=loss,
            mse_loss=mse_loss,
            l1_loss=l1_loss,
            l1_attr=l1_attr,
            reconstr_attr=reconstr_attr,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        """

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )

        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )

    def save_model(self, dir_path: str):
        """
        Save the model's state_dict and the config used to train it.
        """
        os.makedirs(dir_path, exist_ok=True)
        # Save the model state_dict
        torch.save(self.state_dict(), os.path.join(dir_path, "model.pt"))

        # Save the config as a JSON file
        cfg_dict = self.cfg.to_dict()
        with open(os.path.join(dir_path, "cfg.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)


    @classmethod
    def load_from_pretrained(cls, dir_path: str):
        """
        Load function for the model. Loads the model's state_dict and the config used to train it.
        This method can be called directly on the class, without needing an instance.
        """
        model_path = os.path.join(dir_path, "model.pt")
        cfg_path = os.path.join(dir_path, "cfg.json")

        with open(cfg_path, "r") as f:
            cfg_d = json.load(f)

        valid_keys = SAEConfig.__annotations__.keys()
        valid_cfg_d = {key: cfg_d.pop(key) for key in list(cfg_d.keys()) if key in valid_keys}

        cfg = SAEConfig(**valid_cfg_d)

        ignored_keys = list(cfg_d.keys())
        if ignored_keys:
            print(f"ignored keys: {ignored_keys}")

        state_dict = torch.load(model_path, map_location="cpu")
        cfg.device = "cpu"
        cfg.from_pretrained_path = dir_path

        # Create an instance of the class using the loaded configuration
        instance = cls(cfg)
        instance.load_state_dict(state_dict)

        return instance
