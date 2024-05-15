from transformer_lens import HookedTransformer
import torch
from datasets import load_dataset
from tqdm import tqdm

from config import SAEConfig
from model import SparseAutoencoder


@torch.no_grad()
def reconstruction_ce(saes: list[SparseAutoencoder], model: HookedTransformer, dataset):
    """Computes cross-entropy loss when residual stream is replaced by SAE output"""

    hook_name = saes[0].cfg.hook_point.format(layer=saes[0].cfg.hook_point_layer)

    ce_losses = []
    # for sae in saes:
    for sae in tqdm(saes):
        def patching_hook(resid, hook):
            # resid shape is (batch_size, seq_len, d_in)
            resid_shape = resid.shape
            resid = resid.reshape(-1, resid.shape[-1])
            sae_out, _ = sae._forward(resid)
            sae_out = sae_out.reshape(*resid_shape)
            return sae_out
        
        total_loss = 0
        for i, batch in enumerate(dataset):
            batch = batch['tokens'].to(saes[0].device)
            loss = model.run_with_hooks(batch, fwd_hooks=[(hook_name, patching_hook)], return_type="loss")
            total_loss += loss.mean().item()
            if i > 100:
                break
        
        ce_losses.append(total_loss / i)
    
    return ce_losses

            

