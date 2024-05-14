import os
import numpy as np
from tqdm import tqdm
import torch
from transformer_lens import HookedTransformer
from config import SAEConfig
from buffer import ActivationBuffer

def compute_and_save_activations_and_gradients(cfg: SAEConfig, output_dir: str, max_batches: int):
    model = HookedTransformer.from_pretrained(cfg.model_name) 
    buffer = ActivationBuffer(cfg, model)
    
    total_rows = max_batches * cfg.train_batch_size

    acts_output_file = os.path.join(output_dir, f"activations_{total_rows}.npy")
    grads_output_file = os.path.join(output_dir, f"gradients_{total_rows}.npy")

    acts_mmap = np.memmap(acts_output_file, dtype=np.float16, mode='w+', shape=(total_rows, cfg.d_in))
    grads_mmap = np.memmap(grads_output_file, dtype=np.float16, mode='w+', shape=(total_rows, cfg.d_in))
    
    row_idx = 0
    for i in tqdm(range(max_batches)):
        activations, gradients = buffer.get_activations()
        activations = activations.cpu().numpy().astype(np.float16)
        gradients = gradients.cpu().numpy().astype(np.float16)
        acts_mmap[row_idx : row_idx + activations.shape[0]] = activations
        grads_mmap[row_idx : row_idx + gradients.shape[0]] = gradients
        row_idx += activations.shape[0]

if __name__ == "__main__":
    cfg = SAEConfig(
        model_name="gelu-4l",
        hook_point_layer=2,
        device="cuda",
        store_batch_size=8,
        n_batches_in_buffer=200,
        train_batch_size=4096,
    )

    output_dir = "/mnt/ssd/activation_cache/NeelNanda/c4-code-tokenized-2b/gelu-4l"
    os.makedirs(output_dir, exist_ok=True)

    compute_and_save_activations_and_gradients(cfg, output_dir, max_batches=40000)