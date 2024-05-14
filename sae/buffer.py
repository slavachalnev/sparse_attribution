from typing import Optional
import time
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformer_lens import HookedTransformer
from config import SAEConfig


class ActivationBuffer:
    def __init__(self, cfg: SAEConfig, model: HookedTransformer, filter: Optional[callable] = None):
        self.cfg: SAEConfig = cfg
        self.model: HookedTransformer = model
        self.dataset = load_dataset(cfg.dataset_path, split="train", streaming=True)
        self.dataset = self.dataset.with_format("torch")
        self.token_loader = iter(DataLoader(self.dataset, batch_size=cfg.store_batch_size))
        self.hook_point = self.cfg.hook_point.format(layer=self.cfg.hook_point_layer)
        self.buffer_size = cfg.n_batches_in_buffer * cfg.store_batch_size * cfg.context_size
        self.buffer = torch.zeros((self.buffer_size, self.cfg.d_in), dtype=torch.float32, device=cfg.device)
        self.grad_buffer = torch.zeros((self.buffer_size, self.cfg.d_in), dtype=torch.float32, device=cfg.device)
        self.batch_idx = 0  # index of the output batch
        self.fill_buffer()

    def fill_buffer(self):
        buffer_index = self.batch_idx * self.cfg.train_batch_size
        with tqdm(total=self.buffer_size, desc="Filling buffer") as pbar:
            pbar.update(buffer_index)
            while buffer_index < self.buffer_size:
                tokens = self.get_token_batch()
                ######
                # old_acts = self.model.run_with_cache(tokens,
                #                                 stop_at_layer=self.cfg.hook_point_layer + 1,
                #                                 names_filter=[self.hook_point],
                #                                 )[1][self.hook_point]
                # old_acts = old_acts.view(-1, self.cfg.d_in)
                ######

                acts, grads = self.run_and_cache(tokens)

                remaining_space = self.buffer_size - buffer_index
                n_to_add = min(acts.shape[0], remaining_space)
                self.buffer[buffer_index : buffer_index + n_to_add] = acts[:n_to_add]
                self.grad_buffer[buffer_index : buffer_index + n_to_add] = grads[:n_to_add]
                buffer_index += n_to_add
                pbar.update(n_to_add)
        
        perm = torch.randperm(self.buffer_size)
        self.buffer = self.buffer[perm]
        self.grad_buffer = self.grad_buffer[perm]
        self.batch_idx = 0

    def get_activations(self):
        if self.batch_idx * self.cfg.train_batch_size > self.buffer_size // 2:
            self.fill_buffer()  # this resets the batch_idx
        from_idx = self.batch_idx * self.cfg.train_batch_size
        to_idx = (self.batch_idx + 1) * self.cfg.train_batch_size
        activations = self.buffer[from_idx : to_idx]
        grads = self.grad_buffer[from_idx : to_idx]
        self.batch_idx += 1
        return activations, grads

    def get_token_batch(self):
        try:
            return next(self.token_loader)['tokens']
        except StopIteration:
            self.token_loader = iter(DataLoader(self.dataset, batch_size=self.cfg.store_batch_size))
            return next(self.token_loader)['tokens']
    
    def run_and_cache(self, tokens):
        acts = None
        grads = None

        def back_hook(input, hook):
            nonlocal grads
            grads = input

        def forward_hook(input, hook):
            nonlocal acts
            acts = input
        
        with self.model.hooks(fwd_hooks=[(self.hook_point, forward_hook)], bwd_hooks=[(self.hook_point, back_hook)]):
            loss = self.model(tokens, return_type='loss')
            loss.backward()
        
        acts = acts.detach().view(-1, self.cfg.d_in)
        grads = grads.detach().view(-1, self.cfg.d_in)

        return acts, grads


class ActivationDataset(Dataset):
    """Returns batches from memory-mapped numpy arrays for activations and gradients."""
    def __init__(self, acts_mmap, grads_mmap, batch_size):
        self.acts_mmap = acts_mmap
        self.grads_mmap = grads_mmap
        self.batch_size = batch_size

    def __len__(self):
        return len(self.acts_mmap) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = start_idx + self.batch_size
        acts_batch = self.acts_mmap[start_idx:end_idx]
        grads_batch = self.grads_mmap[start_idx:end_idx]
        return torch.from_numpy(acts_batch).to(torch.float32), torch.from_numpy(grads_batch).to(torch.float32)


class ActivationLoader:
    """
    Loads precomputed activations and gradients from disk.
    We assume that the stored activations and gradients are sufficiently shuffled.
    """
    def __init__(self, acts_path: str, grads_path: str, cfg: SAEConfig):
        acts_mmap = np.memmap(acts_path, dtype=np.float16, mode='r')
        grads_mmap = np.memmap(grads_path, dtype=np.float16, mode='r')
        
        total_rows = acts_mmap.shape[0] // cfg.d_in
        self.acts_mmap = np.memmap(acts_path, dtype=np.float16, mode='r', shape=(total_rows, cfg.d_in))
        self.grads_mmap = np.memmap(grads_path, dtype=np.float16, mode='r', shape=(total_rows, cfg.d_in))
        
        self.dataset = ActivationDataset(self.acts_mmap, self.grads_mmap, cfg.train_batch_size)
        self.dataloader = DataLoader(self.dataset, batch_size=1, num_workers=2, pin_memory=True)
        self.iterator = iter(self.dataloader)
    
    def get_activations(self):
        try:
            activations, gradients = next(self.iterator)
            # squeeze removes the extra dimension added by DataLoader
            activations = activations.squeeze(0)
            gradients = gradients.squeeze(0)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            activations, gradients = next(self.iterator)
            activations = activations.squeeze(0)
            gradients = gradients.squeeze(0)
        return activations, gradients