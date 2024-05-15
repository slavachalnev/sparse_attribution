import os
import time
import json
import wandb
import torch
import multiprocessing
from transformer_lens import HookedTransformer
from datasets import load_dataset
from config import SAEConfig
from model import SparseAutoencoder, SAEOutput
from buffer import ActivationLoader
from utils import reconstruction_ce

def wandb_logger(queue, config_dict):
    wandb_run = wandb.init(project=config_dict["wandb_project"], name=config_dict["run_name"], config=config_dict)
    while True:
        log_data = queue.get()
        if log_data is None:  # Sentinel value to terminate the process
            break
        wandb_run.log(log_data)
    wandb_run.finish()

def train_multiple_saes(configs: list[SAEConfig]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saes = []
    optimizers = []
    lr_schedulers = []
    wandb_queues = []
    wandb_processes = []

    # for cross-entropy calculation
    model = HookedTransformer.from_pretrained(configs[0].model_name)
    # test_dataset = load_dataset(f"{configs[0].dataset_path}[:1%]")
    # test_dataset = load_dataset(configs[0].dataset_path)
    dataset = load_dataset(configs[0].dataset_path, split="train", streaming=True)
    dataset = dataset.with_format("torch")

    for cfg in configs:
        cfg.device = device
        sae = SparseAutoencoder(cfg)
        sae.to(device)
        saes.append(sae)

        optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
        optimizers.append(optimizer)

        if cfg.lr_scheduler_name == "constant":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        elif cfg.lr_scheduler_name == "constantwithwarmup":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / cfg.lr_warm_up_steps))
        else:
            raise ValueError(f"Unknown lr_scheduler_name: {cfg.lr_scheduler_name}")
        lr_schedulers.append(lr_scheduler)

        if cfg.log_to_wandb:
            wandb_queue = multiprocessing.Queue()
            wandb_queues.append(wandb_queue)
            wandb_process = multiprocessing.Process(target=wandb_logger, args=(wandb_queue, cfg.to_dict()))
            wandb_process.start()
            wandb_processes.append(wandb_process)

        # Create checkpoint directory
        checkpoint_dir = f"checkpoints/{cfg.run_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        cfg.checkpoint_dir = checkpoint_dir

    buffer = ActivationLoader(
        acts_path="/mnt/ssd/activation_cache/NeelNanda/c4-code-tokenized-2b/gelu-4l/activations_163840000.npy",
        grads_path="/mnt/ssd/activation_cache/NeelNanda/c4-code-tokenized-2b/gelu-4l/gradients_163840000.npy",
        cfg=configs[0]
    )

    num_steps = configs[0].n_training_tokens // configs[0].train_batch_size
    steps_since_last_activation = [torch.zeros(cfg.d_sae, dtype=torch.int64) for cfg in configs]

    t = time.time()

    for step in range(num_steps):
        acts, grads = buffer.get_activations()
        acts = acts.to(device)
        grads = grads.to(device)

        for i, sae in enumerate(saes):
            cfg = configs[i]
            optimizer = optimizers[i]
            lr_scheduler = lr_schedulers[i]

            if cfg.l1_warm_up_steps:
                l1_factor = min(1.0, step / cfg.l1_warm_up_steps)
            else:
                l1_factor = 1.0

            optimizer.zero_grad()
            sae_output: SAEOutput = sae(acts, l1_factor=l1_factor, grad_wrt_x=grads)

            sae_output.loss.backward()
            sae.remove_gradient_parallel_to_decoder_directions()
            optimizer.step()
            sae.set_decoder_norm_to_unit_norm()
            lr_scheduler.step()

            # Update dead feature tracker
            activated_features = (sae_output.feature_acts > 0).any(dim=0).cpu()
            steps_since_last_activation[i][activated_features] = 0
            steps_since_last_activation[i][~activated_features] += cfg.train_batch_size

            if cfg.log_to_wandb and (step + 1) % cfg.wandb_log_frequency == 0:
                dead_features_prop = (steps_since_last_activation[i] >= cfg.dead_feature_threshold).float().mean()

                l_0 = (sae_output.feature_acts > 0).float().sum(dim=-1).mean()

                wandb_queues[i].put({
                    "step": step,
                    "loss": sae_output.loss.item(),
                    "mse_loss": sae_output.mse_loss.item(),
                    "l1_loss": sae_output.l1_loss.item(),
                    "reconstr_attr": sae_output.reconstr_attr.item(),
                    "l1_attr": sae_output.l1_attr.item(),
                    "dead_features_prop": dead_features_prop.item(),
                    "l_0": l_0.item(),
                    "l1_factor": l1_factor,
                })

            # Save checkpoint every cfg.checkpoint_frequency steps
            if cfg.checkpoint_frequency and (step + 1) % cfg.checkpoint_frequency == 0:
                checkpoint_path = os.path.join(cfg.checkpoint_dir, f"checkpoint_step_{step}.pt")
                torch.save(sae.state_dict(), checkpoint_path)
                print(f"Checkpoint saved for model {i} at step {step}")
        
        if step % (cfg.wandb_log_frequency * 100) == 0:
            print('computing ce')
            losses = reconstruction_ce(saes, model, dataset)
            for i, loss in enumerate(losses):
                print(f"Model {i} CE Loss: {loss}")
                wandb_queues[i].put({"ce_loss": loss})


    # Save final models
    for i, sae in enumerate(saes):
        final_model_path = os.path.join(configs[i].checkpoint_dir, "final_model.pt")
        torch.save(sae.state_dict(), final_model_path)
        print(f"Final model saved for model {i}")

    for queue in wandb_queues:
        queue.put(None)  # Signal the logging processes to terminate

    for process in wandb_processes:
        process.join()


def main():
    log = True
    configs = []
    configs.append(SAEConfig(
        unexplained_attrib_coeff=0.05,
        l1_coefficient=8e-4,
        attrib_sparsity_coeff=8e-4,
        unexplained_attrib_method="l2",
        log_to_wandb=log,
    ))
    configs.append(SAEConfig(
        unexplained_attrib_coeff=0.05,
        l1_coefficient=5e-4,
        attrib_sparsity_coeff=5e-4,
        unexplained_attrib_method="l2",
        log_to_wandb=log,
    ))
    configs.append(SAEConfig(
        unexplained_attrib_coeff=0.05,
        l1_coefficient=3e-4,
        attrib_sparsity_coeff=3e-4,
        unexplained_attrib_method="l2",
        log_to_wandb=log,
    ))
    # no attrib
    configs.append(SAEConfig(
        unexplained_attrib_coeff=0.0,
        l1_coefficient=5e-4,
        attrib_sparsity_coeff=0.0,
        log_to_wandb=log,
    ))
    configs.append(SAEConfig(
        unexplained_attrib_coeff=0.0,
        l1_coefficient=2e-4,
        attrib_sparsity_coeff=0.0,
        log_to_wandb=log,
    ))
    configs.append(SAEConfig(
        unexplained_attrib_coeff=0.0,
        l1_coefficient=1e-4,
        attrib_sparsity_coeff=0.0,
        log_to_wandb=log,
    ))
    

    train_multiple_saes(configs)

if __name__ == "__main__":
    main()
