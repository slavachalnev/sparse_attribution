import os
import time
import json
import wandb
import torch
from config import SAEConfig
from model import SparseAutoencoder, SAEOutput
from buffer import ActivationBuffer
from transformer_lens import HookedTransformer


def main():
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SAEConfig(device=device,
                    # log_to_wandb=False,  ## for testing
                    n_batches_in_buffer=10,  ## for testing
                    checkpoint_frequency=None,
                    )
    
    # Initialize wandb
    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=cfg.to_dict())
    
    sae = SparseAutoencoder(cfg)
    sae.to(cfg.device)

    model = HookedTransformer.from_pretrained(cfg.model_name)
    buffer = ActivationBuffer(cfg, model)
    # buffer = ActivationLoader(
    #     np_path="/home/slava/activation_cache/NeelNanda/c4-code-tokenized-2b/gelu-2l/activations_5k.npy",
    #     cfg=cfg
    #     )

    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    if cfg.lr_scheduler_name == "constant":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    elif cfg.lr_scheduler_name == "constantwithwarmup":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / cfg.lr_warm_up_steps))
    else:
        raise ValueError(f"Unknown lr_scheduler_name: {cfg.lr_scheduler_name}")

    # Create checkpoint directory
    checkpoint_dir = f"checkpoints/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    num_steps = cfg.n_training_tokens // cfg.train_batch_size
    steps_since_last_activation = torch.zeros(cfg.d_sae, dtype=torch.int64)

    t = time.time()

    for step in range(num_steps):
        if cfg.l1_warm_up_steps:
            l1_factor = min(1.0, step / cfg.l1_warm_up_steps)
        else:
            l1_factor = 1.0

        optimizer.zero_grad()
        acts, grads = buffer.get_activations()
        acts = acts.to(cfg.device)
        grads = grads.to(cfg.device)

        sae_output: SAEOutput = sae(acts, l1_factor=l1_factor, grad_wrt_x=grads)

        sae_output.loss.backward()
        sae.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()
        sae.set_decoder_norm_to_unit_norm()
        lr_scheduler.step()

        # Update dead feature tracker
        activated_features = (sae_output.feature_acts > 0).any(dim=0).cpu()
        steps_since_last_activation[activated_features] = 0
        steps_since_last_activation[~activated_features] += cfg.train_batch_size

        if cfg.log_to_wandb and (step + 1) % cfg.wandb_log_frequency == 0:
            dead_features_prop = (steps_since_last_activation >= cfg.dead_feature_threshold).float().mean()

            l_0 = (sae_output.feature_acts > 0).float().sum(dim=-1).mean()

            # TODO: record attribution

            wandb.log({
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

        # if step % 10 == 0:
        #     print(f"Step: {step}, Loss: {loss.item()}, Time: {time.time() - t}")
        #     t = time.time()

        # Save checkpoint every cfg.checkpoint_frequency steps
        if cfg.checkpoint_frequency and (step + 1) % cfg.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
            torch.save(sae.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at step {step}")
        
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(sae.state_dict(), final_model_path)
    print("Final model saved")

    if cfg.log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
