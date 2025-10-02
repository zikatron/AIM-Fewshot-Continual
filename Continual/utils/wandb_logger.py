"""
Weights & Biases (W&B) integration for experiment tracking.
This module provides utilities to log metrics like accuracy to W&B.
"""

import wandb


def init_wandb(config, name, tags=None, group=None, notes=None):
    """
    Initialize a new W&B run.
    
    Args:
        config (dict): Configuration parameters for the run (script arguments)
        name (str): Name of the run, should match CSV file name
        tags (list, optional): Tags for categorizing the run. Defaults to None.
        group (str, optional): Group name to organize related runs. Defaults to None.
        notes (str, optional): Additional notes about the run. Defaults to None.
        
    Returns:
        wandb.Run: The initialized W&B run object
    """
    return wandb.init(
        entity="bytefuse",
        project="ICLRMetaCL",
        config=config,
        name=name,
        tags=tags,
        group=group,
        notes=notes
    )


def finish_wandb():
    """
    Properly finish the current W&B run.
    """
    wandb.finish()