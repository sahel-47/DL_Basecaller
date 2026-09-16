import os 
import random 
from pathlib import Path    
import numpy as np
import toml
import torch
from data import ComputeSettings, DataSettings, ModelSetup, load_data
from model import Model
from schedule import linear_warmup_cosine_decay
from training_utils import Trainer
from utils import load_model, load_symbol, init

__dir__ = Path(__file__).parent

class TrainArguments:
    config_path = __dir__/"models"/"dna_r10.4.1_e8.2_400bps_sup@v5.0.0" /"config.toml"
    data_dir = __dir__/"data/" / "example_data_dna_r10.4.1_v0"
    pretrained_weights = __dir__/"models"/"dna_r10.4.1_e8.2_400bps_sup@v5.0.0" 
    output_dir = __dir__/"output"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate = 2e-3
    epochs = 5 
    batch_size = 4
    chunks_per_epoch = 20000
    valid_chunks = 2000 
    valid_ratio = 0.1
    num_workers = 4 
    seed = 42
    use_amp = True 
    force_overwrite = True 
    restore_optim = False 
    save_optim_every = 5 
    grad_accum_split = 1 
    quantile_grad_clip = True
    nondeterministic = False



init(TrainArguments.seed, TrainArguments.device, not TrainArguments.nondeterministic)

device = TrainArguments.device 

config = toml.load(TrainArguments.config_path)


args_dict = {
    k: str(v) if isinstance(v, Path) else v for k, v in vars(TrainArguments).items() if not k.startswith("__")
}

args_dict["pwd"] = os.getcwd()
training_args_dict = dict(training = args_dict)

training_args_dict["training"]["pwd"] = os.getcwd()

workdir = Path(TrainArguments.output_dir)
if workdir.exists() and not TrainArguments.force_overwrite:
    raise FileExistsError(
      f"{workdir} exists. Set force_overwrite = True to overwrite."
  )

workdir.mkdir(parents=True, exist_ok=True)

with open(workdir / "config.toml", "w") as f:
    toml.dump({**config, **training_args_dict}, f)



print("[LOADING MODEL]")
if TrainArguments.pretrained_weights:
    print(f"[USING PRETRAINED MODEL {TrainArguments.pretrained_weights}]")
    model = load_model(
        TrainArguments.pretrained_weights, 
        TrainArguments.device,
        "FP32",
        )
else:
    raise NotImplementedError
    # model = load_symbol(config, 'Model')(config)


train_chunks_path = Path(TrainArguments.data_dir) / "chunks.npy"
val_chunks_path = Path(TrainArguments.data_dir) / "validation" / "chunks.npy"

total_train_available = (
    len(np.load(train_chunks_path, mmap_mode="r"))
    if train_chunks_path.exists()
    else 0
)
total_val_available = (
    len(np.load(val_chunks_path, mmap_mode="r"))
    if val_chunks_path.exists()
    else 0
)

print("-" * 60)
print(f"[DATASET AUDIT] Training Pool:   {total_train_available:,} total chunks")
print(
    f"[DATASET AUDIT] Validation Pool: {total_val_available:,} total chunks"
)
print(
    f"[CONFIG]        Using {TrainArguments.chunks_per_epoch:,} train chunks"
    f" ({TrainArguments.chunks_per_epoch // TrainArguments.batch_size:,} steps)"
    f" | {TrainArguments.valid_chunks:,} val chunks per epoch"
    f"% TrainDataset: {(TrainArguments.chunks_per_epoch/total_train_available) * 100}"
    f"% EvaluationDataset: {(TrainArguments.valid_chunks/total_val_available) * 100}"
)
print("-" * 60)


print(f"[LOADING DATA]")
data = DataSettings(
    training_data=TrainArguments.data_dir,
    num_train_chunks=TrainArguments.chunks_per_epoch,
    num_valid_chunks=TrainArguments.valid_chunks,
    output_dir=TrainArguments.output_dir,
)

model_setup = ModelSetup(
        n_pre_context_bases=getattr(model, "n_pre_context_bases", 0),
        n_post_context_bases=getattr(model, "n_post_context_bases", 0),
        standardisation=config.get("standardisation", {}),
)

compute_settings = ComputeSettings(
    batch_size=TrainArguments.batch_size,
    num_workers=TrainArguments.num_workers,
    seed=TrainArguments.seed
)

train_loader, valid_loader = load_data(data, model_setup, compute_settings)



model.to(TrainArguments.device)


trainer = Trainer(
    model=model,
    device=device,
    train_loader=train_loader,
    valid_loader=valid_loader,
    use_amp=TrainArguments.use_amp,
    lr_scheduler_fn=linear_warmup_cosine_decay(),
    restore_optim=TrainArguments.restore_optim,
    save_optim_every=TrainArguments.save_optim_every,
    grad_accum_split=TrainArguments.grad_accum_split,
    quantile_grad_clip=TrainArguments.quantile_grad_clip,
    chunks_per_epoch=TrainArguments.chunks_per_epoch,  # <-- Pass computed value
    batch_size=TrainArguments.batch_size,
    fixed_grad_scale=config.get("train", {}).get("fixed_grad_scale"),
)


optim_kwargs = config.get("optim", {})
print(f"[Starting Training] -> {TrainArguments.epochs} Epochs")
trainer.fit(
    str(workdir),
    TrainArguments.epochs,
    TrainArguments.learning_rate,
    **optim_kwargs,
)