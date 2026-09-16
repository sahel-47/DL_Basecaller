import re 
import textwrap
from pathlib import Path 
from dataclasses import dataclass
from collections import defaultdict
import torch 
import parasail
import pandas as pd 
from tqdm import tqdm 
from utils import decode_ref, init, load_model, permute, __dir__
from data import load_data, ComputeSettings, DataSettings, ModelSetup
import toml
import numpy as np

class EvalArguments:
    model_directory = __dir__/"output"
    weights = 0
    data_dir = __dir__/"data"/"example_data_dna_r10.4.1_v0"
    output_dir = __dir__/"eval_output"

    dataset = "valid"
    chunks = 2000
    batch_size = 16
    num_workers = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    use_amp = False 
    standardise = True

@dataclass
class AlignResult:
    accuracy: float = 0
    num_correct: int = 0
    num_mismatches: int = 0
    num_insertions: int = 0
    num_deletions: int = 0
    ref_len: int = 0
    seq_len: int = 0
    align_ref_start: int = 0
    align_ref_end: int = 0
    align_seq_start: int = 0
    align_seq_end: int = 0




def align(*, ref, seq):
    if not seq:
        return AlignResult()

    res = parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)

    cigar = res.cigar.decode.decode()
    counts = defaultdict(int)
    for cnt, op in re.findall(r"(\d+)([A-Z\W])", cigar):
        counts[op] += int(cnt)

    # Sometimes parasail.SW will start with xD and we need to handle this explicitly
    del_start = int(match[0]) if (match := re.findall(r"^(\d+)D", cigar)) else 0
    counts['D'] -= del_start

    ref_start = res.end_ref - counts['='] - counts['X'] - counts['D'] + 1
    seq_start = res.end_query - counts['='] - counts['X'] - counts['I'] + 1

    return AlignResult(
        accuracy=counts["="] / sum(counts.values()),
        num_correct=counts["="],
        num_mismatches=counts["X"],
        num_insertions=counts["I"],
        num_deletions=counts["D"],
        ref_len=len(ref),
        seq_len=len(seq),
        align_ref_start=ref_start,
        align_ref_end=res.end_ref,
        align_seq_start=seq_start,
        align_seq_end=res.end_query,
    )




init(EvalArguments.seed, EvalArguments.device)

print(f"[LOADING MODEL] from: {EvalArguments.model_directory}")
model = load_model(str(EvalArguments.model_directory), EvalArguments.device, "FP32", weights= EvalArguments.weights)

if EvalArguments.weights is not None:
    weight_file = Path(EvalArguments.model_directory)/ f"FP32_weights_{EvalArguments.weights}.tar"

else: 
    raise NotImplementedError

model = model.to(EvalArguments.device)

model.eval()

config_file = Path(EvalArguments.model_directory) / "config.toml"

config = toml.load(config_file) if config_file.exists() else {}

standardisation = config.get("standardisation", {}) if EvalArguments.standardise else {}


model_setup = ModelSetup(
    n_pre_context_bases=getattr(model, "n_pre_context_bases", 0),
    n_post_context_bases=getattr(model, "n_post_context_bases", 0),
    standardisation=standardisation,
)

compute_settings = ComputeSettings(
    batch_size=EvalArguments.batch_size,
    num_workers=EvalArguments.num_workers,
    seed=EvalArguments.seed,
)

mean = model_setup.standardisation.get("mean", 0.0)
stdev = model_setup.standardisation.get("stdev", 1.0)


data = DataSettings(
    training_data=str(EvalArguments.data_dir),
    num_train_chunks=EvalArguments.chunks,
    num_valid_chunks=EvalArguments.chunks,
    output_dir=None,
)
_, dataloader = load_data(data, model_setup, compute_settings)


print("* Calling")
seqs = []
targets = []

with torch.no_grad():
        for data, target, *_ in tqdm(dataloader, total=EvalArguments.chunks // EvalArguments.batch_size):
            targets.extend(torch.unbind(target, 0))
            data = data.type(torch.float16).to(EvalArguments.device) if EvalArguments.use_amp else data.to(EvalArguments.device) 
            log_probs = model(data)

            if hasattr(model, 'decode_batch'):
                seqs.extend(model.decode_batch(log_probs))
            else:
                seqs.extend([model.decode(p) for p in permute(log_probs, 'TNC', 'NTC')])


refs = [decode_ref(target, model.alphabet) for target in targets]
alignments = pd.DataFrame([align(ref=ref, seq=seq) for ref, seq in zip(refs, seqs)])

print("* aligning")

print(textwrap.dedent(f"""
        * num_chunks      {len(alignments)}
        * accuracy        {alignments.accuracy.mean():.2%}
        * sub-rate        {(alignments.num_mismatches / alignments.num_correct).mean():.2%}
        * ins-rate        {(alignments.num_insertions / alignments.num_correct).mean():.2%}
        * del-rate        {(alignments.num_deletions / alignments.num_correct).mean():.2%}
        * seq_len         {alignments.seq_len.mean():.1f}
        * seq_lclip       {alignments.align_seq_start.mean():.1f}
        * seq_rclip       {(alignments.seq_len - alignments.align_seq_end - 1).mean():.1f}
        * ref_len         {alignments.ref_len.mean():.1f}
        * ref_lclip       {alignments.align_ref_start.mean():.1f}
        * ref_rclip       {(alignments.ref_len - alignments.align_ref_end - 1).mean():.1f}
        """))

if EvalArguments.output_dir:
        EvalArguments.output_dir.mkdir(exist_ok=True, parents=True)
        with (EvalArguments.output_dir / "seqs.fasta").open("w") as fh:
            fh.write("".join([f">chunk_{i}\n{s}\n" for i, s in enumerate(seqs)]))
        with (EvalArguments.output_dir / "refs.fasta").open("w") as fh:
            fh.write("".join([f">chunk_{i}\n{s}\n" for i, s in enumerate(refs)]))
        alignments.to_csv(EvalArguments.output_dir / "summ.txt", sep="\t")

