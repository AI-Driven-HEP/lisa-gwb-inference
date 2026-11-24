# LISA Gravitational-Wave Background Inference

Research code for *Stochastic gravitational-wave backgrounds in simulated LISA
data: neural separation and SBI-guided likelihood inference*

## Install

Use Python 3.10 for the paper environment. The workflows use CPU computation.

```bash
conda create -n lisa-gwb-inference python=3.10 pip
conda activate lisa-gwb-inference
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Run the commands from the repository root. Generated data and checkpoints
go under `results/`.

## 1. Generate data and train the separator

The separator adapts the multi-scale, multi-headed autoencoder of
[Einsle et al. (2025)](https://doi.org/10.1103/hs9b-drwx). It estimates noise
and combined gravitational-wave power from 512-bin A, E and T spectra.

```bash
python -m lisa_gwb.separator_data --out results/stage_c_pool \
  --n-events 20000 --n-val 1024 --workers 4 --seed 20260920
for seed in 20260922 20260923 20260924; do
  bash training/run_separator.sh --seed "$seed" --out "results/prod/seed$seed" \
    -- --patience 1000000 --allow_bad_bins
done
```

**Paper mode** reweights the fixed pool from loud to weaker signals and trains
each seed for 200 epochs. Run one seed for a standalone separator, or all three
for SBI-S. Rerun a command with the same settings to resume from `last.pt`.
For on-the-fly simulations, run
`python -m lisa_gwb.train_separator --data regen --M 1 --N 512 --n_avg 126 --out results/regen/best.pt`.

## 2. Generate SBI data and choose a model

```bash
python training/generate_sbi_data.py --workers 4
python training/train_sbi.py --variant sbi-s
python training/train_sbi.py --variant sbi-r
```

**SBI-S** uses the frozen separator ensemble's predicted signal in A/E and
noise in A/E/T. **SBI-R** uses measured A/E/T directly. Both fit 32 whitened
principal components and train a neural spline flow on 18,000 simulations,
with 2,000 held out for development. See the [flow settings](training/sbi_paper.json).

For SBI-R alone, generate with `--spectra raw --results results/raw` and train
with `--variant sbi-r --results results/raw`. This data mode needs no separator
checkpoints. Use `--start` and `--stop` for a range of SBI chunks, or rerun
the generator to fill in missing chunks.

## Quick end-to-end check

Use a separate output folder for three one-epoch separator runs and
50 training plus 50 development simulations for SBI.

```bash
python -m lisa_gwb.separator_data --out results/smoke/pool \
  --n-events 16 --n-val 8 --workers 1
for seed in 20260922 20260923 20260924; do
  bash training/run_separator.sh --seed "$seed" --out "results/smoke/prod/seed$seed" \
    --pool results/smoke/pool --stop-after 1 --batch 8 --threads 1 \
    -- --n_train 16 --allow_bad_bins
done
python training/generate_sbi_data.py --mode smoke --results results/smoke --workers 2
python training/train_sbi.py --mode smoke --results results/smoke --variant sbi-s
python training/train_sbi.py --mode smoke --results results/smoke --variant sbi-r
```

## Inference tutorial

Open [inference.ipynb](tutorials/inference.ipynb) with a Python kernel from the
installed environment. Set `VARIANT` to choose SBI-S or SBI-R and point
`RESULTS` at the corresponding training directory. The notebook shows posterior
sampling on a simulated observation and accepts your own binned spectra.

## Citation and license

See [CITATION.cff](CITATION.cff) to cite the manuscript by Moslem Ahmadvand
and Saeed Ansarifard. The code is released under the [MIT License](LICENSE).
