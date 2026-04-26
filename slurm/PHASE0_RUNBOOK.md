# Phase 0 — Bouchet cluster setup runbook

> **Trigger.** This runbook executes only when the user explicitly says **"begin Phase 0"**. Until then, every step below is a draft to be reviewed.
>
> **Execution model.** The user holds an active SSH session to Bouchet in a separate Terminal tab with Duo MFA approved. The agent runs every cluster command as `ssh bouchet "..."` (or `ssh transfer-bouchet "..."` for large transfers), which multiplexes through that session. **No manual SSH steps; everything below is automated.**
>
> **Stopping rule.** Confirm each step succeeded before continuing to the next. If any step fails, stop, report, and wait.

---

## 0.1 — Verify SSH connectivity

```bash
ssh bouchet "hostname && whoami && date"
```

Expected: hostname like `bouchet1`, user `prn22`, current date. If the SSH session isn't active or Duo expired, the command will hang or refuse.

**Failure handling:** stop, ask the user to refresh the SSH tab + Duo.

---

## 0.2 — Gather verified cluster info

Three independent commands; record each output for the rest of Phase 7.

```bash
ssh bouchet "slurm_checkup.sh"
```

Records the verified Slurm account name. Use whatever this reports for every `#SBATCH --account=` directive — even if it disagrees with the assumed `pi_jks79`.

```bash
ssh bouchet "mydirectories"
```

Records actual project and scratch paths. The expected forms are `~/project_pi_jks79/` and `~/scratch_pi_jks79/` but `mydirectories` is the source of truth.

```bash
ssh bouchet "sinfo -o '%P %l %G %D %c %m'"
```

Lists every partition, its TimeLimit, GRES (GPU types), node count, CPUs/node, and memory/node. Determines:
- Whether `pi_jks79` exists as a private partition (the row would have `pi_jks79` in `%P`).
- The GPU types we can request (`%G`).
- Walltime caps per partition (`%l`).

Also useful:

```bash
ssh bouchet "sacctmgr show user \$USER --associations"
```

Records concurrent-job and GPU limits per association.

---

## 0.3 — Transfer the project to Bouchet

```bash
rsync -avP \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.bin' \
  --exclude='*.safetensors' \
  --exclude='medium_smoke_runs/' \
  --exclude='baseline/runs/' \
  --exclude='entropy_filtered/runs/' \
  --exclude='slurm/_rendered/' \
  --exclude='slurm/cluster_config.local.yaml' \
  --exclude='Colab_*.ipynb' \
  --exclude='REPRODUCTION_GUIDE_*.md' \
  '/Users/rishinalem/Reproducing Token Ordering Paper/' \
  'transfer-bouchet:~/project_pi_jks79/mdm_research/'
```

The exclusion list keeps Colab drafts and the historical reproduction guides off the cluster (they're reference-only and waste rsync bandwidth) and skips local checkpoint blobs.

Verify:

```bash
ssh bouchet "ls -la ~/project_pi_jks79/mdm_research/ && wc -l ~/project_pi_jks79/mdm_research/baseline/src/*.py"
```

Should report 8 .py files (data, sudoku, model, diffusion, inference, evaluate, train, utils, plus `__init__.py`).

---

## 0.4 — Set up the Conda env (as a `devel` batch job, NOT on the login node)

Submit `slurm/00_setup.sh` (which is rendered by `slurm/_render_scripts.sh` from the verified `slurm/cluster_config.local.yaml`):

```bash
ssh bouchet "cd ~/project_pi_jks79/mdm_research && sbatch slurm/_rendered/00_setup.sh"
```

Capture the JOBID. Monitor:

```bash
ssh bouchet "squeue --me"
ssh bouchet "tail -100 ~/project_pi_jks79/mdm_research/setup_<JOBID>.out"
```

Wait for the job to reach `COMPLETED` state. The script ends with:
```
mdm env OK <torch_version>
Environment setup complete
```

If it ends with anything else, stop and surface the failure.

---

## 0.5 — Initialize / sync git on Bouchet

```bash
ssh bouchet "cd ~/project_pi_jks79/mdm_research && \
  if [ ! -d .git ]; then \
    git init && \
    git remote add origin https://github.com/pranayrishi/Masked-Diffusions.git && \
    git fetch origin && \
    git checkout -B main origin/main; \
  else \
    git pull origin main; \
  fi"
```

This makes the cluster-side checkout track GitHub. **Note:** local-laptop edits → `git push` from laptop → `git pull` on Bouchet. We will NOT commit from Bouchet (the laptop is the source of truth).

Configure the cluster-side git user (idempotent):
```bash
ssh bouchet "git -C ~/project_pi_jks79/mdm_research config user.name 'Pranay Rishi Nalem' && \
             git -C ~/project_pi_jks79/mdm_research config user.email 'pranayrishi.nalem@gmail.com'"
```

---

## 0.6 — Report Phase 0 status

After 0.1–0.5 succeed, summarize for the user:

- SSH working: hostname, user, date.
- Slurm account confirmed: <verified value from slurm_checkup.sh>.
- Project transferred: file count and total size.
- Conda env created: torch version reported.
- Partitions available: list of partition names with %l and %G.
- Whether `pi_jks79` exists as a partition (vs. only an account).
- GPU types available, with each GPU's typical queue time if reported by `qos.sh` or similar.
- Recommendation for `partition_production` and `gpu_type` based on the above.

**Stop and wait** for the user's next instruction (which will be `submit smoke` once they've reviewed this report).

---

## After Phase 0: render Slurm scripts

Once Phase 0 is approved by the user, the rendering step is:

```bash
# Locally (on the laptop), update slurm/cluster_config.local.yaml with the
# values verified in 0.2 (account, partitions, gpu_type, paths). Then:
bash slurm/_render_scripts.sh
# → produces slurm/_rendered/{00_setup.sh, 01_smoke.sh, 02_production_array.sh}.

# Push the LOCAL config (it's gitignored, so this only puts it on the laptop).
# rsync the rendered scripts to Bouchet:
ssh bouchet "mkdir -p ~/project_pi_jks79/mdm_research/slurm/_rendered/"
rsync -avP slurm/_rendered/ transfer-bouchet:~/project_pi_jks79/mdm_research/slurm/_rendered/
```

Then sit at the gate. The agent does not run any further `sbatch` until the user explicitly says `submit smoke`.
