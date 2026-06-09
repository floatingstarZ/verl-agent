# Packaged Copy Notes

This archive is a code-focused copy of `verl-agent`.

Large or generated artifacts were intentionally omitted to keep the package small:

- checkpoints and model weights: `checkpoints/`, `*.pt`, `*.pth`, `*.safetensors`, `*.ckpt`, `*.bin`
- runtime outputs: `logs/`, `outputs/`, `wandb/`, cache databases, `__pycache__/`
- nested external repositories: `EXPS/repos/`
- images and compiled documents: `*.png`, `*.jpg`, `*.jpeg`, `*.gif`, `*.webp`, `*.pdf`
- bulky environment data/indexes/resources, including WebShop indexes/resources and browser binaries such as `chromedriver`
- visualization figures were omitted; lightweight tabular visualization data under `VISULIZATION/data/` is kept when present

To reproduce full experiments, restore the omitted checkpoints, datasets, WebShop/ALFWorld resources, logs, and external repositories from the original workspace or their upstream sources.

# Current StepPPO-v3 Rule

StepPPO-v3 in this branch is deliberately simple: it is a value-head-only experiment on top of GiGPO.

- Keep GiGPO policy training unchanged: same policy advantage, same policy loss path, same rollout/training recipe unless a user explicitly asks for a separate new algorithm version.
- Add the same actor-side shared value head idea used in StepPPO-v2, and test whether this value head can learn stable step-level value/return targets.
- The key question is only: can the value head learn stably, and can it avoid hurting GiGPO training behavior?
- Do not reinterpret v3 as pure step advantage, action-level PPO ratio, a new `step_ppo_v3` estimator, separate critic training, or a broader algorithm stack by default.
- Treat `detach_value_backbone=true` as the safe default for the main v3 value-head run, because the value loss should not update the actor backbone when checking whether GiGPO is unaffected.
- Primary success signals: value loss/RMSE/MAE become stable, explained variance becomes useful, and GiGPO policy metrics such as valid action ratio, response length, clip ratio, and validation score do not regress versus the no-value-head GiGPO baseline.

# Platform TeX Path

On this platform, use the local Tectonic binary for EXPS LaTeX documents:

```bash
/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/bin/tectonic
```

Recommended compile environment:

```bash
HOME=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan \
XDG_CACHE_HOME=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/.cache \
TECTONIC_CACHE_DIR=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/.cache/Tectonic \
/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/bin/tectonic --outdir EXPS EXPS/tex/<doc>.tex
```
