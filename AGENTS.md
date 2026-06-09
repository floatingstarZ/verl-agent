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
