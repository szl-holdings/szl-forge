# Enterprise family recipes (Wave B)

GitHub source-of-truth YAML mirrors for Massed Unsloth family trains.
Align fields with NVIDIA Megatron-Bridge PEFT (`dim`↔rank, `alpha`, target modules)
while keeping the live engine **Unsloth on A6000**.

## Laws

- MEASURED only — recipes do not ship weights
- Private HF candidates only until Stephen `GO WEIGHT`
- Doctrine v11 honesty: Λ = Conjecture 1 (FALSE as stated); Theorem U conditional
- TORCHDYNAMO/INDUCTOR disabled on Massed eval path

## Families

| File | Mission |
|---|---|
| `chaski.recipe.yaml` | messenger / FIFO organ (F7 touch) |
| `khipu.recipe.yaml` | consensus / DAG (F4/F22; Conjecture 2 OPEN) |
| `navigator.recipe.yaml` | brain navigator |
| `receiptagent_v1.recipe.yaml` | receipt agent v1 |
| `receiptagent_v3.recipe.yaml` | receipt agent v3 |
| `willay.recipe.yaml` | conscience / veto organ |
| `triage.recipe.yaml` | IPI/IH triage |

## Leader shine folded in

- NeMo Megatron-Bridge: `linear_qkv/proj/fc1/fc2`, rank 16, alpha 32
- Unsloth: packing-ready, Triton path, single-GPU A6000
- Axolotl: YAML reproducibility twin
- Chinchilla-Lutar: refuse under-trained stub fills
- Sefirot: cookie = continual adapter stack, not wipe

Promote via green-squash PR into `szl-forge/families/enterprise-recipes/`.
