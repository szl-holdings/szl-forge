# Edge neural nets: float32 → quantized feasibility, with proof gates

Source: Bill McCartney, [Thing Done, Neural nets on chips that "can't do" neural nets, part 1](https://www.thingdone.com/field-notes/fleet-neural-net-float32/), 2026-11-14. Direct HTML fetched successfully on BETTERWITHAGE; SHA-256 `fc97dec5385b4bb3a5b13f1547dc56b7c7095a395788d6351156f4a8735013c2`.

## What the article actually establishes

- A synthetic four-class waveform task (sine, square, chirp, band-limited noise), 32 8-bit ADC readings/window; 4096 train, 512 validation and 512 heldout test windows.
- MLP 32→32→16→8→4 with ReLU: **1,756 weights/biases, 1,696 MACs**, ~7,024 float32 bytes for parameters. NumPy/Adam training 80 epochs, reported 90.4% heldout accuracy. These are **author-reported** metrics, not independently reproduced by SZL.
- 16 build targets on 15 boards and six ISA families; author reports PIC16 ~2.3 inferences/s, AVR Uno ~26/s; 8-bit soft-float can work when decision deadlines permit.
- Reported MSP430FR2433 float32 build 17,320 bytes exceeds its 15,232-byte program memory by 2,088 bytes. PIC16 float32 consumes 99.9% flash. Firmware and toolchain overhead matter as much as MAC count.
- The author plans an int8/int4 follow-up; **do not claim quantization performance or accuracy from this first float32 article**.

## Primary upstream research / implementations

- [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) — optimized fixed-point kernels on Arm Cortex-M; benchmark against reference kernels without importing unsupported ISA assumptions.
- [TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro) and [LiteRT Micro](https://ai.google.dev/edge/litert/microcontrollers/overview) — operator/runtime packaging, memory arena planning, and microcontroller deployment.
- [microTVM / Apache TVM](https://tvm.apache.org/docs/topic/microtvm/index.html) — compiler-driven kernel scheduling and device-specific code generation.
- [emlearn](https://github.com/emlearn/emlearn) — lightweight inference for microcontrollers, including small tree and neural models.
- [Edge Impulse](https://docs.edgeimpulse.com/) — embedded ML training-to-deployment workflow; useful as a measurement/control reference, not source to copy.
- [MCUNet](https://arxiv.org/abs/2007.10319), [TinyNAS/TinyEngine](https://github.com/mit-han-lab/mcunet) — co-design architecture search and runtime memory constraints.
- [HAQ: Hardware-Aware Automated Quantization](https://arxiv.org/abs/1811.08886) — hardware feedback for layerwise precision.
- [HAWQ-V3](https://arxiv.org/abs/2011.10680) — hardware-aware quantization; verify accuracy and kernel availability rather than treating bitwidth as free speed.
- [TinyMLPerf](https://github.com/mlcommons/tiny) — comparable latency/energy/accuracy measurement discipline.

## SZL-specific mathematical co-design

For dense layers with dimensions `d_0, ..., d_L`:

`MAC = Σ_l d_(l-1)·d_l`, `PARAM = Σ_l (d_(l-1)+1)·d_l`.

A conservative first-pass budget is:

`flash_est = ceil(PARAM·bits/8) + measured_code_bytes`

`ram_est = 2·max(d_l)·activation_bytes + scratch_bytes`

`latency_upper_est_ms = 1000·(MAC·cycles_per_MAC_upper + non_MAC_cycles_upper)/clock_hz`.

These are *bounds conditional on truthful measured inputs*, not firmware compiler outputs. int4 packing and dequantization, scaling constants, runtime kernels, stack/interrupt usage, compiler ABI, activation buffers, peripherals and memory alignment must be separately included. The code does not establish an actual bound unless inputs are measured under the target toolchain.

**Proposed research extension:** choose architecture, per-layer bitwidth and kernel subject to latency, flash, RAM and heldout quality constraints:

`min_(architecture, bits, kernels) [α·energy + β·latency + γ·flash] + λ·accuracy_loss`

subject to exact source/weights SHA, no safety regression, `flash ≤ device_flash`, `ram ≤ device_ram`, `latency_p99 ≤ deadline`, independent heldout accuracy and target-board reproducibility. Evaluate energy in joules/decision with calibrated instruments, not cycle counts as a proxy.

## Original SZL Forge implementation

`tools/edge_nn_budget_gate.py` accepts a strictly bounded, externally supplied JSON budget contract and emits a hash-bound **STRUCTURALLY_FEASIBLE_UNVERIFIED** or **HOLD** result. The tests reproduce the reference topology's parameter/MAC counts but **do not train a model or benchmark any MCU**.

The workflow should eventually connect to SZL Forge model receipts, SZL Lake source/data rights, GitHub exact commit and build toolchain, board fleet attestation, HF approved model cards and independently witnessed runtime metrics. It must not auto-deploy firmware, claim real power, overwrite private checkpoints, or promote generated model scores to verified hardware results.

**Next owner admission:** obtain legal access to the author's board data/code or create SZL's own reproducible dataset; train and evaluate pinned float32 baseline, quantize int8/int4 with measured accuracy; compile for authorized real targets; independently check 64-window (or larger heldout) output parity; measure flash/RAM/latency p50/p95/p99 and joules/decision; then publish rights-cleared, source-bound receipts. No paid hardware purchase or MCU flashing authorized in this source-only PR.
