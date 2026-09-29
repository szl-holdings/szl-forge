# Source-aware confirmation

**When is another observation worth buying?** This research package studies that
question in finite synthetic worlds with shared source bias. The public explorer
replays all 5,184 recorded policy runs across 576 worlds. It does not perform new
inference or training in the browser.

The registered shared-bias comparison passed, but the clean-sensor cost guard
failed. The overall registered gate is **FAILED**. The learned acquisition
selector did not outperform the strongest simple control. These outcomes remain
visible in the explorer and the complete release.

- Public proof showcase: https://a11oy.net/experiments/confirmation/
- Published lab mirror: https://huggingface.co/spaces/SZLHOLDINGS/szl-forge-lab
- [Explorer source](../spaces/szl-forge-lab/foundation/)
- [Download the frozen v0.4 research release](../spaces/szl-forge-lab/foundation/data/release.zip)

## Research release and showcase version

The original v0.4 ZIP contains all source, three trained selectors and their
initial checkpoints, the frozen protocol, complete evaluation journal, 71-test
record, mathematical review and runtime evidence. Its bytes are preserved:

`869e318dd5f328205dd181ee836ef267bd2ae278f6430a9e8ddc661fbc689d03`

Extract the ZIP into a new directory, read its README and RESULTS, and run
`python verify_release.py`. The local workbench supports new exploratory trials
after installing its pinned dependencies. Its original documentation correctly
describes the publication state at sealing on 25 September 2026. This README
describes the subsequent publication. The frozen release is kept as an archive
so it remains byte-identical, including its historical documentation and test
sources; this also keeps its isolated dependency and test environment distinct
from the host repository.

The **v0.5 showcase** adds paired evidence replay, comparisons across every
condition, downloadable records and fixed-trace price sensitivity. Price
sensitivity rescales the cost of actions already taken. It does not evaluate a
policy that adapts to a new price and does not change the registered benchmark.
No new model training or scientific advantage is claimed for this presentation
upgrade. The predictor is programmed; only the small acquisition selector is
learned. This is a synthetic research experiment, not AGI.

## Build and verify the public evidence

`build_showcase.py` verifies the frozen archive, original file manifest, protocol
and all journal rows before generating static data. `test_build_showcase.py`
exercises tampered inputs and the complete output set. The public page checks
downloaded evidence against the packaged hashes; these hashes identify a
particular artifact, rather than providing an independent trust authority.

The release follows canonical GitHub source, then the existing Forge Lab
Hugging Face publisher, then the proof-origin showcase. Provider and live-site
verification are distinct from source review and local tests. Public research
influences, license observations and limitations are retained in the archive.
No upstream code or model weights are incorporated into the experiment.
