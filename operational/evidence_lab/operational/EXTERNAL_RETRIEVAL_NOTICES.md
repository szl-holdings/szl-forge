# External retrieval: attribution and rights record

Reviewed against public primary sources on 2026-09-08. This is an engineering
reuse checklist, not blanket legal clearance. Harness code, dataset content,
model weights and dependency code retain separate licenses. Evaluation admission
does not grant training or publication approval in SZL Forge.

## Dataset content — CC BY-SA 4.0

- SQuAD 2.0: Pranav Rajpurkar, Robin Jia and Percy Liang (2018), *Know What
  You Don't Know: Unanswerable Questions for SQuAD*. Original SQuAD: Pranav
  Rajpurkar, Jian Zhang, Konstantin Lopyrev and Percy Liang (2016).
  [Publisher](https://rajpurkar.github.io/SQuAD-explorer/),
  [paper](https://aclanthology.org/P18-2124/),
  [pinned dataset](https://huggingface.co/datasets/rajpurkar/squad_v2/tree/3ffb306f725f7d2ce8394bc1873b24868140c412).
- HotpotQA: Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W.
  Cohen, Ruslan Salakhutdinov and Christopher D. Manning (2018), *HotpotQA:
  A Dataset for Diverse, Explainable Multi-hop Question Answering*.
  [Publisher](https://hotpotqa.github.io/),
  [dataset/code license distinction](https://github.com/hotpotqa/hotpot#license),
  [pinned dataset](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab).

Both publishers declare [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Preserve the original notices, attribution, dataset and Wikipedia provenance,
license links and modification history. Share adapted data under compatible
ShareAlike terms; do not silently relicense passages as Apache code. Other
rights, such as privacy/publicity or third-party claims, are not automatically
resolved by a public license declaration.

Local modifications: fixed revision/file selection; validation split only;
whitespace-based context deduplication while preserving representative original
text and source IDs; linked article groups; deterministic balanced subsampling;
derived positive document IDs; answer-offset validation; original paragraph
negative labels; supporting-sentence-to-document conversion. Hotpot sentences
are concatenated in upstream order. Full source files and cards remain intact.

SQuAD negatives are human labels for an exact paragraph, not labels of global
corpus unanswerability. Positive relevance judgments are derived and incomplete.
These public development benchmarks may have been seen during model development.

## Model artifacts

`Qwen/Qwen3-Embedding-0.6B`, Qwen Team / Alibaba, revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, declares Apache-2.0 in its
[model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/README.md).
Keep supplied copyright, license and NOTICE material and describe modifications.
No model weights were trained, fine-tuned, quantized or republished here.

`deepset/roberta-base-squad2`, deepset; Branden Chan, Timo Möller, Malte Pietsch
and Tanay Soni; revision `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4`, declares
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) in its
[model card](https://huggingface.co/deepset/roberta-base-squad2/blob/adc3b06f79f797d1c575d5479d6f5efe54a9e3b4/README.md).
Retain this attribution, source/card and license links and disclose any changes
when sharing covered artifacts. Its repository has no standalone license file
or per-tokenizer override; do not describe all selected weights as Apache.

RoBERTa tokenizer ancestry is MIT-licensed. Preserve the upstream
[fairseq MIT notice](https://github.com/facebookresearch/fairseq/blob/e75cff5f2c1d62f12dc911e0bf420025eb1a4e33/LICENSE)
alongside the deepset attribution when redistributing those assets:

```text
MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Runtime and distribution

The native Transformers implementation is separately Apache-2.0; other installed
packages retain their own licenses. The local decoder and metric code were
independently implemented; official postprocessing rules are cited in source.
This repository's Apache license does not relicense model/data assets.

The single-payload handoff contains source code, tests and notices, **not** model
weights, Wikipedia passages or benchmark Parquets. Downloads fetch originals
from immutable upstream revisions, using no paid inference API. The generated
payload/index may contain adapted dataset material and must retain this notice
if shared. No deepset, Qwen, Hugging Face, Stanford or other endorsement is implied.
