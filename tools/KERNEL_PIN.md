# Kernel Hub pin contract

Owner: `szl-holdings/szl-forge`.
Does not replace `tools/hf_kernel_hub_ops.py` or any publisher.

`szl_kernel_pin.py` is an offline pin check.

- `kernels>=0.15` requires version XOR a 40-character revision.
- SZLHOLDINGS is not a trusted publisher this observer.
- `trust_remote_code=True` is EVALUATION, not admission.
- Signature verification and `get_kernel` load are out of scope.
- `production_authorization` stays false.

```
python -m unittest discover -s tools -p test_szl_kernel_pin.py -v
python -O -m unittest discover -s tools -p test_szl_kernel_pin.py -v
```
