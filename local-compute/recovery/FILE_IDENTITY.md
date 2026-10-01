# GGUF inspector v1.1: Windows path/descriptor metadata semantics

## Owner failure and source defect

The owner ran the pinned #269 inspector on BETTERWITHAGE / Python 3.12.10.
Both historical model files stopped with `INPUT_CHANGED_AT_OPEN`, before parsing
or a verified full-file digest. This does not prove a changed or corrupted model.
The historical report is preserved; no score or model file is rewritten.

A source review of CPython **v3.12.10** explains a reproducible false positive:
`Modules/posixmodule.c` routes pathname stat through `win32_xstat`, which copies
birthtime into ctime for backwards compatibility. Its FSTAT macro instead routes
to `_Py_fstat_noraise` in `Python/fileutils.c`, whose returned structure retains
`FILE_BASIC_INFO.ChangeTime` as ctime. Creation and metadata-change timestamps
need not be equal for an unchanged file. The prior five-field predicate compared
these different meanings directly. The owner's short report does not identify
which field differed, so this mechanism is not claimed as a byte-level diagnosis
of those particular model files until the corrected run records it.

Primary pinned source:
- https://github.com/python/cpython/blob/v3.12.10/Modules/posixmodule.c
  (`FSTAT`, `win32_xstat`, `os_fstat_impl`)
- https://github.com/python/cpython/blob/v3.12.10/Python/fileutils.c
  (`_Py_attribute_data_to_stat`, `_Py_fstat_noraise`)
- https://docs.python.org/3.12/library/os.html#os.stat_result

## Correction, not a bypass

The cross-API opening check binds file device, nonzero file ID, size, exact
nanosecond mtime, and **explicit nanosecond birthtime on Windows**. On POSIX it
continues to include ctime. It requires regular files and integer metadata;
missing birthtime is a refusal, not a fallback to a weaker comparison.

The open descriptor is sampled before and after the read through **fstat both
times**. Its ctime is still compared exactly, along with identity, size, mtime
and Windows birthtime. The path is separately sampled before and after through
**path stat both times**, with its own ctime comparison retained. No field is
rounded, coerced, made optional, or compared with a tolerance. No error is caught
and relabeled a successful identity check.

The complete pinned blob SHA-256 remains required after the read. Local fixed-drive
admission from #270, reparse refusal, parser/range/type budgets, historical blob
pins and tensor/metadata comparison semantics are unchanged. Recovery v1.2,
paired-attempt claims, inference policy and both installed model tags are untouched.

A successfully inspected file reports a boolean
`file_identity_observation.windows_cross_api_ctime_difference`; raw timestamps,
file IDs and local paths are not included. Genuine mismatches now report only
which fixed field names differed, e.g. `INPUT_CHANGED_AT_OPEN:st_ino`.
The console includes per-model state/error codes instead of hiding them behind
an `INCOMPLETE` summary. These messages remain evidence, not a repair of weights.

## Regression scope

The new tests include an independently assembled tiny GGUF, distinct path/fstat
ctime values, missing fields, strict ID/size/mtime/birthtime changes, descriptor
ctime mutation, path replacement, full-file digest mismatch, and visible errors.
A native Windows test sets an old creation time **only on a disposable fixture**
with SetFileTime, reproduces the prior predicate on Python 3.12.10, and verifies
the corrected inspector can still hash that unchanged file. Production code
never sets timestamps and never imports this test. Native fixture success is
not proof of owner weight health, fresh model quality, or completion of #264.

Existing and new tests run in the normal Windows/Linux recovery workflow. No
GPU, network, model installation or paid compute is part of these tests. A new
read-only owner forensic run may produce a new report after source admission;
it does not consume or reset the separate once-only ReceiptAgent experiment.
