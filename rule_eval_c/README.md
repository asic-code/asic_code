# rule_eval_c — Pure C rule evaluation kernel

Layer 3: bitset AND + popcount + row-level masks. All hot-path bit ops in C; Python only applies results.

## Build

Build the extension **in place** (no `pip install -e`; avoids editable-install copy failures).

**Linux / macOS**:

```bash
cd rule_eval_c
python setup.py build_ext --inplace
cd ..
```

**Windows** (MSVC required):

```batch
cd rule_eval_c
python setup.py build_ext --inplace
cd ..
```

If you see "Microsoft Visual C++ 14.0 or greater is required", install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and run the above again.

The built `.so` / `.pyd` is created inside `rule_eval_c/`. Run from the repo root; `import rule_eval_c` will use it. Re-run after `git pull` if the C sources change.

## Use

When the extension is installed, `apply_range_signatures_to_dataset` in `Heterogeneous_Method.turn_bin_conditions` uses it automatically when:

- `RANGE_MATCH_ENGINE == "bitset"`
- All condition masks are precomputed (`precompute_all`)
- Signatures and conditions are non-empty

If the extension is not installed or import fails, the existing Numba/numpy path is used.  
When the C extension **is** used, it returns both scores and row-level masks (`out_masks`); Python only does `unpackbits` and pandas assignment — no AND/popcount in Python, **C ON ⇒ no Numba dependency**.  
To force the Numba/Python path even when the C extension is installed, run the main script with `--no_rule_eval_c`.

## Interface

```python
import rule_eval_c
# rule_ptr: int64 array, length n_rules+1 (CSR row pointers)
# rule_cond_ids: int64 flat (cond indices per rule, in sorted-by-count order)
# cond_bitsets: uint8 (n_conds, n_bytes), C-contiguous
# n_conds, n_bytes: int
# out_masks: optional uint8 (n_rules, n_bytes), C-contiguous; filled with row-level packed mask per rule
scores = rule_eval_c.eval_rules_c(rule_ptr, rule_cond_ids, cond_bitsets, n_conds, n_bytes)
scores = rule_eval_c.eval_rules_c(rule_ptr, rule_cond_ids, cond_bitsets, n_conds, n_bytes, out_masks=arr)
# returns list of int (popcount per rule)
```

## Data layout

- `n_bytes = (row_count + 7) // 8` (packed bits, numpy `packbits` convention).
- `cond_bitsets[cond_id]` = packed mask for condition `cond_id`.
- Rule `r`’s cond ids are `rule_cond_ids[rule_ptr[r] : rule_ptr[r+1]]`.
