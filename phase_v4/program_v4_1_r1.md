# Program V4.1-R1 — Pre-Confirm Schema Amendment (`governed_loop_guard` v3.9.0)

## Provenance & Zero-Exposure Audit
Prior to **any** evaluation on `Confirm_V4_1` (`v4_1_math_confirm150.jsonl`, `v4_1_lean_confirm150.jsonl`) or `Holdout_V4_1` (`v4_1_math_holdout150.jsonl`, `v4_1_lean_holdout150.jsonl` — `0` Confirm looks and `0` Holdout looks executed), `governed_loop_guard_v39.py select-search-winner` raised a schema validation error (`Unsupported bootstrap_mode 'cluster'`) because `v3.9.0` renamed 1D cluster bootstrap from `"cluster"` to `"cluster_1d"`.

Without modifying `IMMUTABLE_MANIFEST_V4.json` (`8f69efbd508ae4291fde22faa6edcc6a6cae759228d45dbcb147ea5f85e47c50`) or `IMMUTABLE_MANIFEST_V4_1.json` (`b9c58dcb423892cbb461b1e14c77793494ed69a5d3501209a1e8ae991997a984`), `IMMUTABLE_MANIFEST_V4_1_R1.json` locks `configs_v4_1_r1/` and `eval_v4_1_r1/`:
1. Sets `"bootstrap_mode": "cluster_1d"` in `configs_v4_1_r1/gate_rules_v4_1_*.json` matching `governed_loop_guard_v39.py`.
2. Aligns `"paired_aux_noninferiority": {"stop_rate": 0.05}` with the `5.0 pp` one-sided `97.5%` LCB margin used across Phases `V1–V4.0` (appropriate for `N=150` cluster sampling noise where `1.96 * SE ≈ 2.2 pp`).
3. Includes `muse_math_gspo_pure_onpolicy` as a diagnostic paired contrast arm alongside `muse_math_gspo_faithful_luffy` and `b0_base` on `Confirm_V4_1` and `Holdout_V4_1` to test mixed-policy `LUFFY` vs. pure on-policy `GSPO` directly on unexposed splits.
