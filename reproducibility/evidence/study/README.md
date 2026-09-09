# Complete frozen comparison — separate from the selected video gallery

All 60 attempts are retained: ten development-selected houses, three seeds and
two memory variants with one frozen controller. Dynamic memory completes 7/30
tasks and accumulation-only memory 6/30. The difference is +3.3 percentage
points, with a 95% paired-house bootstrap interval of [−16.7, +23.3]. This does
not establish superiority or equivalence. The videos are not this denominator.

`analysis/study_analysis.json` and `analysis/attempts.csv` contain the original
completed analysis. `attempts/` contains one ordinary ZIP per attempt, including
failures, to keep files below the ordinary Git file-size limit. The archived
JSON/JSONL, controls, forces, trajectories and logs are unchanged. Extract all
archives into a new directory to restore their original relative layout.
`audit_reports.zip` retains all compact success-audit reports;
`protocol_and_inputs.zip` retains the declared protocol, task maps and raw-input
omission manifest. `manifest.json` binds both archive and uncompressed hashes.

From the DREAM repository root:

```bash
python -m dream_sim.verify_evidence
```

This verifies bytes, not newly executed behavior. The compact evidence omits
large original sensor arrays/videos and lists their hashes. It is not a full
raw-evidence archive. Re-running the provided policy generates fresh complete
records for physical and observation auditing; preserve failures in that new run.

The frozen comparison can be launched separately from the historical gallery:

```bash
python -m dream_sim.profile \
  --study --gpus 0 --asset-dir .runtime/assets \
  --model-cache .runtime/models \
  --output results/my_comparison --execute
```

This intentionally executes 60 new attempts and can take substantial time. It is
not needed just to reproduce the ten selected videos. The default tutorial's
`--all` selects ten historical case profiles instead, not this common-controller
comparison. Indoor simulation does not independently validate physical SLAM,
hosted mLLM verification, AnyGrasp or outdoor deployment.
