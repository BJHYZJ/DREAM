# Dynamic and static memory comparison

The study evaluates one controller across ten development-selected houses, three seeds per house, and two memory variants: **60 attempts** in total.

| Memory variant | Completed tasks | Completion rate |
| --- | ---: | ---: |
| Dynamic | 7 / 30 | 23.3% |
| Static, accumulation only | 6 / 30 | 20.0% |

The difference is **+3.3 percentage points**, with a 95% paired-house bootstrap interval of **[−16.7, +23.3]**. This interval does not establish superiority or equivalence. The houses were selected during development, so the result describes this study rather than held-out generalization. The ten gallery videos form a separate qualitative selection.

## Records

| Path | Contents |
| --- | --- |
| `analysis/study_analysis.json` | Analysis plan, per-attempt outcomes, and aggregate statistics |
| `analysis/attempts.csv` | Tabular outcomes |
| `attempts/` | One ZIP per attempt, including failures |
| `audit_reports.zip` | Compact evaluation reports for successful outcomes |
| `protocol_and_inputs.zip` | Study protocol, task maps, and input omission manifest |
| `manifest.json` | Archive and member checksums |

Extract the ZIPs into a new directory to restore their recorded relative paths. The compact archives contain JSON/JSONL, controls, forces, trajectories, and logs. Large sensor arrays and videos are listed by hash and retained in the authors' archive.

To verify the stored files:

```bash
python -m dream_sim.verify_evidence
```

## Run the comparison

Prepare the environment, models, and assets using the [setup guide](../../../docs/reproduction.md), then run:

```bash
python -m dream_sim.profile \
  --study --gpus 0 --asset-dir .runtime/assets \
  --model-cache .runtime/models \
  --output results/my_comparison --execute
```

This launches 60 policy attempts with the study's common controller. The gallery command `dream_sim.run --all` instead runs the ten selected case profiles with their recorded versions. Keep the outputs of all attempts when analyzing a new comparison.
