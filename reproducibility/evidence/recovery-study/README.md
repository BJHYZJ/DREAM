# Common-controller memory comparison

This study runs the recovery controller in ten houses with three seeds and both dynamic and static memory. All 60 attempts are included. Completion in the table requires passing independent physics replay and record checks, including the native-environment contact check.

| Memory | Qualified / attempts | Completion | 95% house interval |
| --- | ---: | ---: | ---: |
| Dynamic | 15/30 | 50.0% | [23.3%, 76.7%] |
| Static | 13/30 | 43.3% | [23.3%, 63.3%] |

The original task evaluator reported 16/30 dynamic and 13/30 static completions. 1 reported completion failed the independent native-contact check and remains unsuccessful in the qualified counts, with its attempt retained in the denominator. The original outcomes and failed audit reports are preserved.

Dynamic minus static completion is +6.7 percentage points, with a 95% interval of [-6.7, +23.3]. Intervals use 20,000 house-level bootstrap draws, preserving paired seeds and variants.

These houses and some seeds were inspected during development. This is a comparison on the development set; it does not measure generalization to new houses or establish equivalence to the different real-robot system.

The [analysis](analysis/comparison_analysis.json) includes every outcome and a matched comparison with the [baseline controller](../study/README.md). The input comparison identifies the three changed controller modules and verifies that task definitions and room maps are unchanged.

`protocol_and_inputs.zip` contains the frozen source, task inputs, and batch records. The 60 archives in `attempts/` contain compact execution records; `audit_reports.zip` contains replay and record-check reports. Extract these archives into one directory to obtain the layout accepted by `python -m dream_sim.study_report`.

RGB-D recordings and videos are retained in the complete local archives, identified by each attempt's `recording_archive.json`; those larger files are not included here. `manifest.json` lists checksums for every packaged file and archive member.

See the [reproduction guide](../../../docs/reproduction.md#evaluate-a-common-controller) to execute the controller or recompute the statistics.
