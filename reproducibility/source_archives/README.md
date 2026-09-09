# Immutable, inspectable source release

`selected_profiles_v1.zip` preserves the exact helper, policy, configuration
and original-record files from the pre-refactor release, including independently
pinned reviewers. Its adjacent lock records every member and checksum.

The archive uses historical relative workspace paths so the original code can
execute without source changes. This is not an installation directory: the
public `dream_sim` package checks and expands it automatically. Use
`python -m dream_sim.sources --case 01` to locate and inspect a case's source.

See [architecture](../../docs/architecture.md) for version boundaries and
[reproduction](../../docs/reproduction.md) for the supported runtime.
