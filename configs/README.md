# Numbered reproduction cases

`cases.json` maps Video 01–10 to the exact historical profiles. `tasks/` contains
readable, byte-preserved task/room-map files, while `locks/` fixes the models and
upstream scene resources. Full scene IDs and seeds are retained here rather
than displayed on the website.

The public case index is a checked view of the catalog in the source archive;
the runner uses the verified original catalog. Historical `records` and source
paths in that index refer to archived content, not files alongside this README.
Use `python -m dream_sim.sources --case 01` to locate the expanded source tree.

A task's original `constructed_not_executed` field describes the configuration
when it was prepared, not the eventual experiment outcome. Consult
[`reproducibility/evidence/`](../reproducibility/evidence/) for execution records.
Do not rewrite historical task fields or select a new seed while claiming to
reproduce an existing video. New experiments need their own recorded profiles.
