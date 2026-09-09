# Demonstration catalog

Video 01–10 correspond to the same numbered cases accepted by `dream_sim.run --case`. Each entry in `manifest.json` records the scene, seed, instruction, controller source, task checksum, source recording, and 4× video checksum.

Videos are hosted in the website's `media/simulation/` directory and can be viewed in the [gallery](https://bjhyzj.github.io/dream-web/simulation/).

| Directory | Contents |
| --- | --- |
| `encoding/` | Frame counts, compression settings, checksums, and quality measurements |
| `evidence/` | Per-case physics and record-review results |
| `posters/` | Video cover images |
| `provenance/` | Earlier video and website catalogs used to trace file names |

The manifest's `video` field is relative to the video directory. `poster`, `encoding_report`, and `evidence` paths are relative to this directory; `website_poster` is relative to the website repository. The `files` field contains checksums for local metadata and posters.

From the repository root, verify the catalog and record files with:

```bash
python -m dream_sim.verify_evidence
```

The gallery contains selected demonstrations using six controller versions. Train/Val/Test refer to the upstream house splits. The separate [60-attempt comparison](../study/README.md) provides the aggregate evaluation.

Case 05 retains its original score and documented evaluation correction. Cases 07 and 10 use spectator views rendered from the same saved controls. These details and the repeat-execution results are described in the [evaluation notes](../reproduction/README.md).
