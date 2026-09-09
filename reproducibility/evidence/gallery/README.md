# Video-number mapping and original provenance

The public gallery uses **Video 01**–**Video 10**, with files named `01.mp4`
through `10.mp4`. Each number is the same case ID accepted by
`python -m dream_sim.run --case 01` (substitute the desired ID).
The presentation-name change did not modify or re-encode any video frame,
change a controller/configuration, or execute another simulation trial.

`manifest.json` is the authoritative mapping from these short public names to
the full native scene ID, random seed, instruction, original recording,
original filename, immutable policy source and delivered 4× video checksum.
The public videos live in the delivery's `videos/` folder and the website's
`media/simulation/` folder, not in this code repository.

- `encoding/`: original, byte-preserved compression/frame/quality reports.
- `evidence/`: original, byte-preserved gallery physics and record-review files.
- `posters/`: original poster images, retained with their provenance here;
  the website uses identical images under short numeric filenames.
- `provenance/`: original video and website manifests, preserving the old path
  names as historical records. Those old paths are not current download links.

The manifest's `video` is a filename relative to either stated video directory;
its `poster`, `encoding_report` and `evidence` entries are relative to this
directory. `website_poster` is relative to the website repository. `files`
contains checksums for the local metadata/poster records (not for the manifest
itself). The original frozen profile catalog has not been edited.

From the DREAM repository root, `python -m dream_sim.verify_evidence` checks
the retained record bytes and the numeric-video-to-original-profile mapping.
It does not execute a new simulator task or require the downloaded videos.

Train/Val/Test belong to the upstream scene-asset split names. They do not
switch this controller into training/test runtime modes. These are selected
development demonstrations, not a held-out benchmark or a population success
rate. Each original controller version and seed is retained in DREAM even
though the website does not display these technical identifiers.

Case 05's pre-declared evaluation correction and cases 07/10's same-control
spectator re-renders remain documented in the original records and in
`../../../docs/reproduction.md`. Fresh ten-profile delivery execution evidence is separately
retained in `../reproduction/`; the full 60-attempt comparison, including
failures, remains in `../study/`. This naming cleanup changes none of them.
