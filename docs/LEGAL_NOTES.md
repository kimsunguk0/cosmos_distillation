# Legal Notes

This project uses model and dataset assets with different license terms.

## Reminders

- Alpamayo 1.5 weights are non-commercial according to the referenced spec.
- Alpamayo 1.5 inference code is Apache 2.0.
- Cosmos-Reason2-2B supports derivative model distribution under its model
  license, but the exact obligations still need to be checked before release.
- PhysicalAI AV access may be gated and must be used under the dataset terms.

## Engineering implications

- Keep provenance for all teacher-generated artifacts.
- Do not relabel teacher output as ground truth.
- Preserve enough metadata in reports to audit how a target was produced.
