# Changelog

## 1.0.0

First release.

- Players: detection, pose, tracking under the court constraint, positions in metres.
- Ball: candidates, global path over the sequence, trained detection network.
- Contacts: temporal network over hand-marked minutes, glass inferred from the pace of
  the ball, surface placed in 3D by the camera ray.
- Game: rallies cut on broadcast splices, score read off the scoreboard, statistics per
  player over a rally and per pair over a match.
- Rendering: demonstration video with a statistics panel, replay on the rebuilt court,
  figures of the evaluation report.
- Evaluation: 82 to 83 % of contacts with the right surface on minutes never seen, 86 %
  on a rally from another tournament. Details in `docs/en/evaluation.md`.
