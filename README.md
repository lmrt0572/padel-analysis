<p align="center">
  <img src="docs/figures/banniere.gif" alt="A rally played on the padel court rebuilt in 3D" width="100%">
</p>

# Padel Analysis

> 🇫🇷 [Version française](docs/README.fr.md)

Analysis of padel matches filmed by **a single camera**: the players, the ball, every
contact and **what the ball touched** (racket, floor, glass, mesh or net), then **the
statistics of the game**: distance covered, speed, time at the net, strokes and
volleys, per player over a rally and per pair over a match.

**82 to 83 %** of contacts recognised with the right surface on minutes never seen ·
**86 %** on a tournament never seen · a pair's distance **within 1 %** over a whole
match · checked against more than **2,000 contacts marked by hand**.

## The project in one minute

https://github.com/user-attachments/assets/d661c68a-3d02-4013-a0c8-071019c78d98

<sub>With sound. The rally in the film, its contacts and its statistics are the ones
the project measured; between two contacts, the ball's 3D trajectory is an
illustration. Footage: PadelTracker100 dataset (CC-BY-4.0), World Padel Tour
broadcast.</sub>

## What the project produces

https://github.com/user-attachments/assets/a1eb8fcd-89af-4678-96fa-427a242407b9

<sub>The whole rally as the project renders it: the tracked players, the ball, the area
touched at each contact and the statistics panel. Footage: PadelTracker100 dataset
(CC-BY-4.0), World Padel Tour broadcast.</sub>

A video of the rally with, beside it, a panel that moves with the play: the players'
minimap, the rally number and length, each pair's strokes in the rally, the points read
off the scoreboard and credited to the last striker, and for each player their strokes,
volleys, distance covered, top speed and time at the net. At each contact, the area
touched (service box, back court, glass panel) lights up in perspective.

<p align="center">
  <img src="docs/figures/panneau.png" alt="Statistics panel of a rally" width="300">
</p>

## How it works

```mermaid
flowchart LR
  V[Video] --> C[Court calibration]
  V --> J[Players: detection, tracking, identity]
  V --> B[Ball: candidates, then the best path over the whole sequence]
  J --> K[Contacts: temporal network and ball physics]
  B --> K
  C --> S[Surface touched, placed in 3D]
  K --> S
  V --> T[Scoreboard and broadcast cuts]
  S --> P[Rallies, points, statistics]
  T --> P
```

- **The ball is chosen over the whole sequence**, not frame by frame: recall goes from
  16 % to 72 %, then to 80 % with a trained detection network.
- **Contacts** are decided by a small temporal network, trained on hand-marked minutes,
  which combines the trajectory, the players' gestures and the court geometry.
- **Glass contacts the picture does not show are inferred from physics**: a ball too
  slow to have reached the player straight after its bounce went by the glass (right
  13 times out of 14).
- **A camera does not see depth**, but at a contact the ball lies on a known surface of
  the court: the camera ray places it in three dimensions.

## Results

| Stage | Measure | On data never seen |
|---|---|---|
| Players | Detection (F1) | 0.88 |
| Players | Tracked identity (IDF1) | 0.84 |
| Ball | Found within 10 px | 80 % |
| **Contacts, end to end** | **Right surface** | **82-83 %** |
| of which strokes · bounces · glass | Right surface | 93 % · 78 % · 64 % |
| Another tournament, another venue | Right surface | 86 % |

The path, over 716 marked contacts: hand-tuned rules give **62 %**, a learned model
**78 %**, glass physics and an ensemble of 18 networks **83 %**. Every figure is
measured on minutes **never looked at while tuning**; when cross-validation promised
87 % and a fresh judge answered 82 %, 82 % is the figure kept.

The details (measurements, ablations, abandoned attempts and why) are in the
**[evaluation report](docs/evaluation.md)**.

## A match report

Both finals analysed in full, by pair: the points read off the scoreboard, the
strokes, volleys, shots after the glass, distance covered and time at the net. Every
figure is checked: movement against the dataset's annotated positions (within 1 %),
strokes against the hand-marked minutes (within 1 %; volleys within 7 %), points
against the scoreboard.

![Report by pair for both finals](docs/figures/bilan_paires.png)

<p align="center">
  <img src="docs/figures/points_longueur.png" alt="Points won by rally length" width="62%">
  <img src="docs/figures/occupation_paires.png" alt="Court occupancy by pair" width="34%">
</p>

The report is given **by pair, not by player**: when tracking confuses two partners,
the pair's total stays right, but one player's distance is off by more than 12 % once
in ten. The teams are followed from one change of ends to the next by the score, which
says when they change; the report therefore starts at the first scoreboard reading.
Occupancy is folded onto one half: the net at the top.

## Limitations

- **Post-match analysis**, at a few frames per second on a GTX 1650.
- **A single camera**: at the far end of the court, a bounce off the glass moves the
  ball by a few pixels only, and a third of those contacts are still missed. Sound does
  not recover them: tested, they are silent in the broadcast recording.
- **Transfer to another court** is shown on a single rally, not on a match.
- **Player identity** is lost at broadcast cuts, mostly in the men's match.
- **Mesh and net** contacts are too rare to be learned.
- **A single annotator** for all the hand-marked ground truth.

## Quick start

```bash
conda create -n padel python=3.11 -y
conda run -n padel pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
conda run -n padel pip install -e ".[dev]"
conda run -n padel python -m pytest -q
```

Installing CUDA torch explicitly is not optional on Windows: the default torch is a CPU
build, and inference would go from minutes to hours.

```bash
python scripts/download_dataset.py
python scripts/analyse_minutes.py --weights weights/ball_net.pt --tag 360 --match FinalF --minute 8000
python scripts/stats_video.py --match FinalF --minute 8000 --start 9084 --stop 9799 \
    --contact-model weights/contact_net.pt --out outputs/stats.mp4 --replay
```

Every other command (calibrating a court, marking contacts, training and judging the
model, rebuilding the figures) is in **[docs/usage.md](docs/usage.md)**.

## Data and credits

- **[PadelTracker100](https://doi.org/10.5281/zenodo.14653706)** (CC-BY-4.0): two matches
  of the 2022 World Padel Tour Finals at 1920×1080 and 30 frames per second, with the
  players' poses, the ball and the strokes annotated. Its pose annotations swap left and
  right for every paired joint except the ears; `perception/keypoints.py` puts them back
  in COCO order.
- **Decorte et al.**, *Multi-Modal Hit Detection and Positional Analysis in Padel
  Competitions*, CVPR Workshops 2024: a rally of their dataset served for the test on
  another tournament.
- No video, broadcast image or network weights are versioned.

## Licence

AGPL-3.0, see `LICENSE`. The project depends on Ultralytics, distributed under AGPL-3.0,
which imposes this licence on the whole: the code cannot be reused in a proprietary
product.
