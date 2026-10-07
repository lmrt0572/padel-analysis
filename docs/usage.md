# Usage

> 🇫🇷 [Version française](utilisation.md)

Every command of the project, from calibrating a court to the measurements of the
report. They are run from the root of the repository, in the `padel` environment
(see the installation in the [README](../README.md#quick-start)).

## Data

Download the PadelTracker100 dataset (8.2 GB):

```bash
python scripts/download_dataset.py
```

## Calibrating a court

Thirteen points on the ground are asked for; a diagram of the court and a 5× magnifier
are shown in the window to guide each click. The last four serve as a check, and the
error printed at the end is measured on them alone:

```bash
python scripts/calibrate.py --video <video.mp4> --frame 200 --out ground_truth/calibrations/<name>.json
```

Check the calibration by eye by overlaying the court model:

```bash
python scripts/overlay_court.py --video <video.mp4> --frame 200 \
    --calibration ground_truth/calibrations/<name>.json --out outputs/overlay.png
```

The camera pose, which places the ball in three dimensions on the glass, also needs
references above the ground; those of the camera of the two finals are added with
`scripts/add_height_references.py`, and checked with:

```bash
python scripts/check_camera_pose.py --calibration ground_truth/calibrations/<name>.json
```

## The statistics video

The main demonstration: the rally, and beside it a panel that moves with the play
(minimap, rally in progress, points read off the scoreboard, strokes, volleys,
distance, top speed and time at the net of each player). It needs the saved analysis
of the minute (see below), the contact model and ffmpeg:

```bash
python scripts/stats_video.py --match FinalF --minute 8000 --start 9084 --stop 9799 \
    --contact-model weights/contact_net.pt --out outputs/stats.mp4
```

With `--replay`, the same video is drawn on the court rebuilt from the calibration,
without any image of the broadcast: players, ball, contacts and panel, that is
everything the analysis reconstructed, and only that.

## The report of a whole match

Analyse both finals in full, minute by minute (about three and a half hours on a
GTX 1650; running the command again resumes at the next minute):

```bash
python scripts/analyse_match.py --weights weights/ball_net.pt
```

Read the scoreboard, then assemble the report by pair (player tracking replayed over
the whole match, contacts, rallies, changes of ends, points and movement) into
`outputs/match_stats/<match>.json`:

```bash
python scripts/read_scores.py --match FinalF --out outputs/scores/FinalF.json
python scripts/match_stats.py --match FinalF --contact-model weights/contact_net.pt
```

The figures of the report are rebuilt with `scripts/make_figures.py`, which reads
`outputs/match_stats/`.

## The full pipeline and the annotated video

Annotated video with a minimap, and positions written to a cache:

```bash
python -m padel_analysis.cli --video <video.mp4> \
    --calibration ground_truth/calibrations/<name>.json \
    --out outputs/annotated.mp4 --cache cache/<name>.json --start 5000 --frames 1800
```

For a run meant for statistics, `--no-video` skips the rendering and produces only the
cache. A whole match then takes about an hour on a GTX 1650:

```bash
python -m padel_analysis.cli --video <video.mp4> \
    --calibration ground_truth/calibrations/<name>.json --cache cache/<name>.json --no-video
```

Then compute the statistics and the figures from that cache, without any new inference:

```bash
python -m padel_analysis.analyse --cache cache/<name>.json \
    --out outputs/<name>_report.json --figures outputs/
```

The contact demonstration needs the weights of the ball detection network. It analyses
the whole range before drawing, since the ball is chosen over the entire sequence. For
display only, the ball is hidden where the network is not sure of itself: measured on
an annotated minute, phantom trajectories (ball out of frame, ball in hand before the
serve) go from 222 frames to 36, for 97.5 % of the right positions kept. Wall contacts
whose ray meets a single surface are hidden as well: over the two annotated matches,
they carry 38 of the 42 false walls, for 2 true walls out of 33. The measured figures
are not filtered.

```bash
python -m padel_analysis.demo --video <video.mp4> \
    --calibration ground_truth/calibrations/<name>.json \
    --weights weights/ball_net.pt --start 16000 --frames 1800 --out outputs/demo.mp4 \
    --contact-model weights/contact_net.pt
```

## Marking contacts by hand

The tool shows the video and nothing of what the system detects. `s v g t f` mark a
floor bounce, a glass contact, the mesh, the net or a stroke on the frame shown; two
contacts can follow each other from one frame to the next, and the screen says so:

```bash
python scripts/mark_contacts.py --video <video.mp4> --video-name FinalF --start 16000 \
    --frames 1800 --out ground_truth/contact_marks/FinalF_16000.json
```

## The contact model and its judges

Analyse the marked minutes (the expensive pass, on a graphics card), train the model
with cross-validation, then score it on a judge (minutes that were marked and never
looked at, and that serve only once):

```bash
python scripts/analyse_minutes.py --weights weights/ball_net.pt --tag 360
python scripts/train_contact_model.py --cv --out weights/contact_net.pt
python scripts/score_minutes.py --tag 360 --contact-model weights/contact_net.pt --juge-5
```

The figures of the report are rebuilt from the numbers written by the measurement
scripts:

```bash
python scripts/train_contact_model.py --cv --curve --results outputs/measures/cv.json \
    --out weights/contact_net_rerun.pt
python scripts/make_figures.py --cache cache/<whole match>.json
```

The commands that reproduce each measurement are in the
[evaluation report](evaluation.md#reproducing-the-evaluation).

## Checking the code

The tests need neither the videos nor the weights. The pylint settings are in
`pyproject.toml`:

```bash
python -m ruff check src scripts tests
python -m pytest -q --cov=padel_analysis
python -m pylint src/padel_analysis scripts
```
