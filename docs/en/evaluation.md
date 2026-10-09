# Padel Analysis: evaluation report

> 🇫🇷 [Version française](../fr/evaluation.md)

This document details every measurement summarised in the [README](../../README.md): how it was obtained, what it is worth, and what it does not say. The figures cover two matches of the PadelTracker100 dataset: the women's final is used for tuning, the men's final is used only once, for judging.

## Results

### Camera stability

ORB + RANSAC homography between the first frame and the following ones, over 407 frames:

| Measure | Value |
|---|---|
| Largest displacement of the image corners | **0.02 px** |

The broadcast camera is fixed. A single calibration is enough per video.

### Calibration accuracy

Thirteen clicked points, including **four control points excluded from the fit** and
spread over the whole court, so that the reported error is not optimistic.

| Measure | Pixels | Centimetres |
|---|---|---|
| RMSE | 3.66 | 11.6 |
| Median | 3.66 | 9.9 |

Detail by control point:

| Point | Error (px) | Error (cm) |
|---|---|---|
| Near service line, centre | 2.78 | 3.1 |
| Net, right wall | 4.34 | 7.4 |
| Net, centre | 3.95 | 12.4 |
| Far service line, centre | 3.37 | 17.8 |

### Accuracy asymmetry between the two halves of the court

The error in pixels is uniform (2.8 to 4.3 px). The error in metres is not, because
the scale varies strongly with depth:

| Position on the court | Centimetres per pixel |
|---|---|
| Near back wall | 1.51 |
| Near service line | 2.05 |
| Net | 3.56 |
| Far service line | 5.49 |
| Far back wall | 6.47 |

**A pixel is worth 4.3 times more at the far end than at the near end.** This asymmetry
is a property of the camera angle, not a flaw of the method: it affects every position
estimate, however it is obtained.

Two consequences for what follows:

- The positions of the players in the far half are intrinsically four times noisier
  than those of the near half.
- Noise inflates the measured distances covered. The comparison between the two teams
  is therefore not symmetric within a single game.

### Inference resolution

The model input is resized before inference. Measured on 300 annotated frames, that
is 1200 players to find, by comparing the predicted ankle midpoint with the annotated
ankles:

| imgsz | ms/frame | Error (px) | Near half (cm) | Far half (cm) | Players found |
|---|---|---|---|---|---|
| 640 | 41 | 4.03 | 6.4 | 34.3 | 572 / 1200 |
| 960 | 38 | 2.70 | 4.7 | 7.9 | 745 / 1200 |
| 1280 | 59 | 2.46 | 4.4 | 7.1 | 1170 / 1200 |
| **1600** | **90** | **2.06** | **3.8** | **6.2** | **1200 / 1200** |

**The decisive column is the last one.** Resolution does not only govern accuracy, it
governs **recall**: at 640 pixels the model finds only 48 % of the players. Those at
the far end of the court, about a hundred pixels tall and often hidden by their
partner, simply disappear. The median error of 4 px shown at 640 does not reveal it,
since it only covers the players actually found.

**Resolution kept: 1600.** It finds every player, and its error at the far end of the
court (6.2 cm) drops below the calibration error (9.9 cm): refining perception further
would improve nothing, the floor being geometric.

### Perception and tracking

Measured on 1800 consecutive frames, that is one minute of play:

| Measure | Value |
|---|---|
| Speed of the full pipeline (GTX 1650, 4 GB) | 9.7 frames/s |
| Frames with four identified players | **96.50 %** |
| Reference: frames with four annotated people | 99.94 % |
| Positions outside the court in `x` | 0.00 % |

The same pipeline at `imgsz` 1280 reached only 92.33 %: resolution explains most of
the gap.

An observation whose projected position falls well outside the enclosure is rejected.
Without this bound the displayed rate rose to 97.94 %, but the difference came from
frames where a spectator on the right side of the net filled a free slot: the figure
was flattering and wrong. The rejection also separated the two far slots, whose mean
positions were merged because these intruders filled them intermittently.

Identity is maintained by Hungarian matching over four fixed slots, two on each side
of the net. The side comes from the sign of the `y` coordinate after projection: the
calibration therefore feeds the tracking directly. The assignment cost combines the
distance to the position predicted by a constant-velocity model, the colour signature
of the outfit and the confidence of the ankles.

This structural constraint is not decorative: **the detector finds more than four
people in 63 % of the frames** (spectators, ball kids, umpire), and the constrained
tracking consistently keeps the right four.

The colour signature was validated by measurement before being kept: the drift of one
player from one frame to the next is 0.031, against 0.199 between two partners. The
ratio of 6.3 confirms that it does tell apart team-mates wearing the same outfit, and
not only the two teams.

### Tactical analysis: net control

Measured on the full match, 45,934 frames, of which 44,911 carry all four players.

The players occupy two distinct depths. Over 182,713 positions, the offensive mode
peaks at **3.95 m** from the net and the defensive mode at **7.85 m**, against the back
glass. The trough between them falls at **5.85 m**, and that is where the threshold is
placed.

The service line, at 6.95 m, is deliberately not used: it is a service rule and not a
marker of tactical position, and it falls on the wrong side of the trough: it would
class the whole defensive band as offensive.

| Measure | Value |
|---|---|
| Control, near side | **38.0 %** |
| Control, far side | **20.5 %** |
| Contested | 41.5 % |
| Frames evaluated | 44,911 |

**The trough is wide and shallow**, so the exact value of the threshold is partly a
convention, and the percentages follow it:

| Threshold | Near | Far | Contested | Near/far ratio |
|---|---|---|---|---|
| 5.00 m | 28.0 % | 12.3 % | 59.8 % | 2.28 |
| 5.50 m | 33.9 % | 18.0 % | 48.2 % | 1.88 |
| **5.85 m** | **38.0 %** | **20.5 %** | **41.5 %** | **1.85** |
| 6.00 m | 39.7 % | 21.4 % | 38.9 % | 1.86 |
| 6.50 m | 43.5 % | 23.5 % | 33.0 % | 1.85 |

The absolute values therefore depend on the threshold, but **the ratio between the two
pairs hardly moves** beyond 5.5 m. The robust conclusion for this match is that the
pair on the near side held the net about **1.85 times more often** than the other,
whatever convention is chosen.

### Tactical analysis: distance and speed

Distance is given raw and smoothed. The gap between the two quantifies the share that
position noise took in it, instead of hiding it.

| Slot | Half | Raw distance | Smoothed distance | Noise share | Speed p95 | Mean depth |
|---|---|---|---|---|---|---|
| near_1 | near | 2890 m | 2261 m | **21.8 %** | 3.50 m/s | 5.41 m |
| near_2 | near | 2807 m | 2210 m | **21.3 %** | 3.51 m/s | 5.44 m |
| far_1 | far | 3282 m | 2233 m | **32.0 %** | 3.77 m/s | 6.75 m |
| far_2 | far | 3300 m | 2285 m | **30.8 %** | 3.92 m/s | 6.77 m |

**These rows are slots, not players.** The four slots stand for two positions on each
side of the net, and the teams change ends seven times during this match: `near_1` is
therefore several people in turn. These distances aggregate the ground covered *at
that place on the court*, which remains a valid and interpretable measure, but it is
not a distance per player. The [Evaluation](#evaluation) section explains why following
a player across a change of ends is out of reach of the picture alone.

**The noise share is half as high again for the far half**, which the 4.3× asymmetry
documented above predicts. The smoothed distances, for their part, are comparable
between the four slots while the raw distances were not: the apparent gap of 400
metres between the two pairs was noise, not play.

## Evaluation

The measurements that follow compare the output of the pipeline with the annotations
of the dataset over 9,000 frames, that is five minutes of play. **Both ablations come
from the same pass over the video**: no comparison can be skewed by a different sample.

### Detection

| Measure | Value |
|---|---|
| Precision | 0.778 |
| **Recall** | **0.954** |
| F1 | 0.857 |
| People predicted | 44,134 |
| People annotated | 36,006 |
| Matched (IoU ≥ 0.5) | 34,338 |

The precision of 0.778 does not say that the detector is wrong. It finds **8,000 more
people than there are annotated players**: umpire, ball kids, front rows of the
audience. These detections are correct, they are simply not players. Discarding them
is the job of the constrained tracking, and that is what ablation 2 measures.

Recall is therefore the measure that counts here: **95.4 % of the annotated players
are found**.

### Ablation 1: where the ground point comes from

Two ways of deciding where a player touches the ground: the midpoint of the ankles, or
the centre of the bottom edge of the bounding box. Both implementations coexist in the
code so that the choice is settled by measurement.

| Strategy | Overall | Near half | Far half |
|---|---|---|---|
| **Ankle midpoint** | **1.89 px** | 2.01 px (3.0 cm) | 1.77 px (**11.5 cm**) |
| Bottom of the box | 19.61 px | 24.19 px (36.5 cm) | 16.64 px (**107.7 cm**) |

Over 34,338 matched samples, **the ankles do ten times better**. The bottom of the
bounding box is not where the player touches the ground: it is the lowest point of the
box, which includes the lowered racket and a raised foot.

The gap is larger in pixels near the camera, and larger in metres at the far end of
the court: both readings are true, and it is the 4.3× asymmetry that separates them. A
metre of error at the far end with the bottom of the box is half a service box.

### Identity ground truth

The dataset gives four people per frame but never says which is which. Rebuilding that
information is almost free: over a whole match, two partners **never** come within
fifty centimetres of each other. The machine therefore associates by nearest neighbour
over the whole match, and a human arbitrates only the moments where it becomes
doubtful.

But this association assumes a continuous picture, and it is not one. **The video of
the dataset is a concatenation of the playing sequences**, dead time removed. The
scoreboard proves it: between two consecutive frames, the score goes from 30 to 40.

At each cut, the players reappear elsewhere. It is not a close approach (they do not
brush past each other, they teleport), so nothing looks ambiguous, and identity can
change silently.

| | close approaches | cuts | changes of ends |
|---|---|---|---|
| Women's final | 14 | 83 | 7 |
| Men's final | 54 | 115 | 8 |

**266 clips arbitrated by hand**, each replayed in a loop with the four players framed
in the colour of their slot.

Detecting these cuts takes a counter-intuitive criterion. Requiring that **both**
players of a pair move seems safer, and it is exactly the reverse: nearest-neighbour
association minimises apparent displacement, so a pair that swaps places at a cut
produces the signal "they did not move". The strict criterion is blind to the cases it
should catch. A single player above the threshold is therefore enough, and the
tolerance grows with the time elapsed: eight metres in three missing frames is a cut,
ten metres in seventy-seven is a player running.

A change of ends, for its part, is not an error to correct. The slots stand for a half
of the court: when the teams change ends, `near_1` is someone else, and no swap of
labels can express it. **Identity stops there and starts again**: the identity metrics
cut both sides of the comparison at that place, and neither credit nor penalise anyone
for a boundary that no information in the picture allows to be crossed.

### Ablation 2: the court constraint

The constrained tracking holds exactly four slots, two on each side of the net, and
rejects any position outside the enclosure. The baseline is ByteTrack, without any of
these constraints.

| | Mean tracks | Frames > 4 tracks | MOTA | IDF1 | Switches |
|---|---|---|---|---|---|
| **Constrained tracking** | **3.98** | **0** | **0.912** | **0.819** | **4** |
| ByteTrack alone | 4.78 | 5,152 | 0.704 | 0.281 | 67 |

**ByteTrack exceeds four tracks on 5,152 frames of the 8,990 evaluated**, more than
one in two. Nothing bounds it, and the detector hands it eight thousand people too
many. The IDF1 of 0.281 means that most reference identities are not covered by any
stable track: per-player statistics computed on that would be noise.

### What arbitration changes in the measurement

The identity ground truth can be built automatically, without human arbitration. It
then gives this, on exactly the same frames and the same code:

| | Without arbitration | After arbitration |
|---|---|---|
| MOTA | 0.913 | 0.912 |
| **IDF1** | **0.956** | **0.819** |
| **Identity switches** | **0** | **4** |

**The automatic version announces zero switches. There are four.**

The explanation fits in one sentence: the nearest-neighbour association that builds
the reference is the same assumption as that of the tracker being evaluated. At the
cuts, both are wrong together, and the metric compares an error with itself. IDF1 was
overestimated by 0.137.

MOTA does not move (0.913 → 0.912), which is consistent: it is dominated by false
positives and missed detections, not by identity. **IDF1 was needed to see the
problem, and an independent reference for IDF1 to be able to say it.**

Over these 9,000 frames there are 18 cuts, 3 close approaches and 1 change of ends:
the constrained tracking loses identity 4 times in 21 opportunities, ByteTrack 67
times.

### Airborne phases

A point at height `h` projected by a ground homography lands at `d × H / (H − h)` from
the point below the camera instead of `d`. Padel is played jumping (smash, bandeja,
vibora), so the question is not whether the bias exists but what it weighs.

| | Frames in an airborne phase |
|---|---|
| Women's final | 1,640 / 183,456, that is **0.89 %** |
| Men's final | 2,293 / 211,252, that is **1.09 %** |

| Jump height | Bias at the net | Bias at the far end |
|---|---|---|
| 15 cm | 16 cm | 36 cm |
| 30 cm | 33 cm | **74 cm** |
| 50 cm | 56 cm | **127 cm** |

**The bias is large when it occurs, and it occurs rarely.** A 30 cm jump shifts the
projected position by 74 cm at the far end of the court, six times the median error of
the ankles at the same place. But on 1 % of the frames: its contribution to a heatmap
or to a cumulative distance is marginal, while it dominates any instantaneous position
measured during a smash.

### Held-out match

Everything above concerns the women's final, which was used to tune the pipeline:
inference resolution, net threshold, court bounds, ground-point strategy. The men's
final was never used to tune anything. It was calibrated by transfer, annotated for
identity, then evaluated once.

| | Women's final (tuning) | Men's final (held out) |
|---|---|---|
| Precision | 0.778 | **0.828** |
| Recall | 0.954 | 0.943 |
| **Detection F1** | 0.857 | **0.882** |
| Ankles, overall | 1.89 px | 2.00 px |
| Bottom of the box, overall | 19.61 px | 20.57 px |
| MOTA, constrained tracking | 0.912 | **0.856** |
| **IDF1, constrained tracking** | **0.819** | **0.764** |
| **Identity switches** | **4** | **28** |
| IDF1, ByteTrack | 0.281 | 0.252 |
| Switches, ByteTrack | 67 | 101 |

**Detection and localisation transfer. Identity tracking does not.**

Detection is even better on the held-out match (F1 of 0.882 against 0.857) because its
precision rises by five points: the detector finds fewer people there who are not
playing. Localisation is identical to within eleven hundredths of a pixel, which was
expected since the geometry does not depend on the players.

Identity, for its part, degrades sharply: **4 switches become 28**. The raw figure
overstates the gap, because the men's match offers more opportunities to lose identity
over the same duration. Normalised, the gap remains:

| | Opportunities | Switches | Rate |
|---|---|---|---|
| Women's final | 21 (18 cuts, 3 close approaches) | 4 | **19 %** |
| Men's final | 38 (24 cuts, 14 close approaches) | 28 | **74 %** |

The cause is in the second column: **14 close approaches against 3**, over the same
number of frames. The men cross much more often and much more tightly: the smallest
separation between partners goes down to 0.41 m in their match against 0.56 m in the
women's. The weak point of the constrained tracking is there, and a match that tests it
five times more catches it out five times more.

What the change of match does not call into question is the ablation: the constrained
tracking keeps an IDF1 three times higher than ByteTrack (0.764 against 0.252) and
never exceeds four tracks, where ByteTrack does so on 3,231 frames.

### Ball candidates

The ball measures **10 px across** in the median, in a picture of two million pixels.
On a frozen frame, a painted line, a logo or a reflection look exactly like it. What
sets it apart is not its appearance but its **motion**: the camera being fixed, what
moves in the picture really moves.

The detection stage therefore does not decide. It compares each frame with its two
neighbours, keeps what is brighter than both, and returns a **ranked list of
candidates**. Choosing which one is the ball is left to the trajectory stage.

Measured on the evaluation slice of the tuning match, 3,638 annotated balls:

| Temporal gap | Recall 5 px | 10 px | 20 px | Median rank | In the top 10 | Candidates per frame |
|---|---|---|---|---|---|---|
| 1 frame | 0.445 | 0.611 | 0.658 | 4 | 52.3 % | 59 |
| **2 frames** | **0.662** | **0.912** | **0.989** | **6** | **70.2 %** | 78 |
| 3 frames | 0.661 | 0.913 | 0.991 | 7 | 63.7 % | 81 |
| 4 frames | 0.653 | 0.907 | 0.990 | 8 | 59.9 % | 81 |

**Comparing one frame apart loses a third of the balls.** At 30 frames per second, a
slow ball travels less than its own diameter between two consecutive frames: it
overlaps itself and the difference cancels out. Two frames apart it has moved enough
not to overlap any more, and recall goes from 0.611 to 0.912.

Gaps 2, 3 and 4 are equal on recall. **Gap 2 is kept because it places the ball higher
in the list**: rank 6 against 7 and 8, and 70 % presence in the top ten against 64 %
and 60 %. At equal recall, it is the one that helps the next stage most.

On the held-out match, 19,259 annotated balls, with the gap kept:

| | Recall 5 px | 10 px | 20 px | Median rank | In the top 10 | Candidates per frame |
|---|---|---|---|---|---|---|
| Tuning match | 0.662 | 0.912 | 0.989 | 6 | 70.2 % | 78 |
| **Held-out match** | **0.718** | **0.928** | **0.986** | 7 | 67.5 % | 87 |

**Recall transfers without loss**, and is even slightly better on the match never used
to tune anything.

**This figure is a ceiling, not a performance.** It says that the ball is available in
the list, never that anything chose it. The real difficulty is in the last two columns:
the ball is the sixth candidate among **78**, and one time in three it is not even in
the top ten. Picking it out is the job of the trajectory stage, and it is not started
here.

The drop in recall at 5 px (0.662 against 0.912 at 10 px) does not come from a
correctable bias. Over 512 balls, the offset between the centre of the motion blob and
the annotated centre is (−0.67, +0.88) px on average, and −0.37 px once projected onto
the direction of travel. It is scatter, with a median norm of 3.3 px, not a systematic
offset.

### Ball trajectory

The previous stage returns a ranked list of candidates, the ball being in it 91 % of
the time but at the sixth rank among 78. This stage must draw from it **one position
per frame**, or the absence of a position. Two methods are implemented and measured
side by side.

**Greedy growth** is the baseline. It starts from a candidate, extrapolates at constant
velocity, takes the candidate closest to the prediction, tolerates two missed frames,
and stops. The segments obtained are then arbitrated, and the ones kept are
concatenated.

**Global optimisation** decides nothing frame by frame. It keeps the 8 best candidates
of each frame, adds an "absent" state at a fixed cost, and searches by dynamic
programming for the sequence that minimises, over the whole window, the sum of an
acceleration cost and an emission cost. The state carries the current candidate **and
the previous one**, which is enough to know the velocity, and so to penalise a sudden
change without ever having had to "track" anything.

Measured on both matches, the ground truth being the ball annotation of the dataset:

| | | 5 px | 10 px | 20 px | Frames covered |
|---|---|---|---|---|---|
| **Tuning** (3,638 balls) | Greedy growth | 0.074 | 0.162 | 0.203 | 2,042 / 4,100 |
| | **Global optimisation** | **0.534** | **0.716** | **0.764** | 4,100 / 4,100 |
| **Held out** (19,259 balls) | Greedy growth | 0.050 | 0.109 | 0.132 | 10,769 / 21,471 |
| | **Global optimisation** | **0.576** | **0.733** | **0.774** | 21,471 / 21,471 |

Recall; the precision of global optimisation is equal to it, the path answering on
every frame. For the greedy one it is 0.317 and 0.214 at 10 px.

**The factor is 4.4 on the tuning match and 6.7 on the held-out match.** The four
parameters of the path were swept over 800 frames of the women's match alone and were
not touched afterwards; the men's match, five times longer, gives a slightly better
result. The fraction of the ceiling captured is the same there to within half a point:
79.0 % against 78.5 %.

**Why the baseline plateaus.** Three measurements in a row say it without ambiguity:
the ball is in the list of candidates **93.9 %** of the time, a greedy segment covers
it **52.5 %** of the time, and **16.8 %** of it remains after arbitration between
segments. The first drop is the price of the local decision: an extrapolation that
started on a wrong candidate never comes back. The second is the price of arbitration:
one has to choose between competing segments without knowing anything of what happens
elsewhere in the sequence.

**Arbitrating segments by their length was backwards.** The segments that really
follow the ball are **27 frames** long in the median; the others are **31**. An arc of
the ball is short by nature (it ends at every contact), while a false track hooked to
a slow element can run indefinitely. The correct criterion is **speed**: 13.2 px/frame
for the good segments against 8.8 for the others. This change alone takes precision
from 0.168 to 0.405.

**What the acceleration cap does, and does not do.** It was presented at the start as
the central mechanism, the one that allows sudden changes of direction at contacts.
The sweep contradicts this: from 40 to infinity, recall moves by one thousandth. It
almost never bites, and it is kept as a safeguard against a pathological frame, not as
the spring of the method.

**What remains to be gained.** 0.912 and 0.928 were available in the list of
candidates, 0.716 and 0.733 are captured. The gap, a fifth of the ceiling, is what
will justify, or not, replacing motion detection with a network.

**A caveat on method: the metric rewards always answering.** A frame without a
prediction counts as a recall failure, while a position produced where no ball is
annotated cannot be counted: 2,212 frames in this case on the held-out match. The
sweep therefore found optimal an absence cost so high that the path never gives up,
which is partly an artefact of the measurement and not a quality of the method itself.
The comparison above remains valid, both methods being judged by the same yardstick,
but the 0.733 must not be read as "the ball is located three times out of four in all
circumstances".

### Ball detection by a network

Motion detection places the ball in its list of candidates 91 % of the time, but at
the sixth rank among 78, and the trajectory recovers only 79 % of that. **Does a
trained network detect better?** The question is asked as an **ablation**: the network
replaces the candidate stage and nothing else. It answers to the same protocol as
motion detection, and the same trajectory optimisation runs behind it, with the same
costs. The measured gap therefore belongs to the detector alone.

#### The network

A narrow U-Net reads **three stacked frames**, spaced three images apart, and returns a
**heat map** at 640×360. The training target is a Gaussian centred on the annotated
ball rather than a binary mask: a ball covers only three pixels here, and a mask would
not say where its centre is. The local maxima of the map become the candidates.

Only the frames carrying an annotated ball are used for training. A frame without an
annotation is not a frame without a ball: 17.5 % of the annotated frames carry none,
and nothing says whether the ball was absent there or only unlabelled.

#### What the graphics card imposed

A GTX 1650 offers 4.29 GB, of which 3.45 are free. Measured before writing the training
loop:

| Configuration | Memory | Throughput |
|---|---|---|
| **Width 16, batch of 4, FP32** | **1.82 GB** | **15.1 frames/s** |
| Width 32, batch of 4 | 3.61 GB | 5.7 frames/s |
| Width 16, batch of 4, **mixed precision** | 0.91 GB | **5.3 frames/s** |

**Mixed precision is three times slower.** It is the usual speed-up, and on this card
it slows things down: the TU117 has no tensor cores, so the half format gains nothing
and the conversions cost everything.

**The real bottleneck was time, not memory.** At 15 frames/s, one epoch over the
34,265 training frames takes 38 minutes. Two budgets were therefore trained:

| Model | Frames | Epochs | Duration | Validation |
|---|---|---|---|---|
| One frame in three | 11,470 | 10 | 2 h 03 | 0.0103 → 0.0049 |
| Every frame | 34,265 | **6 out of 10** | 4 h 38 | 0.0056 → 0.0037 |

The second stopped at the sixth epoch, with the session that carried it, while its
validation was still falling. The weights being written after each epoch, nothing was
lost. The two validation losses **cannot be compared with each other**: they do not
cover the same frames.

#### The result

Final recall after the trajectory optimisation:

| | Detector | 5 px | 10 px | 20 px |
|---|---|---|---|---|
| **Tuning** | Motion | 0.534 | 0.716 | 0.764 |
| | Network, one frame in three | 0.690 | 0.815 | 0.858 |
| | **Network, every frame** | **0.756** | **0.837** | **0.893** |
| **Held out** | Motion | 0.576 | 0.733 | 0.774 |
| | Network, one frame in three | 0.689 | **0.798** | 0.836 |
| | **Network, every frame** | **0.728** | 0.791 | **0.845** |

Held-out match: frames 0 to 21,472, 19,259 annotated balls, as for motion detection.
The networks were measured there in two passes, cut at frame 20,100, and combined in
proportion to the annotated balls of each pass.

**The network wins on the match it has never seen**, with both models and at all three
tolerances: +6.5 points at 10 px, +15 at 5 px. The evaluation slice of the tuning match
had never been used for training, but it came from the same match: same players, same
lighting. The men's match answers the question that one could not settle: the network
learned the ball, not that match.

**The largest gain is at 5 px** on both matches. The network does not only find the
ball more often, it **locates** it more precisely than the centre of a motion blob,
whose median scatter was 3.3 px.

**Tripling the data improves localisation, not recall.** The model trained on every
frame gains 4 points at 5 px on the held-out match, but none at 10 px: it even does 0.7
point worse there than the model at one frame in three. Its lead of 2 points at 10 px
on the tuning match therefore does not transfer. It only did six epochs out of ten:
that is the only caveat, and it can only go in its favour.

A side observation: with the candidates of the network, greedy growth, which is not
the method kept, degrades (0.162 → 0.093 at 10 px). The cause was not looked for.

The weights and the frame cache are not versioned. They are rebuilt with the commands
of [Reproducing the evaluation](#reproducing-the-evaluation).

### Contact instants

The previous stage returns a position per frame and **never gives up**: there is
therefore no gap in which to read a contact. The criterion must bear on the shape of
the path.

What marks a contact is a change of direction. Measured in pixels it is not comparable
from a lob to a smash, so the turn is **divided by the speed that produced it**: a gap
of 40 px is an elbow at 5 px/frame and a trifle at 30. The contacts kept are the local
maxima of this ratio, a single one per window of 5 frames.

Measured on both matches, with the same detector applied to the reconstructed path and
to the annotated ball (the gap between the two rows is therefore attributable to the
trajectory and to nothing else):

| | Contacts | Recall of strokes | Bounces per exchange |
|---|---|---|---|
| **Tuning** (92 strokes), annotated ball | 192 | 0.891 | 0.89 |
| Tuning, reconstructed path | 244 | 0.902 | 1.23 |
| **Held out** (475 strokes), annotated ball | 874 | 0.806 | 0.86 |
| Held out, **reconstructed path** | **1,257** | **0.895** | **1.38** |

**The reconstructed path gets a better recall than the annotated ball. It is not a
quality, it is a symptom:** it produces 44 % more contacts, and detecting more
mechanically raises recall. The column that counts is the third.

**A trajectory that is 73 % right costs only a few points.** The number of bounces per
exchange goes from 0.86 to 1.38: the excess is the trajectory error, and it is
measurable as such rather than hidden in a flattering recall.

#### Precision cannot be reported as a performance

The annotation marks only the contacts with a **racket**, as intervals. These
intervals cover **48.2 %** of the annotated frames of the tuning match. A detector
drawing its instants **at random** therefore gets a precision of 0.485 there, and the
turn detector applied to the perfectly annotated ball gets 0.573. The gap is too thin
to demonstrate anything.

Two corrections were tried and changed nothing: a one-to-one matching between contacts
and strokes gives the same gain, and tightening the target around the centre of the
interval fails because the impact is not concentrated there: it is spread over almost
the whole width, standard deviation 0.48 in half-widths.

**The held-out match is the best instrument**, its intervals covering only 31.0 % of
the frames:

| | Precision | At random | Gain |
|---|---|---|---|
| Tuning, reconstructed path | 0.537 | 0.428 | 1.25× |
| **Held out, reconstructed path** | **0.429** | **0.285** | **1.51×** |
| Held out, annotated ball | 0.501 | 0.308 | 1.63× |

On the least indulgent instrument, the detector beats chance by a factor of 1.5. It is
a measurement, but a weak one, and it stayed so until a ground truth produced by hand
replaced it, further down.
The random control is computed by the code and printed next to every precision, so
that none of these figures can be read in isolation.

#### What replaces precision

The physics of padel. Between two strokes, the ball bounces **0 times** (volley), **1**
(floor) or **2** (floor then glass, or the reverse). It is a criterion that the
annotation does not provide and cannot skew. On the held-out match, the distribution
obtained is 0: 170, 1: 138, 2: 78, 3: 40, beyond 42, that is **82 % of the exchanges
within what the game predicts**.

It is also this criterion that set the tuning, and not the event metric.

#### Bounding the absolute turn

The relative criterion only knows ratios, which makes it blind to an error specific to
the reconstructed path: its speed at the 95th percentile is **512 px** against **186**
for the annotated ball. It makes jumps that no ball makes, and each jump manufactures a
turn. The absolute turn is therefore bounded between 25 and 300 px.

| | Contacts | Recall | Bounces per exchange |
|---|---|---|---|
| Relative threshold alone | 317 | 0.957 | 1.85 |
| **Bounded turn** | **246** | **0.902** | **1.24** |

The long tail (up to ten contacts between two strokes) disappears with the cap. They
were path errors, not bounces.

#### Precision, measured afterwards

A ground truth of contact instants, all surfaces together, was missing, and no public
padel dataset provides it. It was produced to classify the surfaces, in the next
section: each detected contact was replayed there and judged by hand, with a possible
answer **"no contact"** when the trajectory went straight through.

These judgements give the precision that the stroke annotation could not give:

| | Contacts judged | No contact | **Precision** |
|---|---|---|---|
| Tuning match | 194, all | 49 | **0.747** |
| Held-out match | 150, drawn at random out of 886 | 36 | **0.760** |

**One detected contact in four did not happen.** The figure is stable from one match
to the other, and it replaces the "factor of 1.5 over chance" of the previous table:
that one remained an indirect measurement, this one is direct.

An important point about its scope: these contacts were detected on the **annotated
ball**, not on the reconstructed path. It is therefore the precision of the turn
criterion itself, free of any trajectory error. On the reconstructed path, which
produces 44 % more contacts, it is very probably lower, and it is not measured.

### Contact surfaces

Knowing *when* the ball was touched does not say *against what*. A padel court is
enclosed: the ball bounces off the floor, off glass, off mesh and off rackets. That is
what this stage must decide, and it is what a tennis pipeline does not have to do, an
open court having neither glass nor mesh.

**Why the homography is not enough.** It projects onto the ground plane. It is
therefore exact for a floor bounce and wrong for any contact above the ground.
Measured on 194 real contacts: **a third project outside the rectangle of the court**,
some at 23 m for a court that is 20 long. And the distribution is almost identical
between annotated strokes and non-strokes: 67.3 % against 64.3 % inside the rectangle.
**The projected position alone separates nothing.**

**What replaces it.** A camera only gives a ray: the ball is somewhere on it, and
nothing says where. That is true in flight, and it stays true. But **at the moment of
a contact the ball is on a surface**, and the surfaces of a court are five known
planes. A ray and a plane meet at one point. The height, undetermined in general, is
determined precisely at the instant we care about.

#### Recovering the camera

A homography makes do with points on the ground; a camera pose cannot, a coplanar set
leaving the vertical direction free. The missing references come from a manual
annotation of the wall panels: top of the glass at 3 m, top of the mesh at 4 m, top of
the net at 0.92 m.

Result: camera at **x = −0.06 m, y = −26.18 m, z = +7.86 m**, centred on the axis of
the court, twenty-six metres behind the near back wall, nearly eight metres high.

The figure that commits to something is not that of the fit but that of the **control
points, which never enter the fit**:

| Control points | Median gap |
|---|---|
| On the ground (net, service lines) | **4.2 px** |
| **Above the ground (0.92 m to 4 m, at both ends)** | **8.6 px** |
| Maximum, at the far end | 14.4 px |

**What 8.6 px are worth in metres depends on depth**: 13 cm near the camera, 56 cm at
the far end, the factor of 4.3 already measured above. On a glass / mesh threshold at
3 m, that is an uncertainty of about 20 % at worst.

#### The rule

**Racket**: a wrist nearby. The dataset provides seventeen points per player,
including both wrists. Measured: the ball is **50 px** from the nearest wrist when a
stroke is annotated, against **168 px** otherwise.

**Floor or wall**: the ray is intersected with the five planes and only the physically
admissible intersections are kept: in front of the camera, and within the real extent
of the surface. The margin that absorbs the pose error is expressed **in metres and
never in pixels**: a pixel being worth 1.51 cm near and 6.47 cm far, a margin in
pixels would be four times more lax at the far end.

**Glass or mesh**: a table, once the point of impact is known in three dimensions.
Back walls: glass below 3 m. Side walls: glass within 4.1 m of a back wall. No
heuristic.

#### The ground truth, which existed nowhere

No public padel dataset labels contact surfaces: the dataset used here declares a
`Wall` category and never filled it. It was therefore produced by hand, on both
matches.

| | Tuning match | Held-out match |
|---|---|---|
| Contacts submitted | **194**, full census | **150**, drawn from 886 |
| Real contacts | 145 | 112 |
| Unreadable | 0 | 2 |

The tool replays each instant in a loop, the ball marked with a fixed cross, and
**never shows what the rule predicts**. A ground truth built on the assumption it has
to judge only measures two errors agreeing: for the identity ground truth, correcting this flaw had
taken IDF1 from 0.956 to 0.819.

The men's match has 886 contacts, that is two hours of arbitration. The sample is
**stratified**, and its size as well as its seed are recorded in the file: a draw that
cannot be redone would not be a measurement.

#### What the annotation measures of the previous stage

**A quarter of the detected contacts did not happen**: the trajectory went straight
through. The precision of the contact stage is therefore **0.747** on the tuning match
and **0.760** on the held-out match, on the perfectly annotated ball, so free of any
trajectory error.

This is the measurement that the previous section declared impossible. The stroke
annotation could not give it: its intervals cover half the frames, so that a detector
drawing at random already got 0.480 there. This one is direct.

#### What the rule is worth

| | Tuning | | | Held out | | |
|---|---|---|---|---|---|---|
| **Class** | **n** | **Precision** | **F1** | **n** | **Precision** | **F1** |
| racket | 75 | 0.830 | 0.896 | 58 | 0.806 | **0.892** |
| floor | 51 | 0.842 | 0.719 | 42 | 0.923 | **0.706** |
| wall | 17 | 0.789 | 0.833 | 12 | 0.714 | **0.769** |

**Overall accuracy 0.828 in tuning, 0.821 held out.** Seven thousandths apart: the
thresholds were not overfitted to the match that was used to choose them.

**Mesh and net cannot be measured.** One example and two on the tuning match, neither
in the held-out sample. It was expected: the mesh only occupies the top of the back
walls and the middle of the sides. No rate is published for them, and their counts are
given rather than kept quiet.

#### Arbitration by depth

When the floor and a wall are both admissible, which one to choose? The first version
preferred the floor, systematically. Measured, this choice cost **ten walls out of
seventeen**.

The rule kept prefers the wall when the candidate floor point falls beyond
`y = −7.5 m`. This threshold is not a fitted number: the camera being at `y = −26.18`
and `z = 7.86`, a contact on the near glass projects onto the floor at

| Height of the contact | 0.3 m | 0.5 m | 0.8 m | **1.0 m** | 1.6 m | 2.0 m |
|---|---|---|---|---|---|---|
| projected y | −9.36 | −8.90 | −8.17 | **−7.64** | −5.86 | −4.48 |

The threshold therefore separates the glass contacts **below about 1.05 m**. Beyond
that, the projection enters the court and nothing distinguishes it from a floor bounce
any more.

| Recall of walls | Before | After |
|---|---|---|
| Tuning match | 0.412 | **0.882** |
| Held-out match | not measured | **0.833** |

**The ceiling is not that of the threshold but that of the geometry**: the walls
missed are the high walls, and a single camera cannot tell them from a bounce.

#### What was not corrected, and why

Two errors were measured, only one can be corrected.

The second is that **floor bounces are taken for strokes**: fifteen on the tuning
match. The wrist proximity threshold was swept from 50 to 120 px:

| Threshold | 50 | 60 | 70 | **80** | 90 | 100 | 120 |
|---|---|---|---|---|---|---|---|
| Accuracy | 0.745 | 0.793 | 0.828 | **0.828** | 0.828 | 0.834 | 0.786 |

It is a plateau. Moving it trades strokes for bounces at zero sum: another signal
would be needed, not another threshold. **The threshold therefore stayed at 80 px**,
and correcting anyway, to announce two corrections rather than one, would have been
fitting noise.

#### A stratum named backwards

The contacts where **a single** surface is admissible had been labelled "settled",
assuming that a single answer meant confidence. The two campaigns say the opposite:

| | Isolated cases | Of which without a contact |
|---|---|---|
| Tuning match | 24 | **24 (100 %)** |
| Held-out match | 16 | **14 (88 %)** |

The explanation is geometric. A real contact happens in the volume of play, where the
near back wall is always admissible too, the camera being behind it: it is a candidate
for 140 of the 194 contacts of the tuning match. A ray that meets only one surface is
therefore a ray that points outside the play.

It is not a measure of confidence but a **detector of false positives**, and the
stratum now carries that name. It is not applied as a filter: 88 % on sixteen cases
does not yet justify removing detections, and that would be a decision to measure for
itself.

### End to end: what the demonstration shows

The previous sections judge each stage on what the stage before gives it. The viewer,
for their part, sees the whole chain: a forgotten contact is not shown, an invented
contact lights a zone for nothing, and none of the measurements above counts them
both.

#### A complete ground truth

Judging the contacts a chain proposes only measures its precision: a wall it never
proposed is never judged. Twenty minutes were therefore **marked in full**, each real
contact to the frame, with a tool that shows nothing of what the system detects
(`scripts/mark_contacts.py`): 1,579 contacts.

| Minutes | Contacts | Role |
|---|---|---|
| Women's final, 4 minutes | 316 | tuning, then training |
| Men's final, 5 minutes | 387 | training |
| Men's final, 3 other minutes | 239 | **first judge, scored only once** |
| Two women's minutes, one men's | 238 | **second judge, scored only once** |
| One minute of each final | 160 | training |
| Two women's minutes, one men's | 239 | **third judge, scored only once** |

A detected contact counts as right if it falls within three frames of a marked contact
and carries the right surface. The score combines the two visible errors:
2 × right / (shown + real).

#### What the thresholds gave

The rule chain of the demonstration was tuned on the four women's minutes. Measuring
the turn over three frames on either side instead of two, with a sharpness threshold
lowered from 0.50 to 0.40, takes the right contacts from 177 to 194 out of 316.
Requiring a clearer gesture for a stroke seen without a turn, 20 px per frame instead
of 10, removes 19 invented contacts without losing a right one.

Everything else was swept without gain:

- **the ball network at 720p**, trained for ten epochs: 194 right against 194. Of 97
  missed contacts, 94 had the ball correctly shown at the instant of the contact. The
  bottleneck was no longer seeing the ball, but reading its turn;
- the confidence threshold of the display, the distance to the wrist, the margin of
  the surfaces and the depth cut: the values in place were already the best;
- the height of the ball along the nearest player, to separate a bounce from a stroke:
  an interval corrected seven errors in tuning and created one elsewhere, on counts
  too small to conclude.

The remaining errors were **combinations** of cues: a soft turn next to an
accelerating wrist is a stroke, the same turn with the ball at the player's feet is a
bounce. One threshold per cue cannot express that.

#### A learned model

`contact/learned.py` describes each frame by 58 cues: trajectory, speeds and turns
over one, two and three frames, score of the network and the other positions it
proposed, gesture of the nearest wrist, distances from the ball to the elbows, wrists,
hips and ankles of the nearest player, their apparent size which stands in for depth,
surfaces the ray can meet and where, decision of the rule chain. A dilated temporal
convolutional network, which sees
about sixty frames of context, classes each frame as no contact, racket, floor, wall
or net; the contacts are the probability peaks. Glass or mesh is then read from the
geometry, as for the rules: three mesh contacts are not enough to learn it. Three
networks are averaged.

Each marked minute was first predicted by a model trained **on the others only**:

| Training minutes | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Right contacts | 71.9 % | 75.5 % | 78.0 % | 80.1 % |

The curve was still rising: four more men's minutes were marked. Over the nine
minutes, in cross-validation:

| | Right | Shown | Score |
|---|---|---|---|
| Rules | 411 / 703 (58.5 %) | 598 | 0.632 |
| **Model** | **568 / 703 (80.8 %)** | 658 | **0.835** |

The model wins on each of the nine minutes. It transfers between the matches: trained
on the women's minutes alone, it goes from 45 to 55 right contacts out of 75 on a
men's minute. The decision threshold, 0.7, was chosen on this cross-validation.

#### The verdict

The three judge minutes had been neither looked at nor scored before the model was
frozen.

| Minute | Rules | Model |
|---|---|---|
| 12000 | 55 / 82 | **60 / 82** |
| 25000 | 55 / 78 | **62 / 78** |
| 40000 | 48 / 79 | **66 / 79** |
| **Total** | 158 / 239 (66.1 %) | **188 / 239 (78.7 %)** |
| Shown contacts that are real | 84.4 % | **93.6 %** |
| Score | 0.672 | **0.823** |

The model shows fewer contacts than the rules, finds more of them and is wrong about
the surface less often. The judge's figure, 78.7 %, is within two points of the
cross-validation: the selection was not flattered.

The judge minutes come from a match of which other minutes were used for training. The
verdict therefore measures the move to **rallies never seen**, not to a match, a court
or a camera never seen.

#### A second judge, and what the follow-up cost

The first verdict was used to decide that the model replaced the rules. It could
therefore no longer measure what was built next. Three more minutes were set aside
(two in the women's final, one in the men's), marked then scored only once, after the
model was frozen.

What was tried between the two verdicts, all in cross-validation:

| | Right contacts out of 703 | Score |
|---|---|---|
| Model of the first verdict | 568 | 0.835 |
| Left-right symmetry of the court | 566 | 0.829 |
| Temporal context doubled, then quadrupled | 573 / 574 | 0.846 / 0.847 |
| Wider network | 562 | 0.836 |
| **Candidates of the detector and skeleton of the player** | **573** | **0.848** |

Three seeds per trial were needed to tell them apart: from one seed to the next, the
score moves by ±0.006, as much as most of these gaps. Only the additional cues win
with all three seeds: five more right contacts and ten fewer invented contacts on
average. The wider context, for its part, wins twice out of three and loses the third:
mean 0.843 against 0.841, so nothing. It was not kept.

The learning curve, extended, flattens: 71.1 % at two training minutes, 76.8 % at
four, 78.3 % at six, 80.2 % at eight. Marking more would bring about half a point per
minute.

| Second judge, 238 contacts | Rules | Model |
|---|---|---|
| Right surface | 139 (58.4 %) | **188 (79.0 %)** |
| Shown contacts that are real | 83.6 % | **91.7 %** |
| Score | 0.608 | **0.826** |

Three independent measurements (cross-validation 80.7 %, first judge 78.7 %, second
judge 79.0 %) give the same figure. These minutes come from both finals, so the result
does not rest on a single match; it remains established on one tournament and one
camera angle.

#### A third judge, for the sequence of the rally

The last lever considered was the structure of the rally: after a stroke comes a floor
or a wall, two strokes a few frames apart are rare. Written as strict rules, it had
failed (see above). It was taken up again as probabilities: the network proposes
candidate contacts with a low threshold, a table learned on the marks gives the
probability of each label given the previous one and the gap in frames, and the
Viterbi algorithm chooses over the whole minute which candidates to keep and how to
label them.

Two more training minutes bring the set to eleven minutes and 863 contacts. The model
is at 80.8 % there, as on nine minutes: the learning curve is indeed flat.

| Cross-validation, 863 contacts | Seeds 0-2 | 3-5 | 6-8 |
|---|---|---|---|
| Peak decoding | 0.838 | 0.838 | 0.838 |
| Sequence, weight 0.5 | 0.842 | 0.836 | 0.833 |
| Sequence, weight 1.5 | 0.844 | 0.842 | 0.835 |

The mean gap is of the order of +0.002, won on one set of seeds and lost on another:
the sequence model was not kept. The network already sees two seconds around each
instant, and what the rally could teach it, it had learned.

The model was therefore frozen as it was, then scored on three new minutes:

| Third judge, 239 contacts | Rules | Model |
|---|---|---|
| Right surface | 149 (62.3 %) | **180 (75.3 %)** |
| Shown contacts that are real | 88.7 % | **94.5 %** |
| Score | 0.687 | **0.818** |

Over the three judges together, 716 contacts never looked at before their verdict, the
model gives the right surface to **77.7 %** of the real contacts, against 62.3 % for
the rules. That is the figure to keep: the three judges taken one by one vary from
75.3 % to 79.0 %, and it is this spread, more than the cross-validation, that says the
real precision of a measurement made on three minutes.

#### What the display lights

The model chooses floor, wall or net; the exact wall and the zone are then read from
the ray. When several walls were admissible, the first in the list won, and the back
walls come before the sides there: 6 % of the glass contacts lit the back wall for a
contact on the side. The wall kept is now the one the ray reaches first from the
camera, since the ball, being visible, cannot be behind another surface.

The whole zone is lit: service box, back rectangle, glass panel. A patch centred on
the impact was also tried: it absorbs the position error instead of flipping a zone
near a line, but the whole zone reads better on screen. It remains available
(`--impact-patch`). Measured in passing, the zone drawn from the detected instant is
that of the real instant in 98 bounces out of 101: what flips is the position near a
line, not the instant.

#### Where the model is still wrong

Over the eleven training minutes, each minute predicted by a model that has not seen
it, here is for each hand-marked contact what the model answered ("nothing" for a
missed contact, and a "nothing" row for the invented contacts):

![Confusion matrix of the surfaces](../figures/confusion.png)

Strokes are found at 92 % (411 out of 445) and the floor at 81 % (219 out of 272).
**The glass is the weak point: only 64 out of 128**, 44 missed and 12 taken for the
floor. It is the confusion the geometry announced (above about one metre, a contact on
the near glass and a floor bounce fall on the same pixel), and that is where the next
gain would be, not in more marked minutes:

![Learning curve](../figures/learning_curve.png)

The curve, redone over eleven minutes, rises from 73.2 % at two training minutes to
81.6 % at eight, then 81.9 % at ten.

**Why the glass, and what was tried.** Sorted by wall, the marked glass contacts do not
pose the same problem everywhere:

| Glass | Right | Missed | Taken for the floor or a stroke |
|---|---|---|---|
| Back wall near the camera | 43 | 17 | 21 |
| Far back wall | 14 | **24** | 1 |
| Sides | 8 | 4 | 0 |

At the near end, the glass is confused with the floor: it is the geometric ambiguity
already described, above about one metre. At the far end, it is simply **missed**, and
often the model saw no contact there at all. The trajectory explains it: around a
contact on the back glass, the ball carries on in the picture along a smooth course,
without a turn. Thirty metres from the camera, the trip in depth to the glass and back
moves the ball by only a few pixels, while its rise or fall moves it by many more; its
apparent size, for its part, would vary by a third of a pixel. The bounce is almost
invisible to a broadcast camera.

Giving more weight to the walls in training, and accepting them earlier, was measured
on three sets of seeds:

| Cross-validation, mean of 3 seeds | Right / 863 | Score | Right glass / 135 |
|---|---|---|---|
| Model kept | 697 | 0.838 | 66 (49 %) |
| Walls weighted ×2, threshold 0.7 | 701 | 0.840 | 70 (52 %) |
| Walls weighted ×2, threshold 0.3 | 706 | 0.835 | 76 (56 %) |

The aggressive version finds about ten more glass contacts with each seed, but invents
as many: it is a trade, not a gain, and the score goes down. **The model was not
changed.** Finding these glass contacts would take another view (a second camera, or a
microphone) rather than another setting.

#### What the display was losing

Cross-validation scores what the model decides; the demonstration, and so the judges,
score what it shows. Between the two, a contact was discarded when the ball was
missing at its exact frame, for want of a position at which to light it. It is often
missing there: the display filter removes the vertex of a sharp turn as an outlier,
and a stroke hides the ball behind the racket. Over the eleven training minutes, each
predicted by a model that has not seen it, **51 decided contacts disappeared this way,
43 of them right**.

The ball is now taken at the nearest neighbouring frame; the instant of the contact
remains the one the model chose.

| Reach of the search | Right / 863 | Shown | Score |
|---|---|---|---|
| 0 frames (before) | 654 (75.8 %) | 750 | 0.811 |
| 1 frame | 696 (80.6 %) | 799 | 0.838 |
| **2 frames (kept)** | **697 (80.8 %)** | **801** | **0.838** |
| 4, 6 or 8 frames | 697 | 801 | 0.838 |

At two frames, the display shows exactly what the model decides: the figures are those
of the decoding, already measured on three sets of seeds. **The three judges above
were scored before this correction, on what was shown**: they underestimate the model
by what the display was losing.

#### The logic of padel to complete the picture: a negative result

What the picture does not show, the rules of the game could deduce: after a stroke,
the ball bounces once on the opponent's side before any glass; a player at the back
does not take it on the volley; a stroke does not follow a stroke from the same side.
The ground truth first says how predictable the game is. Between two successive
strokes, over the 1,579 marked contacts:

| Between two strokes | Share |
|---|---|
| Nothing: volley | 47 % |
| One bounce | 24 % |
| Bounce then glass | 16 % |
| Bounce then two glass contacts | 3 % |
| Other sequences | 10 % |

The same visible sequence therefore hides several real sequences. The rules were
crossed with what the table does not see (the side of the striker, their distance to
the net, the side and the depth of a bounce) and measured on the eleven training
minutes, each predicted by a model that has not seen it, display corrected. A deduced
contact counts as found if a really missed contact of the same kind lies between the
two contacts that frame it, without requiring the exact instant; nine successes out of
ten were needed to keep a rule.

| Rule | Contacts deduced | Really missed there | Score (0.838 without) |
|---|---|---|---|
| Bounce before a stroke taken more than 7 m from the net | 57 | 7 (12 %) | 0.815 |
| The same, at more than 9 m | 11 | 1 (9 %) | 0.832 |
| Bounce before a glass contact detected without a bounce since the stroke | 8 | 2 (25 %) | 0.836 |
| Back glass, when the player strikes closer to the net than the bounce | 16 | 5 (31 %) | 0.829 |

**None was kept.** The wrongly deduced bounces split into two causes. The first is in
the game: 25 out of 50 were real volleys, taken between 7 and 9 m from the net: the
bandeja and the víbora are played in the air, far from the net. The second is in the
inputs of the rule: 22 times, a stroke missed by the model came between the two, and
the sequence the rule reasons on was wrong from the start. The last 3 doubled a bounce
already shown.

The reverse rule, removing one of two consecutive strokes from the same side, would do
worse: out of 55 pairs of this kind, 38 are two real strokes. Between them, the
opposing stroke was missed 9 times, and 26 times there was none: one of the two
strokes is attributed to the wrong side. A wrongly attributed stroke makes two faulty
pairs, with the one before and the one after: these 26 pairs are 12 strokes, measured
further down.

The logic of the game therefore brings nothing the network does not already have: it
sees two seconds around each instant, which the sequence learned from the rally had
already shown.

#### Glass: physics, decoding, and a wider ensemble

A rule of logic fails because the game allows several sequences. A rule of physics
does not have this flaw. A ball arrives at a bounce at a speed given by the stroke
before and the bounce, and a bounce keeps well over half of it. When the player on
that side then strikes so close to the bounce that the ball, at that pace, would have
reached them several times over, it went somewhere else first: to the wall in its
line. Out of 151 bounces followed by a stroke from the same side, direct returns go
from 40 to 90 % of the arrival speed, detours stay under 35 %. In those cases, and
when the path by the wall remains possible in the time, a real undetected glass
contact lies between the bounce and the stroke **13 times out of 14**.

Knowing that a glass contact happened does not say when. The instant drawn from
physics rarely falls within three frames; the one where the network saw a wall as most
likely, even under its threshold, falls there far more often, and when this peak is on
the bounce, it is that the bounce was the glass. `contact/glass_inference.py` does
both.

Three other changes followed, each measured on the eleven training minutes, each
minute predicted by a model that has not seen it, on several sets of seeds:

| Chain | Right contacts / 863 | Right glass / 128 | Invented | Score |
|---|---|---|---|---|
| Model, display corrected | 696 to 697 | 64 to 66 | 54 to 56 | 0.838 to 0.839 |
| + deduced glass | 699 to 703 | 70 to 73 | 60 to 61 | 0.838 to 0.841 |
| + a contact of another kind at three frames | 708 to 713 | 72 to 75 | 62 to 66 | 0.840 to 0.847 |
| **+ 18 networks averaged instead of 3** | **718** | **75** | **57** | **0.855** |

- **Two contacts of different kinds can follow each other closely.** A bounce is often
  followed by the back glass five or six frames later, and the decoding kept a single
  peak every four frames. Two contacts of the same kind stay at four frames; a contact
  of another kind can come at three, above 0.85.
- **A wider ensemble invents less.** Three networks give 0.844 on average over six
  sets of seeds; nine, 0.847 and 0.856; eighteen, 0.855, with 57 invented contacts
  instead of 67 on average.
- **The side of the striker alternates.** Against the alternation of the marked
  strokes, 12 strokes out of 403 were attributed to the wrong half of the court, almost
  all of them a lob or a smash by the near player, who rises in the picture next to the
  far players. Three strokes in a row from the same half being impossible in a rally,
  the middle one goes to the nearest player of the other half: 6 errors out of 403.
  The glass gains nothing from it, the per-player statistics do.

What was tried without being kept:

| Trial | Result |
|---|---|
| Physics given to the network as cues, computed on the contacts of the rules | score 0.843 to 0.845 on 3 sets of seeds, glass 62 to 63: the bounces of the rules are too unreliable |
| Two passes: physics computed on the contacts of the model, read again by a second network (nested validation, 110 models) | glass 67 to 70, score 0.836 to 0.843: the same gain as the written rule, for two models |
| Weighting the walls ×1.5, ×2 or ×3 in training | +2 glass on average with 3 networks; with 9, 75 glass and 0.852 against 77 and 0.856 |
| Accepting the walls from 0.3 to 0.6 | up to 79 glass, as many more invented: a trade, not a gain |
| Realigning the instant of a glass contact on the sharpest turn of the trajectory | unchanged at best |
| A second glass contact when the path by the first remains too slow | 3 double glass contacts out of 52 cases, not separable |
| A glass contact between two opposing strokes with nothing detected between them | 2 glass contacts out of 70 cases: even slow, the ball is taken on the volley |
| The striker chosen by the geometry of the ray rather than by the picture | 11 to 25 % of errors of half, against 3 % |

**What remains.** At the near end, 50 glass contacts out of 78 are right, 9 out of 10
on the sides. At the far end, 22 glass contacts out of 38 remain missed: it is the
limit of the camera described above. Elsewhere, the glass contacts are often found at
the wrong instant rather than missed: 17 are detected between four and eight frames
from the marked instant, against 7 bounces and 5 strokes on much larger counts, and
without a bias one way or the other. The exact instant of a contact against the glass
is perhaps also the hardest to mark by hand. These figures are those of the training
minutes: the judges were scored before all these changes.

#### The judges' verdict, after these changes

Everything above was decided on the eleven training minutes. The chain frozen in this
way was then scored only once on the nine judge minutes. They had already served for
the first verdict, but none of the changes was chosen by looking at them:

| Contacts shown with the right surface | Judge 1 (239) | Judge 2 (238) | Judge 3 (239) | Three judges (716) |
|---|---|---|---|---|
| First verdict | 78.7 % | 79.0 % | 75.3 % | 77.7 % |
| **Current chain** | **83.7 %** | **82.8 %** | **84.1 %** | **83.5 %** |

All three judges improve, by 3.8 to 8.8 points. Over the 716 contacts: strokes right
at 93.2 % (355 out of 381), bounces at 81.4 % (180 out of 221), **glass at 61.4 % (62
out of 101)**, like the 59 % of the training minutes. The contacts shown are real at
93 %, 92.5 % and 95 % depending on the judge. The net remains missed (0 out of 9): too
rare to be learned.

These minutes had already been used. Three new minutes were therefore marked
afterwards, without seeing anything of what the chain detects, and scored only once:

| Fourth judge (247 contacts) | FinalF 32000 | FinalM 16000 | FinalM 45000 | Total |
|---|---|---|---|---|
| Right surface | 70 / 81 | 66 / 76 | 69 / 90 | **205 / 247 (83.0 %)** |

Strokes at 92.8 %, bounces at 81.3 %, **glass at 62.8 % (27 out of 43)**: the figures
of the first three judges, to within a point. 217 of the 224 contacts shown are real
(96.9 %). By wall, the side glass contacts are all found (5 out of 5), those of the
near end at 12 out of 21, those of the far end at 10 out of 17.

#### The judges become data

A judge that has given its verdict can no longer judge, but its hand-marked contacts
remain examples. The twelve judge minutes joined the training: 23 minutes, 1,826
contacts, 272 glass contacts instead of 128. The cross-validation covers the 23
minutes, each predicted by 18 networks trained on the 22 others, and compares on the
same minutes the model trained on the eleven original minutes:

| Cross-validation, 1,826 contacts | 11 training minutes | 22 training minutes |
|---|---|---|
| Right surface | 1,521 (83.3 %) | **1,591 (87.1 %)** |
| Glass | 164 / 272 (60.3 %) | **184 / 272 (67.6 %)** |
| Bounces | 467 / 568 (82.2 %) | 490 / 568 (86.3 %) |
| Strokes | 884 / 951 (93.0 %) | 905 / 951 (95.2 %) |
| Invented contacts | 108 | 98 |
| Score | 0.859 | **0.888** |

The gain is the same on the eleven original training minutes (83.2 → 87.1 %) and on
the twelve judge minutes (83.4 → 87.1 %). The learning curve, flat for the first model
between eight and ten minutes, is no longer flat for the current chain: doubling the
data is worth nearly four points, and seven on the glass. Minute by minute, 21 of the
23 minutes improve, by 1 to 7 contacts; two do not move, one falls back by 3.

A fifth judge was then marked, three minutes never looked at, and scored only once
with both models, both frozen beforehand:

| Fifth judge (253 contacts) | FinalF 5000 | FinalF 23000 | FinalM 33000 | Total | Glass |
|---|---|---|---|---|---|
| Model trained on 11 minutes | 69 / 83 | 70 / 79 | 68 / 91 | 207 (81.8 %) | 24 / 38 |
| Model trained on 23 minutes | 69 / 83 | 71 / 79 | 68 / 91 | 208 (82.2 %) | 25 / 38 |

**The judge does not confirm the gain**: one more contact, where the cross-validation
announced about ten. Three minutes at 0, +1 and 0 exist in the cross-validation, but
drawing all three is unlikely, of the order of 2 %. Either adding the judges helps less
than it says, or this judge fell on minutes where it changes nothing; one clue points
the first way: the three minutes of the fourth judge, also marked later, gained less
than the others (+2.7 against +3.5 on average). The model shipped remains that of the
23 minutes, never worse on the new minutes; **the figure to keep for a minute never
seen is that of the fresh judges, 82 to 83 %**, and not that of the cross-validation.

#### Corners, a known limit

Until 29 September, the marking tool replaced any mark placed within two frames of
another. A ball that touches two glass panels in a corner, from one frame to the next,
kept only one: the minutes marked before count 12.0 glass contacts per minute, the
last two judges 13 to 14. The tool now accepts two neighbouring contacts. The review
of the glass contacts marked near a corner was done on three minutes out of nineteen:
4 second glass contacts added for 19 moments reviewed, about one per minute. The
model, for its part, never shows two contacts of the same kind within five frames: the
second glass of a corner is always missed. Learning it would be worth less than one
contact per minute, below the spread from one batch of minutes to another, and would
need a new judge to prove it; it was not done.

#### Cutting a match into rallies

The video of the dataset keeps the rallies and cuts the dead time: a new point opens
at the cut. A cut can be read in the picture itself: two successive frames from
different shots differ everywhere, while in play only the players and the ball move:
the mean grey-level gap between two thumbnails is 0.5 in the median during play, and
from 5.6 to 11.5 at the cuts (`io/splices.py`).

The truth already exists: the dataset marks the serves over the first 20,100 frames of
each final. An announced start of a rally is right if it falls within two seconds of a
serve. The minutes that trained the contact model tuned the rule, the six others of
the annotated zone judged it once.

| | Serves found | Starts announced | Precision | Recall |
|---|---|---|---|---|
| Tuning, 5 minutes | 16 / 17 | 18 | 89 % | 94 % |
| **Judge, 6 minutes** | **14 / 16** | **15** | **93 %** | **88 %** |

The simplest rule was the best: a cut opens a rally. Requiring a stroke after the cut
changed nothing; opening a point after a long silence without a contact added more
false starts than it recovered serves filmed without a cut: those are the missed
serves. The cut threshold is stable between 3 and 4, and loses half the points at 5.

#### Reading the score off the scoreboard

The scoreboard of the broadcast is at a fixed place, one row per pair: the names, one
column per set, then a light cell for the points. This cell is the landmark: it is the
only light one (white, or golden at the golden point) and it shifts by one column at
each set, which gives the current set; the games are in the dark cell to its left, and
the serving pair carries a yellow dot (`io/scoreboard.py`). Eleven values are enough
(0, 15, 30, 40 and the games from 0 to 6) and the typeface never changes: each cell is
compared with templates, without a text recognition engine. The templates are drawn
from frames listed with their value (`ground_truth/scoreboard/templates.json`) and
recomputed from the video, which the repository does not contain.

The scoreboard is read at the start of each sequence between two cuts, and the grammar
of the score says who won the point: a single step from 0 to 15, 30, 40, or a game won
with the points reset to zero (`analytics/points.py`).

| | Sequences | Scoreboard read | Points attributed | Changes rejected |
|---|---|---|---|---|
| Women's final | 63 | 54 | 48 | 5 |
| Men's final | 141 | 83 | 69 | 7 |

Two checks. **By eye, 48 scoreboards drawn at random, 24 per final: all 48 readings
are right**, games, points and serve. **By the grammar**: the twelve rejected changes
all have several points of gap between two readings: a point absent from the video, or
played during a sequence where the scoreboard was hidden. None is a wrong reading: the
rule refuses to guess. Two facts come out in passing: no women's sequence repeats the
same score, which confirms that a cut does open a point; and the scoreboard is missing
on two men's sequences out of five, slow motions and close-ups included.

**From the pair's point to the player's point.** The statistics video credits each
point to a player: the winning pair comes from the scoreboard, its side of the court
from the serving pair (the yellow dot) and from the first stroke of the rally, which is
the serve; the last striker of the rally receives a **winner** if they are of the
winning pair, an **error** otherwise. This split has no ground truth: the contact marks
do not say who strikes. It inherits the strokes missed by the model, about one in ten,
which give the point to the wrong player; the video says so under the panel.

### Another tournament, with sound

Everything above is measured on two matches of one tournament, filmed by the same
camera. A second public dataset, published with the paper by Decorte et al.
(*Multi-Modal Hit Detection and Positional Analysis in Padel Competitions*, CVPR
Workshops 2024), offers other tournaments of the tour, filmed from the same place but
in other venues, at 25 frames per second instead of 30, **and with sound**. It only
annotates stroke windows. One of its rallies, `20230528_VIGO_11`, one minute of play,
served as a test: no video or image is versioned, only its calibration and its
hand-marked contacts.

**The calibration** of a new court takes ten minutes: 3.4 px of error on the ground
(5.5 cm), 3.7 px in the median for the camera pose, better than the 5.8 px of the
women's final.

**The chain, without retraining anything**, scored against the 88 contacts of the
rally (46 strokes, that is the 46 windows of the authors, then 25 bounces, 15 glass
contacts, 2 mesh contacts):

| VIGO_11, 88 contacts | Model trained on 11 minutes | On 23 minutes |
|---|---|---|
| Right surface | 68 (77 %) | **76 (86 %)** |
| Strokes | 41 / 46 | 43 / 46 |
| Bounces | 19 / 25 | 21 / 25 |
| Glass | 8 / 15 | **12 / 15** |
| Invented | 2 | 1 |

Another court, another light and another frame rate do not throw the chain off. On a
single rally, the gap between the two models is a trend, not a measurement.

**Sound.** A ball impact is a very brief burst of energy in the high frequencies,
which the voice and the audience do not have: the spectral flux between 2 and 12 kHz,
compared with its sliding median, finds 44 of the 46 annotated strokes. But not every
contact is heard as much:

| Median strength of the sound peak | Strokes | Glass | Bounces |
|---|---|---|---|
| | **48** | 6 | 4 |
| Heard above 8 | 43 / 46 | 6 / 15 | 4 / 25 |

At this level of 8, the rally has 46 other peaks without any contact: shoes, voices,
echoes of the strokes. The timbre separates the stroke from the rest (41 out of 46),
not the bounce from the glass. Of the twelve remaining errors of the chain, sound
would have recovered two or three strokes; **the three missed glass contacts are
silent** in the recording. The broadcast microphone does not pick up what the camera
does not see: sound would help the strokes, already found at 93 %, and not the back
glass.

### The report of a whole match

Both finals were analysed in full, minute by minute (`scripts/analyse_match.py`, three
and a half hours on a GTX 1650), then assembled into a report by pair
(`scripts/match_stats.py`). Three difficulties did not exist at the scale of a rally;
each is measured.

**Following the players from start to finish.** The tracking starts from zero at each
analysed minute. Replayed in one go over the match, against the identity truth:

| IDF1 over the whole match | Minute by minute | In one go |
|---|---|---|
| Women's final | 0.615 (80 identity changes) | **0.791** (30) |
| Men's final | 0.662 (159) | **0.639** (87) |

Almost all the remaining confusions are between partners. Their effect on the figures
is measured by applying the same computation to the annotated positions of the
dataset, segment by segment between two changes of ends:

| Gap from the truth, by segment | Women | Men |
|---|---|---|
| Distance of a **player**: median, 1 in 10, worst | 2.3 %, 12.5 %, 28 % | 3.2 %, 11.9 %, 19 % |
| Time at the net of a **player**: median, 1 in 10 | 1.6 pt, 5.4 pt | 5.3 pt, 9.2 pt |
| Distance of a **pair**: median, worst | 0.5 %, 2.4 % | 0.7 %, 6.0 % |

When the tracking confuses two partners, the metres of one go to the other, and the
sum of the pair is not changed by it. **The report is therefore given by pair**; a
per-player figure that is wrong by more than 12 % once in ten is not published.

**Knowing which pair plays where.** The teams change ends after the first, the third
and every odd game of a set, and the scoreboard says how many games have been played
(`analytics/sides.py`). Against the identity truth, this rule finds every change of
ends after the first reading of the scoreboard, to the very frame: 6 out of 6 for the
women, 7 out of 7 for the men. It finds an eighth one for the men, at frame 32,137,
which the ground truth did not have: checked in the picture, the pair in black is on
the camera side at frame 31,900 and the one in light blue at frame 32,287. The truth
had missed this change. The serve, the first stroke of each rally, then says which of
the two rows of the scoreboard plays on which side: 50 serves out of 54 and 74 out of
80 vote for the same orientation, the others being strikers attributed to the wrong
half. The change of ends that precedes the first reading escapes the rule; the report
therefore starts at that reading, that is 23 minutes of play out of 25 for the women
and 29 out of 30 for the men. The colour of the shirts was tried to recover it, and to
check the others: seen from behind in the foreground and from the front at the far
end, the same outfit does not give the same colour, and it sees only 2 changes for the
women against 18, most of them false, for the men.

**What the figures of the report are worth:**

| Statistic | Checked against | Gap |
|---|---|---|
| Points won | the scoreboard | exact: 48 and 69 points that the scoreboard settles |
| Distance of a pair, whole match | the annotated positions | +0.3 % to +1.3 % |
| Time at the net of a pair | the annotated positions | 0.3 point at most |
| Strokes | 26 marked minutes, 1,077 strokes | +1.2 % (2.6 % per minute, in the median) |
| Volleys | the same | +7.2 % |
| Strokes after a bounce | the same | −4.9 % |
| Strokes after the glass | the same | −5.8 % (14 % per minute) |
| Rallies of 1-3, 4-7, 8 shots and more | the same | 14, 25, 52 against 15, 24, 53 |

The strokes are counted at each minute by a model that has not seen it. A missed
bounce turns a stroke after a bounce into a volley: the split between the two leans by
a few per cent towards the volleys, always in the same direction. The winners and
errors per player, credited to the last striker, have no ground truth and stay in the
statistics video, outside the report.

![Report by pair for both finals](../figures/pair_report.png)

![Points won by rally length](../figures/points_by_length.png)

![Court occupancy by pair](../figures/pair_occupancy.png)

## Parameter notes

The measurements behind individual parameters, where the sections above do not already
give them.

**Ball candidates.** The annotated balls run from 4 to 29 px a side: a blob area capped
at 900 px² keeps the largest and refuses a limb.

**Greedy trajectory.** Over free play, the distance between the real position and the
constant-velocity prediction has a median of 5.4 px and a 90th percentile of 28.5 px: a
gate of 30 px covers 91 % of it. The 95th percentile of the real displacement between
two frames is 52.7 px, which bounds the step between the two seed frames. Over 814
annotated arcs, the ball touches something every 15 frames in median and never goes more
than 64 without: a cap of 60 frames keeps 99.8 % of the real arcs. It travels 14.4 px a
frame in median, and a floor of 6 keeps 85 % of them. The ball being the second of its
list in median, each seed frame offers several candidates.

**Global path.** Once the candidates inside player boxes are demoted, the ball sits in
the top ten 96 % of the time, which sets the width. The ceiling on the acceleration
cost, swept from 40 px to infinity, moves the recall by a thousandth. The weight of the
candidate score has an interior optimum: the recall runs 0.739, 0.764, 0.748 and 0.712
at 120, 240, 480 and 960. Above an absence cost of about a thousand the recall
saturates: the path no longer gives up at all.

**Displayed trajectory.** On an annotated minute, the path read with the network's
scores as they are, then filtered by confidence, gives 1,374 right positions, 87 wrong
and 2 phantoms, against 1,174, 199 and 27 for the relative path tuned for recall.
Bridging three missing frames while smoothing raises the wrong positions from 194 to
261, so no gap is bridged. A process noise of 100 takes the displayed jerk from 6.0 to
4.5 px with no loss of accuracy; at 4 it falls to 1.8 px, but the wrong positions rise
from 199 to 280, every undetected bend being rounded off.

**Contacts of the rule chain.** Searching the contacts on a smoothed trajectory removes
2 false contacts and 3 unjudged ones for 1 true one lost, on the annotated minute of the
tuning match; a stronger smoothing loses 7. The contacts whose ray meets a single
surface carry 38 of the 42 false walls of the two annotated matches, for 2 true walls
out of 33. The wrist nearest the ball moves at 26 px per frame in median at a marked
stroke and at 5 px where nothing happens; the strokes the turn criterion missed move
just as fast, at 29. Reading them from the gesture adds 11 right contacts on the tuning
match and 9 on the held-out one, tuning frozen.

**Top speed.** A speed held for one second. Any shorter, a small jump or an identity
swap between partners passes for a sprint: on two rallies, 21 and 50 km/h with a median
over 5 frames, 15 and 19 over 31.

**Tracker bounds.** Without a bound on the court extent, about one position in a hundred
landed several metres past the glass. The person held for twenty seconds on the women's
final sat 1.7 m behind the far glass. The jump that marks a splice gives the same result
between 0.6 and 1 m per frame; at 1.5 m some splices go unseen.

**Cuts in the identity truth.** A player covering more than a metre between two frames
marks a cut, where the 99th percentile of the movement between two frames is 0.43 m. One
player is enough, since two partners who exchange places across a splice barely move in
the measurement, at the price of clips raised by a lone annotation glitch: nine in the
women's match, twenty-nine in the men's. Across a hole in the annotations the allowance
grows with the time elapsed, at 6 m/s where the measured peaks sit near 3.9.

**Splices.** The mean grey-level difference between two frames stays under about 2.3
during play; the threshold is 4.

**Surface sample.** Arbitrating the 886 contacts of the men's match would take two
hours. A stratified sample is enough to separate 0.83 from 0.75, and its seed is
recorded.

**Training throughput.** The training reads 10.7 frames/s against 15.1 in synthetic:
JPEG decoding is the bottleneck, each frame being read three times, once per position in
the stack.

## Known limitations

**A slot could follow someone behind the back glass.** The tracking tolerated four
metres of overflow in both directions, to let through a player who goes out by a side
opening. But these openings are on the sides: behind a back glass, across the width of
the court, there is only the audience and the staff. Over the twenty minutes analysed,
**3,783 frames out of 36,000** had a "player" placed there: on one, a slot followed
for twenty seconds a person sitting behind the back wall while the real player was not
tracked. These positions are now rejected, and any position outside the court is
penalised, so that a person on the court is always preferred to a person beside it.
The statistics video replays the corrected tracking. The tactical statistics and the
identity measurements of this report were made before this fix, and are to be redone
on the whole match.

**What the fix changed, and what it did not solve.** The identity campaign redoes the
detection of the players over the whole match, hours per trial. A lighter bench
replays the tracking on the minutes already analysed (`scripts/identity_bench.py`):
twenty minutes for tuning, and eight new minutes, four per final, to judge only once:
the identity truth covering both matches, they required no marking.

| Judge, 8 new minutes | IDF1 | Identity changes |
|---|---|---|
| Original tracking | 0.811 | 26 |
| Rejection behind the back glass | **0.841** | **22** |
| of which women's final | 0.812 → **0.872** | 10 → 6 |
| of which men's final | 0.809 → 0.809 | 16 → 16 |

Almost all the remaining changes happen **at the cuts** of the video, where the
players reappear elsewhere, with partners three to six metres apart; tight crossings
explain only a handful. Three ideas were tried on the tuning minutes:

- **erasing the velocity that a cut leaves behind**. Measured across a cut, it is that
  of a teleportation: +5.6 points of IDF1 on the men's tuning minutes, **nothing on
  the judge**. Kept, because it is right and costs nothing, but without a demonstrated
  gain;
- **the usual side of each partner** (the drive player on the right, the backhand one
  on the left): held in 94.8 % of the cuts for the women but 83.7 % for the men. Added
  to the motion, it gained only 0.005 of IDF1 over the twenty minutes: noise, **not
  kept**;
- **the colour of the head and of the lower body**, since the partners wear the same
  shirt: it finds the right pair in only 76 % of the cases, and changed nothing in the
  tracking. **Not kept.**

The men's identity tracking therefore remains the weak point of the per-player
statistics.


**A player cannot be followed across a change of ends.** The four slots stand for
halves of the court, and the tracking rejects by construction an observation on the
wrong side of the net: that is what gives it its "0 frames above four". The price of
this constraint is that a player who changes ends changes slot. Nothing in the picture
would allow them to be linked up: the pipeline reads neither faces nor numbers. The
per-slot statistics remain valid over the whole match; the **per-player** statistics
are only valid within a segment between two changes of ends.

**Identity tracking does not generalise as well as detection.** On the tuning match it
loses identity 4 times in 21 opportunities, that is 19 %. On the held-out match, 28
times in 38 opportunities, that is **74 %**. Detection, for its part, transfers
without loss, and so does localisation. A pipeline judged on its detection F1 alone
would seem to generalise; it only generalises on half of what it does.

**The weak point is the tight crossing between partners**, not the cut. Over the same
number of frames, the men's match has 14 close approaches against 3, and its partners
come down to 0.41 m from each other against 0.56 m. That is where the gap between 19 %
and 74 % is decided, and it is the lead to work on first.

Measuring all this cost 266 clips of human arbitration. Of the 47 clips of the men's
final whose answer was traced, 11 carried a real switch: **nearly one cut in four
makes identity slip**.

**The identity ground truth depends on a human judgement that cannot be reproduced.**
The 266 arbitrations were given by a single person, without a second annotator, so
without an inter-annotator agreement to report. The seven changes of ends of the
women's final were however confirmed by two independent routes: the outfit of the
teams sampled over the whole match, and the scoreboard on the doubtful passage.

**The incomplete frames are detection failures**, not tracking failures: they are
exactly those where the model finds only three people, always at the far end of the
court, when two adjacent players hide each other.

**The side walls are not modelled.** Only the back walls are; the stepped geometry of
the sides needs checking against the FIP rules.

**Smoothing does not remove all the noise.** The gap between raw distance and smoothed
distance says what smoothing removed, not what remains. The smoothed distances
correspond to about 88 metres per minute of **effective play**: the video being edited
down to the rallies, it contains no dead time. This figure is therefore not directly
comparable with the distances per match reported in the literature, which include the
interruptions.

**The net threshold is a convention, although a measured one.** The trough between the
two modes is real but wide: the absolute control percentages move by fifteen points
depending on where it is placed in that trough. The ratio between the two pairs, for
its part, is stable: that is the form to quote.

## Reproducing the evaluation

The tracking metrics add two dependencies, kept separate because they are only used
for measuring:

```bash
pip install -e ".[eval]"
```

The dataset does not provide identity. It has to be rebuilt, then the moments where
the reconstruction is doubtful have to be arbitrated by hand.

```bash
python scripts/build_identity_truth.py --annotations <pose.json> \
    --calibration ground_truth/calibrations/<name>.json --out ground_truth/identity/<name>.json

python scripts/detect_cuts.py --annotations <pose.json> \
    --calibration ground_truth/calibrations/<name>.json --identity ground_truth/identity/<name>.json

python scripts/review_identity.py --video <video.mp4> \
    --annotations <pose.json> --identity ground_truth/identity/<name>.json
```

The first associates by nearest neighbour over the whole match and lists the doubtful
close approaches; the second adds the broadcast cuts, which proximity does not see;
the third replays each doubtful moment in a loop, the players framed in the colour of
their slot.

The question asked is not "did they cross" but **"does the same player wear the same
colour before and after"**.

| Key | On a close approach | On a cut |
|---|---|---|
| `n` | no switch | no switch |
| `s` | switch | not applicable |
| `p` / `e` / `b` | not applicable | the near pair, the far pair, or both have switched |
| `c` | not applicable | the teams have changed ends |
| `r` | go back to the previous clip and cancel its answer | same |
| `q` | quit, keeping the answers given | same |

The file is rewritten after each answer, atomically: a power cut costs the clip in
progress, not the whole arbitration.

The campaign then computes detection, localisation and both ablations in a single pass
over the video:

```bash
python scripts/run_evaluation.py --video <video.mp4> --annotations <pose.json> \
    --calibration ground_truth/calibrations/<name>.json --identity ground_truth/identity/<name>.json \
    --out outputs/<name>_eval.json --frames 9000
```

The ball is measured separately, both trajectory methods being computed in a single
pass over the requested range:

```bash
python scripts/measure_trajectory.py --video <video.mp4> --annotations <ball.json>     --start 0 --stop 21472 --out outputs/<name>_trajectory.json
```

The contacts are measured in the same way, on the reconstructed path and on the
annotated ball in a single pass:

```bash
python scripts/measure_contacts.py --video <video.mp4> --annotations <ball.json>     --shots <shots.csv> --identity ground_truth/identity/<name>.json     --start 0 --stop 20099 --out outputs/<name>_contacts.json
```

The surfaces need a ground truth that does not exist: it is produced by hand. The
first script draws up the list of contacts to judge, the second replays them one by
one.

```bash
python scripts/build_surface_tasks.py --annotations <ball.json> --poses <pose.json>     --calibration ground_truth/calibrations/<name>.json --start 16000 --stop 20099     --video <name> --out ground_truth/surfaces/<name>.json

python scripts/review_surfaces.py --video <video.mp4>     --annotations <ball.json> --truth ground_truth/surfaces/<name>.json
```

| Key | Answer |
|---|---|
| `s` `v` `g` `t` | the floor, a glass panel, the mesh, the net |
| `f` | a stroke, so a racket |
| `n` | no contact: the trajectory goes straight through |
| `x` | unreadable, I cannot decide |
| `r` / `q` | go back to the previous clip / quit, keeping the answers |

`n` and `x` do not say the same thing and are never added together. `x` is a
non-measurement; `n` is an observed false positive of the contact stage.

The ball detection network is trained on a cache of reduced frames, built outside the
evaluation slice. An interrupted session is resumed with `--resume`:

```bash
python scripts/build_frame_cache.py --video <video.mp4> --annotations <ball.json>     --exclude 16000 20099 --step 1 --out cache/<name>

python scripts/train_ball_net.py --cache cache/<name> --epochs 10 --out weights/ball_net

python scripts/measure_trajectory.py --video <video.mp4> --annotations <ball.json>     --start 16000 --stop 20099 --spacing 3 --weights weights/ball_net_best.pt     --out outputs/<name>_trajectory_net.json
```

`--spacing 3` is mandatory with `--weights`: the network was trained on frames spaced
three apart, and the script refuses any other gap rather than silently returning an
empty list of candidates.

The camera pose is checked on the references it never fitted:

```bash
python scripts/check_camera_pose.py --calibration ground_truth/calibrations/<name>.json
```

## What this repository versions

No image, no video, no model weights. `data/`, where the downloaded dataset lands, is
excluded as a whole and without exception.

`ground_truth/` on the other hand is versioned, because without it the figures of the
[Evaluation](#evaluation) section would not be reproducible:

| File | Content |
|---|---|
| `calibrations/*.json` | 23 clicked points per video: 13 on the ground including 4 control points, and 10 above the ground including 8 control points |
| `identity/*.json` | assignment of the 4 slots over the whole match, list of the doubtful moments, and the 266 human arbitrations |
| `surfaces/*.json` | the 194 contacts to judge and the 194 judgements given |
| `contact_marks/*.json` | all the contacts of twenty minutes marked by hand, with their surface |

**`surfaces/` and `contact_marks/` are the only ones of these files that derive from nothing.** Contact
surfaces are not labelled in any public padel dataset: these files are the measurement
itself, and without them the section on surfaces would only be a rule without a judge.
The points above the ground in `calibrations/` are in the same case: they are taken by
hand on the wall panels, and without them the camera pose could not be solved.

These files derive from the annotations of the dataset, under CC-BY-4.0, and contain
none of its image data. With them, reproducing the evaluation takes downloading the
public dataset and launching the campaign, not redoing the arbitration.

**The three calibrations are identical**, and that is intentional. Both matches are
filmed from the same position at the same tournament, and the trial excerpt is taken
from the women's final. The calibration fitted on the latter was transferred to the
other two then checked by overlaying the court model on a frame of each: outline,
service lines, centre line and net fall right. A single set of clicked points
therefore covers the whole dataset.
