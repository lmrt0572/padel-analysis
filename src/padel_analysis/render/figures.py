"""The figures of the README and of the evaluation report, in a dark style.

Each function takes plain data and writes one PNG. None of them draws a video frame.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import figure_style as style

SURFACE_ORDER = ("raquette", "sol", "verre", "grillage", "filet", "aucun")


def _save(figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    import matplotlib.pyplot as plt

    plt.close(figure)


def _share(counts: dict) -> float:
    return 100.0 * counts["right"] / counts["real"]


def judges_chart(verdicts: dict, cv: dict | None, path: Path) -> None:
    """Draw rules against the learned model: cross-validation, each judge, all judges."""
    style.use()
    import matplotlib.pyplot as plt

    groups, rules, model = [], [], []
    if cv is not None:
        groups.append("Cross-\nvalidation")
        rules.append(_share(cv["regles"]))
        model.append(_share(cv["modele"]))
    for judge in verdicts["judges"]:
        groups.append(judge["name"])
        rules.append(_share(judge["rules"]))
        model.append(_share(judge["model"]))
    pooled = {kind: {"right": sum(j[kind]["right"] for j in verdicts["judges"]),
                     "real": sum(j[kind]["real"] for j in verdicts["judges"])}
              for kind in ("rules", "model")}
    groups.append(f"All judges\n({pooled['model']['real']} contacts)")
    rules.append(_share(pooled["rules"]))
    model.append(_share(pooled["model"]))

    figure, axis = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(groups))
    for offset, values, colour, label in ((-0.2, rules, style.RULES, "Hand-tuned rules"),
                                          (0.2, model, style.MODEL, "Learned model")):
        bars = axis.bar(x + offset, values, 0.38, color=colour, label=label, zorder=2)
        for bar, value in zip(bars, values, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.0f} %",
                      ha="center", color=style.TEXT, fontsize=9)
    axis.set_xticks(x, groups)
    axis.set_ylim(0, 100)
    axis.set_ylabel("real contacts given the right surface (%)")
    axis.set_title("Contacts and surfaces, on what the video shows")
    axis.legend(loc="upper left", ncol=2)
    axis.grid(axis="x", visible=False)
    _save(figure, path)


def learning_curve_chart(curve: Sequence[dict], path: Path) -> None:
    """Draw how the model improves with the number of hand-marked minutes."""
    style.use()
    import matplotlib.pyplot as plt

    sizes = [point["minutes"] for point in curve]
    shares = [_share(point) for point in curve]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(sizes, shares, color=style.MODEL, marker="o", linewidth=2.2, zorder=2)
    for size, share in zip(sizes, shares, strict=True):
        axis.annotate(f"{share:.1f} %", (size, share),
                      textcoords="offset points",
                      xytext=(0, 9), ha="center", fontsize=9)
    axis.set_xlabel("match minutes marked by hand for training")
    axis.set_ylabel("right surface (%), cross-validation")
    axis.set_xticks(sizes)
    axis.set_ylim(min(shares) - 6, max(shares) + 5)
    axis.set_title("Beyond eight minutes, marking more adds almost nothing")
    _save(figure, path)


def confusion_chart(confusion: Sequence[dict], path: Path) -> None:
    """Draw, for each true surface, what the model answered, including nothing at all."""
    style.use()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    present = {row["marked"] for row in confusion} | {row["detected"] for row in confusion}
    labels = [kind for kind in SURFACE_ORDER if kind in present]
    index = {kind: i for i, kind in enumerate(labels)}
    grid = np.zeros((len(labels), len(labels)))
    for row in confusion:
        grid[index[row["marked"]], index[row["detected"]]] += row["count"]
    totals = grid.sum(axis=1, keepdims=True)
    shares = np.divide(grid, totals, out=np.zeros_like(grid), where=totals > 0)

    figure, axis = plt.subplots(figsize=(6.6, 5.6))
    colours = LinearSegmentedColormap.from_list("vert", [style.PANEL, style.MODEL])
    axis.imshow(shares, cmap=colours, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if grid[i, j] == 0:
                continue
            axis.text(j, i, f"{int(grid[i, j])}", ha="center", va="center", fontsize=9,
                      color=style.BACKGROUND if shares[i, j] > 0.55 else style.TEXT)
    names = [style.KIND_NAMES[kind] for kind in labels]
    axis.set_xticks(range(len(labels)), names)
    axis.set_yticks(range(len(labels)), names)
    axis.set_xlabel("the model's answer")
    axis.set_ylabel("truth, marked by hand")
    axis.set_title("What is taken for what")
    axis.grid(False)
    _save(figure, path)


def heatmaps_chart(positions: dict, court, path: Path) -> None:
    """Draw where each of the four players stood, over the whole match."""
    from ..analytics.heatmap import occupancy_grid

    style.use()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(positions), figsize=(3.2 * len(positions), 6.4))
    for axis, (slot, track) in zip(np.atleast_1d(axes), positions.items(), strict=True):
        grid, extent = occupancy_grid(track, court)
        style.draw_court(axis, court.half_width, court.half_length,
                         court.service_line_distance)
        axis.imshow(np.ma.masked_equal(grid, 0), origin="lower", extent=extent,
                    cmap="magma", alpha=0.85, zorder=2)
        axis.set_title(slot, color=style.PLAYER.get(slot, style.TEXT))
    figure.suptitle("Court occupancy, whole match", color=style.TEXT)
    _save(figure, path)


def net_control_chart(depths: np.ndarray, threshold: float, control: dict, path: Path) -> None:
    """Draw the two depths players hold, the threshold between them, and who held the net."""
    style.use()
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2),
                                         gridspec_kw={"width_ratios": [2.2, 1]})
    left.hist(depths[~np.isnan(depths)], bins=50, range=(0, 10), color=style.KIND["verre"],
              zorder=2)
    left.axvline(threshold, color=style.KIND["filet"], linewidth=2,
                 label=f"threshold {threshold} m, in the trough between the two modes")
    left.set_xlabel("distance to the net (m)")
    left.set_ylabel("positions")
    left.set_title("Two depths: at the net, or at the back")
    left.legend()

    labels = ["Near pair", "Far pair", "Contested"]
    values = [control["near_percent"], control["far_percent"], control["contested_percent"]]
    colours = [style.PLAYER["near_1"], style.PLAYER["far_1"], style.MUTED]
    bars = right.bar(labels, values, color=colours, zorder=2)
    for bar, value in zip(bars, values, strict=True):
        right.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.0f} %",
                   ha="center", color=style.TEXT, fontsize=9)
    right.set_ylabel("playing time (%)")
    right.set_title("Who holds the net")
    right.grid(axis="x", visible=False)
    _save(figure, path)


PAIR_COLOURS = ("#3987e5", "#d95926")
"""The two pairs, top row of the scoreboard then bottom."""
DUEL_ROWS = (
    ("points_won", "Points won", "exact: read off the scoreboard"),
    ("strikes", "Strokes", "± 1 %"),
    ("volleys", "Volleys", "± 7 %"),
    ("after_glass", "Strokes after the glass", "± 6 %"),
    ("distance_m", "Distance covered", "± 1 %"),
)
MATCH_NAMES = {"FinalF": "Women's final", "FinalM": "Men's final"}


def _pair_legend(axis, names) -> None:
    from matplotlib.patches import Patch

    axis.legend(handles=[Patch(color=c, label=n) for c, n in zip(PAIR_COLOURS, names,
                                                                  strict=False)],
                loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=10,
                labelcolor=style.TEXT, handlelength=1.2)


def pair_duel_chart(reports: Sequence[dict], path: Path) -> None:
    """Draw each pair's share of the points, strikes, volleys, glass shots and distance."""
    style.use()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(reports), 1, figsize=(10.5, 3.3 * len(reports)))
    for axis, report in zip(np.atleast_1d(axes), reports, strict=True):
        names = list(report["pairs"])
        first, second = (report["pairs"][n] for n in names)
        for row, (key, label, reliability) in enumerate(DUEL_ROWS):
            a, b = first[key], second[key]
            share = a / (a + b) if a + b else 0.5
            y = len(DUEL_ROWS) - 1 - row
            axis.barh(y, share, color=PAIR_COLOURS[0], height=0.62, edgecolor=style.PANEL,
                      linewidth=2)
            axis.barh(y, 1 - share, left=share, color=PAIR_COLOURS[1], height=0.62,
                      edgecolor=style.PANEL, linewidth=2)
            unit = " m" if key == "distance_m" else ""
            axis.text(0.015, y, f"{a:,.0f}{unit}", va="center",
                      color=style.TEXT, fontsize=10, fontweight="bold")
            axis.text(0.985, y, f"{b:,.0f}{unit}", va="center", ha="right",
                      color=style.TEXT, fontsize=10, fontweight="bold")
            axis.text(1.02, y, reliability, va="center", color=style.MUTED, fontsize=9,
                      transform=axis.get_yaxis_transform())
        axis.axvline(0.5, color=style.MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
        axis.set_yticks(range(len(DUEL_ROWS)))
        axis.set_yticklabels([label for _, label, _ in reversed(DUEL_ROWS)], color=style.TEXT)
        axis.set_xlim(0, 1)
        axis.set_xticks([])
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        net = " · ".join(f"{n} {100 * report['pairs'][n]['net_share']:.0f} %" for n in names)
        axis.set_title(f"{MATCH_NAMES.get(report['match'], report['match'])}, "
                       f"{report['minutes']:.0f} min of play, time at the net {net} (± 0.3 pt)",
                       loc="left", pad=28, fontsize=11)
        _pair_legend(axis, names)
    _save(figure, path)


def points_by_length_chart(reports: Sequence[dict], path: Path) -> None:
    """Draw the points each pair won, by the number of strikes in the rally."""
    from ..analytics.match_stats import LENGTHS

    style.use()
    import matplotlib.pyplot as plt

    buckets = [name for _, _, name in LENGTHS]
    figure, axes = plt.subplots(1, len(reports), figsize=(5.2 * len(reports), 3.8), sharey=True)
    for axis, report in zip(np.atleast_1d(axes), reports, strict=True):
        names = list(report["pairs"])
        x = np.arange(len(buckets))
        for k, name in enumerate(names):
            won = [report["pairs"][name]["won_by_length"].get(b, 0) for b in buckets]
            bars = axis.bar(x + (k - 0.5) * 0.36, won, width=0.34, color=PAIR_COLOURS[k],
                            edgecolor=style.PANEL, linewidth=2, label=name)
            for bar, value in zip(bars, won, strict=True):
                axis.text(bar.get_x() + bar.get_width() / 2, value + 0.4, str(value),
                          ha="center", va="bottom", color=style.TEXT, fontsize=9)
        axis.set_xticks(x)
        axis.set_xticklabels(buckets)
        axis.set_title(MATCH_NAMES.get(report["match"], report["match"]), loc="left", pad=28,
                       fontsize=11)
        axis.grid(axis="x", visible=False)
        _pair_legend(axis, names)
    np.atleast_1d(axes)[0].set_ylabel("points won")
    _save(figure, path)


def pair_occupancy_chart(reports: Sequence[dict], path: Path) -> None:
    """Draw where each pair stood, folded onto one half: the net on top, the wall below."""
    from matplotlib.colors import LinearSegmentedColormap

    style.use()
    import matplotlib.pyplot as plt

    ramp = LinearSegmentedColormap.from_list(
        "bleu", ["#184f95", "#3987e5", "#86b6ef", "#cde2fb"])
    figure, axes = plt.subplots(len(reports), 2, figsize=(7.2, 4.2 * len(reports)))
    for row_axes, report in zip(np.atleast_2d(axes), reports, strict=True):
        for axis, (name, occupancy) in zip(row_axes, report["occupancy"].items(), strict=True):
            grid = np.array(occupancy["grid"])
            style.draw_court(axis)
            axis.imshow(np.ma.masked_less(grid, grid.max() * 0.02), origin="lower",
                        extent=occupancy["extent"], cmap=ramp, alpha=0.9, zorder=2)
            axis.set_ylim(-10.6, 0.6)
            axis.set_title(f"{name}\n{MATCH_NAMES.get(report['match'], report['match'])}",
                           fontsize=10)
    _save(figure, path)
