"""One look for every figure: a dark palette, and a court to draw on.

Green is the floor, cyan the glass, and each player keeps a colour.
"""

BACKGROUND = "#0e1117"
PANEL = "#161b22"
LINE = "#2a313c"
TEXT = "#e8eaed"
MUTED = "#8b95a5"
COURT = "#123a2a"
COURT_BLUE = "#1d4f8f"  # the blue of a padel court, for the minimap

KIND = {
    "sol": "#3ddc97",
    "verre": "#4cc9f0",
    "grillage": "#adb5bd",
    "filet": "#ff6b6b",
    "raquette": "#e8eaed",
    "aucun": "#2a313c",
    "fin": "#2a313c",
}
PLAYER = {"near_1": "#ffd166", "near_2": "#f4845f", "far_1": "#9d7bea", "far_2": "#f15bb5"}
RULES = "#8b95a5"
MODEL = "#3ddc97"
KIND_NAMES = {"raquette": "stroke", "sol": "floor", "verre": "glass", "grillage": "mesh",
              "filet": "net", "aucun": "nothing", "fin": "nothing after"}


def use() -> None:
    """Switch matplotlib to the dark style, without a display."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BACKGROUND,
        "axes.edgecolor": LINE,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.grid": True,
        "grid.color": LINE,
        "grid.linewidth": 0.6,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 11,
    })


def draw_court(axis, half_width: float = 5.0, half_length: float = 10.0,
               service: float = 6.95) -> None:
    """Draw a court seen from above, camera at the bottom, in metres."""
    from matplotlib.patches import Rectangle

    axis.add_patch(Rectangle((-half_width, -half_length), 2 * half_width, 2 * half_length,
                             facecolor=COURT, edgecolor=KIND["sol"], linewidth=1.5, zorder=0))
    lines = {"color": KIND["sol"], "alpha": 0.45, "linewidth": 0.9, "zorder": 1}
    for depth in (-service, service):
        axis.plot([-half_width, half_width], [depth, depth], **lines)
    axis.plot([0, 0], [-service, service], **lines)
    axis.plot([-half_width - 0.3, half_width + 0.3], [0, 0], color=TEXT, linewidth=2, zorder=1)
    axis.set_xlim(-half_width - 1, half_width + 1)
    axis.set_ylim(-half_length - 1, half_length + 1)
    axis.set_aspect("equal")
    axis.grid(False)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
