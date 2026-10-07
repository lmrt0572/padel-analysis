"""Adds to the calibration file the references above the ground taken on the walls.

A homography makes do with points on the ground; a camera pose cannot, a coplanar set
leaving the vertical direction free. These ten points come from the manual zoning of
the walls, frame 200 of the women's final.

The first two are used for the fit, the other eight remain controls and must stay so:
pouring them into the fit would make the only honest measure of the pose disappear.

The three calibrations share the same pixels, the camera being fixed and identical from
one video to the next (checked by overlay in sub-project A).

Usage:
    python scripts/add_height_references.py ground_truth/calibrations/*.json
"""

import argparse
import json
from pathlib import Path

# name, (x, y) in metres, height in metres, pixel, is a control point
REFERENCES = [
    ("glass_top_near_left", [-5.0, -10.0], 3.00, [168.0, 531.0], False),
    ("glass_top_near_right", [5.0, -10.0], 3.00, [1764.0, 528.0], False),
    ("mesh_top_near_left", [-5.0, -10.0], 4.00, [149.0, 369.0], True),
    ("mesh_top_near_right", [5.0, -10.0], 4.00, [1780.0, 362.0], True),
    ("glass_top_far_left", [-5.0, 10.0], 3.00, [576.0, 95.0], True),
    ("glass_top_far_right", [5.0, 10.0], 3.00, [1345.0, 89.0], True),
    ("mesh_top_far_left", [-5.0, 10.0], 4.00, [578.0, 19.0], True),
    ("mesh_top_far_right", [5.0, 10.0], 4.00, [1346.0, 10.0], True),
    ("net_top_left", [-5.0, 0.0], 0.92, [469.0, 443.0], True),
    ("net_top_right", [5.0, 0.0], 0.92, [1455.0, 437.0], True),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()

    for path in args.paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        existing = {p["name"] for p in payload["points"]}
        added = 0
        for name, court_xy, height, pixel, is_control in REFERENCES:
            if name in existing:
                continue
            payload["points"].append(
                {
                    "name": name,
                    "court_xy": court_xy,
                    "image_xy": pixel,
                    "is_control": is_control,
                    "height": height,
                }
            )
            added += 1
        for point in payload["points"]:
            point.setdefault("height", 0.0)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{path.name}: {added} points added, {len(payload['points'])} in total")


if __name__ == "__main__":
    main()
