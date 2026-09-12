"""Ajoute au fichier de calibration les reperes en hauteur releves sur les murs.

Une homographie se contente de points au sol ; une pose de camera ne le peut pas, un
ensemble coplanaire laissant la direction verticale libre. Ces dix points viennent du
zonage manuel des murs, frame 200 de la finale feminine.

Les deux premiers servent a l'ajustement, les huit autres restent des controles et
doivent le rester : les verser dans l'ajustement ferait disparaitre la seule mesure
honnete de la pose.

Les trois calibrations partagent les memes pixels, la camera etant fixe et identique
d'une video a l'autre - verifie par recouvrement au sous-projet A.

Usage:
    python scripts/add_height_references.py ground_truth/calibrations/*.json
"""

import argparse
import json
from pathlib import Path

# nom, (x, y) en metres, hauteur en metres, pixel, est un point de controle
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
        print(f"{path.name} : {added} points ajoutes, {len(payload['points'])} au total")


if __name__ == "__main__":
    main()
