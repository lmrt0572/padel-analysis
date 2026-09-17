"""Pointage a la main de TOUS les contacts d'une plage de match.

Juger les contacts qu'une chaine propose ne mesure que sa precision : un contact
qu'elle n'a jamais propose n'est jamais juge, et un mur rate ne coute rien. Ici on
regarde la plage entiere et on pointe chaque contact reel, sans rien voir de ce que le
systeme a detecte. Toute variante se note ensuite en precision ET en rappel.

    espace  lecture / pause
    j  l    reculer / avancer d'une image
    h  ;    reculer / avancer d'une seconde
    +  -    lecture plus rapide / plus lente

    s  le SOL      v  une VITRE      g  le GRILLAGE      t  le FILET      f  une FRAPPE
       marquent un contact sur l'image affichee : mettre en pause et se placer sur
       l'image du contact avant de taper

    r  annuler la derniere marque
    q  quitter (tout est deja sauvegarde)

Le fichier est reecrit de facon atomique a chaque marque, et la position de lecture
aussi : on reprend la ou l'on s'etait arrete.

Usage:
    python scripts/mark_contacts.py --video <video.mp4> --start 16000 --frames 1800 \
        --video-name FinalF --out ground_truth/contact_marks/FinalF_16000.json
"""

import argparse
from pathlib import Path

import cv2

from padel_analysis.eval.contact_marks import ContactMarks
from padel_analysis.io.video_source import VideoSource

WINDOW = "pointage des contacts"
KEYS = {"s": "sol", "v": "verre", "g": "grillage", "t": "filet", "f": "raquette"}
HELP = "espace lecture   j l image   h ; seconde   + - vitesse   s v g t f marquer   r annuler   q quitter"
DISPLAY = (1280, 720)
BUFFER = 150


class Frames:
    """Lecture sequentielle rapide, avec un tampon pour revenir en arriere sans rechercher."""

    def __init__(self, source: VideoSource) -> None:
        self.source = source
        self.buffer: dict[int, object] = {}
        self.stream = None
        self.next = None

    def get(self, index: int):
        if index in self.buffer:
            return self.buffer[index]
        if self.stream is None or index != self.next:
            self.stream = self.source.iter_frames(start=index)
            self.next = index
        while True:
            got, frame = next(self.stream)
            self.next = got + 1
            self.buffer[got] = cv2.resize(frame, DISPLAY)
            for old in [f for f in self.buffer if abs(f - got) > BUFFER]:
                del self.buffer[old]
            if got >= index:
                return self.buffer[index]


def draw(frame, marks: ContactMarks, index: int, playing: bool, delay: int, flash: str):
    canvas = frame.copy()
    start, stop = marks.frame_range
    seconds = (index - start) / 30.0
    status = "LECTURE" if playing else "PAUSE"
    header = (
        f"image {index}   {seconds:5.1f} s   {status}   {1000 // max(delay, 1)} img/s   "
        f"{len(marks.marks)} contacts marques"
    )
    lines = [
        (header, 36, 0.8),
        (HELP, 70, 0.55),
    ]
    if flash:
        lines.append((flash, 118, 1.1))
    for text, y, scale in lines:
        cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
        cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)

    # Frise de la plage : ses propres marques seulement, jamais les detections du systeme.
    width, height = DISPLAY
    top = height - 26
    cv2.rectangle(canvas, (16, top), (width - 16, top + 12), (40, 40, 40), -1)
    span = max(stop - start, 1)
    for frame_marked in marks.marks:
        x = 16 + round((frame_marked - start) / span * (width - 32))
        cv2.line(canvas, (x, top - 4), (x, top + 16), (0, 220, 255), 2)
    x = 16 + round((index - start) / span * (width - 32))
    cv2.line(canvas, (x, top - 8), (x, top + 20), (255, 255, 255), 3)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-name", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    start, stop = args.start, args.start + args.frames - 1
    if args.out.exists():
        marks = ContactMarks.load(args.out)
        if marks.frame_range != (start, stop):
            raise SystemExit(f"{args.out} porte sur la plage {marks.frame_range}, pas {(start, stop)}")
    else:
        marks = ContactMarks(video=args.video_name, frame_range=(start, stop), position=start)
        marks.save(args.out)

    index = min(max(marks.position, start), stop)
    playing, delay, flash, flash_left = False, 100, "", 0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    with VideoSource(args.video) as source:
        frames = Frames(source)
        while True:
            cv2.imshow(WINDOW, draw(frames.get(index), marks, index, playing, delay, flash))
            key = cv2.waitKey(delay if playing else 0) & 0xFF
            flash_left -= 1
            if flash_left <= 0:
                flash = ""

            if key == 255:
                if playing:
                    index = min(index + 1, stop)
                    playing = index < stop
                continue
            char = chr(key)
            if char == "q":
                break
            if char == " ":
                playing = not playing
            elif char in "jlh;":
                playing = False
                step = {"j": -1, "l": 1, "h": -30, ";": 30}[char]
                index = min(max(index + step, start), stop)
            elif char == "+":
                delay = max(15, delay // 2)
            elif char == "-":
                delay = min(400, delay * 2)
            elif char in KEYS:
                marks.mark(index, KEYS[char], merge=2)
                flash, flash_left = f"MARQUE : {KEYS[char]} a l'image {index}", 25
            elif char == "r":
                undone = marks.undo()
                if undone:
                    flash, flash_left = f"ANNULE : {undone[1]} a l'image {undone[0]}", 25
            marks.position = index
            marks.save(args.out)
    marks.position = index
    marks.save(args.out)
    cv2.destroyAllWindows()
    print(f"{len(marks.marks)} contacts marques, position sauvegardee a l'image {index}")


if __name__ == "__main__":
    main()
