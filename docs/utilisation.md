# Utilisation

Toutes les commandes du projet, de la calibration d'un court aux mesures du rapport.
Elles se lancent depuis la racine du dépôt, dans l'environnement `padel`
(voir l'installation dans le [README](../README.md#démarrage-rapide)).

## Données

Télécharger le dataset PadelTracker100 (8,2 Go) :

```bash
python scripts/download_dataset.py
```

## Calibrer un court

Treize points au sol sont demandés ; un schéma du court et une loupe 5× s'affichent dans
la fenêtre pour guider chaque clic. Les quatre derniers servent de contrôle, et l'erreur
affichée à la fin est mesurée sur eux seuls :

```bash
python scripts/calibrate.py --video <video.mp4> --frame 200 --out ground_truth/calibrations/<nom>.json
```

Vérifier visuellement la calibration en superposant le modèle du court :

```bash
python scripts/overlay_court.py --video <video.mp4> --frame 200 \
    --calibration ground_truth/calibrations/<nom>.json --out outputs/overlay.png
```

La pose de la caméra, qui place la balle en trois dimensions sur les vitres, demande
aussi des repères en hauteur ; ceux de la caméra des deux finales s'ajoutent avec
`scripts/add_height_references.py`, et se vérifient avec :

```bash
python scripts/check_camera_pose.py --calibration ground_truth/calibrations/<nom>.json
```

## La vidéo de statistiques

La démonstration principale : l'échange, et à côté un panneau qui avance avec le jeu —
minicarte, échange en cours, points lus au tableau d'affichage, frappes, volées,
distance, vitesse maximale et temps au filet de chaque joueur. Elle demande l'analyse
sauvegardée de la minute (voir plus bas), le modèle de contacts et ffmpeg :

```bash
python scripts/stats_video.py --match FinalF --minute 8000 --start 9084 --stop 9799 \
    --contact-model weights/contact_net.pt --out outputs/stats.mp4
```

Avec `--replay`, la même vidéo est dessinée sur le court reconstruit à partir de la
calibration, sans aucune image de la retransmission : joueurs, balle, contacts et
panneau, c'est-à-dire tout ce que l'analyse a reconstruit, et seulement cela.

## Le bilan d'un match entier

Analyser les deux finales en entier, minute par minute (environ trois heures et demie
sur une GTX 1650 ; relancer la commande reprend à la minute suivante) :

```bash
python scripts/analyse_match.py --weights weights/ball_net.pt
```

Lire le tableau d'affichage, puis assembler le bilan par paire — suivi des joueurs
rejoué sur tout le match, contacts, échanges, changements de côté, points et
déplacements — dans `outputs/match_stats/<match>.json` :

```bash
python scripts/read_scores.py --match FinalF --out outputs/scores/FinalF.json
python scripts/match_stats.py --match FinalF --contact-model weights/contact_net.pt
```

Les figures du bilan se refont avec `scripts/make_figures.py`, qui lit
`outputs/match_stats/`.

## La chaîne complète et la vidéo annotée

Vidéo annotée avec minimap, et positions mises en cache :

```bash
python -m padel_analysis.cli --video <video.mp4> \
    --calibration ground_truth/calibrations/<nom>.json \
    --out outputs/annotated.mp4 --cache cache/<nom>.json --start 5000 --frames 1800
```

Pour un passage destiné aux statistiques, `--no-video` saute le rendu et ne produit
que le cache. Un match entier prend alors environ une heure sur une GTX 1650 :

```bash
python -m padel_analysis.cli --video <video.mp4> \
    --calibration ground_truth/calibrations/<nom>.json --cache cache/<nom>.json --no-video
```

Puis calculer les statistiques et les figures depuis ce cache, sans réinférence :

```bash
python -m padel_analysis.analyse --cache cache/<nom>.json \
    --out outputs/<nom>_report.json --figures outputs/
```

La démonstration des contacts demande les poids du réseau de détection de balle. Elle
analyse toute la plage avant de dessiner, puisque la balle est choisie sur la séquence
entière. Pour l'affichage seulement, la balle est masquée là où le réseau n'est pas sûr
de lui : mesuré sur une minute annotée, les trajectoires fantômes — balle hors champ,
balle en main avant le service — passent de 222 images à 36, pour 97,5 % des positions
justes conservées. Les contacts de mur dont le rayon ne rencontre qu'une seule surface
sont aussi masqués : sur les deux matchs annotés, ils portent 38 des 42 faux murs, pour
2 vrais murs sur 33. Les chiffres mesurés ne sont pas filtrés.

```bash
python -m padel_analysis.demo --video <video.mp4> \
    --calibration ground_truth/calibrations/<nom>.json \
    --weights weights/ball_net.pt --start 16000 --frames 1800 --out outputs/demo.mp4 \
    --contact-model weights/contact_net.pt
```

## Pointer les contacts à la main

L'outil montre la vidéo et rien de ce que le système détecte. `s v g t f` marquent un
rebond au sol, une vitre, le grillage, le filet ou une frappe sur l'image affichée ;
deux contacts peuvent se suivre d'une image à l'autre, et l'écran le signale :

```bash
python scripts/mark_contacts.py --video <video.mp4> --video-name FinalF --start 16000 \
    --frames 1800 --out ground_truth/contact_marks/FinalF_16000.json
```

## Le modèle de contacts et ses juges

Analyser les minutes pointées (la passe coûteuse, sur carte graphique), entraîner le
modèle en validation croisée, puis le noter sur un juge — des minutes pointées et jamais
regardées, qui ne servent qu'une fois :

```bash
python scripts/analyse_minutes.py --weights weights/ball_net.pt --tag 360
python scripts/train_contact_model.py --cv --out weights/contact_net.pt
python scripts/score_minutes.py --tag 360 --contact-model weights/contact_net.pt --juge-5
```

Les figures du rapport se refont depuis les chiffres écrits par les scripts de mesure :

```bash
python scripts/train_contact_model.py --cv --curve --results outputs/measures/cv.json \
    --out weights/contact_net_rerun.pt
python scripts/make_figures.py --cache cache/<match entier>.json
```

Les commandes qui reproduisent chaque mesure sont dans le
[rapport d'évaluation](evaluation.md#reproduire-lévaluation).
