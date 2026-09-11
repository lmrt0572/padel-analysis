# Padel Analysis

Analyse automatique de matchs de padel à partir de diffusions vidéo : détection et
suivi des quatre joueurs, projection de leurs positions sur une représentation du
court en mètres, et statistiques tactiques.

## État

Jalon 1 terminé : chaîne géométrique complète et validée. Détection, suivi et
statistiques à venir.

## Installation

```bash
conda create -n padel python=3.11 -y
conda run -n padel pip install -e ".[dev]"
```

Les tests et le linter :

```bash
conda run -n padel python -m pytest -q
conda run -n padel ruff check src tests scripts
```

## Utilisation

Télécharger le dataset PadelTracker100 (8,2 Go) :

```bash
python scripts/download_dataset.py
```

Calibrer le court sur une frame. Treize points sont demandés ; un schéma et une
loupe 5× s'affichent dans la fenêtre pour guider chaque clic :

```bash
python scripts/calibrate.py --video <video.mp4> --frame 200 --out data/calibrations/<nom>.json
```

Vérifier visuellement la calibration en superposant le modèle du court :

```bash
python scripts/overlay_court.py --video <video.mp4> --frame 200 \
    --calibration data/calibrations/<nom>.json --out outputs/overlay.png
```

## Résultats

### Stabilité de la caméra

Homographie ORB + RANSAC entre la première frame et les suivantes, sur 407 frames :

| Mesure | Valeur |
|---|---|
| Déplacement maximal des coins de l'image | **0,02 px** |

La caméra de diffusion est fixe. Une seule calibration suffit par vidéo.

### Précision de la calibration

Treize points cliqués, dont **quatre points de contrôle exclus de l'ajustement** et
répartis sur l'ensemble du court, pour que l'erreur rapportée ne soit pas optimiste.

| Mesure | Pixels | Centimètres |
|---|---|---|
| RMSE | 3,66 | 11,6 |
| Médiane | 3,66 | 9,9 |

Détail par point de contrôle :

| Point | Erreur (px) | Erreur (cm) |
|---|---|---|
| Ligne de service proche, centre | 2,78 | 3,1 |
| Filet, paroi droite | 4,34 | 7,4 |
| Filet, centre | 3,95 | 12,4 |
| Ligne de service éloignée, centre | 3,37 | 17,8 |

### Asymétrie de précision entre les deux moitiés du court

L'erreur en pixels est uniforme (2,8 à 4,3 px). L'erreur en mètres ne l'est pas,
parce que l'échelle varie fortement avec la profondeur :

| Position sur le court | Centimètres par pixel |
|---|---|
| Fond proche | 1,51 |
| Ligne de service proche | 2,05 |
| Filet | 3,56 |
| Ligne de service éloignée | 5,49 |
| Fond éloigné | 6,47 |

**Un pixel vaut 4,3 fois plus au fond éloigné qu'au fond proche.** Cette asymétrie
est une propriété de la prise de vue, pas un défaut de la méthode : elle affecte
toute estimation de position, quelle que soit la façon dont elle est obtenue.

Deux conséquences pour la suite :

- Les positions des joueurs de la moitié éloignée seront intrinsèquement quatre
  fois plus bruitées que celles de la moitié proche.
- Le bruit gonfle les distances parcourues mesurées. La comparaison entre les deux
  équipes n'est donc pas symétrique à l'intérieur d'un même jeu.

## Données

Ce projet utilise le dataset PadelTracker100 (Zenodo, DOI 10.5281/zenodo.14653706),
distribué sous licence CC-BY-4.0. Il contient deux matchs des World Padel Tour
Finals 2022 en 1920×1080@30, avec annotations COCO de pose (17 keypoints),
de balle et d'événements de frappe.

## Licence

AGPL-3.0. Voir `LICENSE`.

Ce projet dépend d'Ultralytics, distribué sous AGPL-3.0, ce qui impose cette licence
à l'ensemble. Le code n'est donc pas réutilisable dans un produit propriétaire.
