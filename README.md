# Padel Analysis

Analyse automatique de matchs de padel à partir de diffusions vidéo : détection et
suivi des quatre joueurs, projection de leurs positions sur une représentation du
court en mètres, et statistiques tactiques.

## État

Jalon 3A terminé : statistiques tactiques sur un match complet de vingt-cinq minutes.
Évaluation chiffrée et ablations à venir.

## Installation

```bash
conda create -n padel python=3.11 -y
conda run -n padel pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
conda run -n padel pip install -e ".[dev]"
```

L'installation explicite de torch CUDA n'est pas optionnelle sous Windows : le torch
tiré par défaut est une version CPU, et l'inférence passerait de minutes à heures.

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

Calibrer le court sur une frame. Treize points sont demandés ; un schéma du court et
une loupe 5× s'affichent dans la fenêtre pour guider chaque clic :

```bash
python scripts/calibrate.py --video <video.mp4> --frame 200 --out data/calibrations/<nom>.json
```

Vérifier visuellement la calibration en superposant le modèle du court :

```bash
python scripts/overlay_court.py --video <video.mp4> --frame 200 \
    --calibration data/calibrations/<nom>.json --out outputs/overlay.png
```

Lancer la chaîne complète — vidéo annotée avec minimap, et positions mises en cache :

```bash
python -m padel_analysis.cli --video <video.mp4> \
    --calibration data/calibrations/<nom>.json \
    --out outputs/annotated.mp4 --cache cache/<nom>.json --start 5000 --frames 1800
```

Pour un passage destiné aux statistiques, `--no-video` saute le rendu et ne produit
que le cache. Un match entier prend alors environ une heure sur une GTX 1650 :

```bash
python -m padel_analysis.cli --video <video.mp4> \
    --calibration data/calibrations/<nom>.json --cache cache/<nom>.json --no-video
```

Puis calculer les statistiques et les figures depuis ce cache, sans réinférence :

```bash
python -m padel_analysis.analyse --cache cache/<nom>.json \
    --out outputs/<nom>_report.json --figures outputs/
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

- Les positions des joueurs de la moitié éloignée sont intrinsèquement quatre fois
  plus bruitées que celles de la moitié proche.
- Le bruit gonfle les distances parcourues mesurées. La comparaison entre les deux
  équipes n'est donc pas symétrique à l'intérieur d'un même jeu.

### Résolution d'inférence

L'entrée du modèle est redimensionnée avant inférence. Mesuré sur 300 frames
annotées, soit 1200 joueurs à retrouver, en comparant le milieu des chevilles
prédit aux chevilles annotées :

| imgsz | ms/frame | Erreur (px) | Moitié proche (cm) | Moitié éloignée (cm) | Joueurs retrouvés |
|---|---|---|---|---|---|
| 640 | 41 | 4,03 | 6,4 | 34,3 | 572 / 1200 |
| 960 | 38 | 2,70 | 4,7 | 7,9 | 745 / 1200 |
| 1280 | 59 | 2,46 | 4,4 | 7,1 | 1170 / 1200 |
| **1600** | **90** | **2,06** | **3,8** | **6,2** | **1200 / 1200** |

**La colonne décisive est la dernière.** La résolution ne gouverne pas seulement la
précision, elle gouverne le **rappel** : à 640 pixels le modèle ne retrouve que 48 %
des joueurs. Ceux du fond du court, hauts d'une centaine de pixels et fréquemment
occultés par leur partenaire, disparaissent purement et simplement. L'erreur médiane
de 4 px affichée à 640 ne le révèle pas, puisqu'elle ne porte que sur les joueurs
effectivement trouvés.

**Résolution retenue : 1600.** Elle retrouve tous les joueurs, et son erreur au fond
du court (6,2 cm) passe sous l'erreur de calibration (9,9 cm) : affiner davantage la
perception n'améliorerait plus rien, le plancher étant géométrique.

### Perception et suivi

Mesuré sur 1800 frames consécutives, soit une minute de jeu :

| Mesure | Valeur |
|---|---|
| Cadence de la chaîne complète (GTX 1650, 4 Go) | 9,7 frames/s |
| Frames à quatre joueurs identifiés | **96,50 %** |
| Référence : frames à quatre personnes annotées | 99,94 % |
| Positions hors du court en `x` | 0,00 % |

Le même pipeline à `imgsz` 1280 n'atteignait que 92,33 % : la résolution explique
l'essentiel de l'écart.

Une observation dont la position projetée sort largement de l'enceinte est refusée.
Sans cette borne le taux affiché montait à 97,94 %, mais la différence tenait à des
frames où un spectateur du bon côté du filet occupait un emplacement libre : le
chiffre était flatteur et faux. Le refus a aussi séparé les deux emplacements du
fond, dont les positions moyennes étaient confondues parce que ces intrus les
occupaient par intermittence.

L'identité est maintenue par appariement hongrois sur quatre emplacements fixes, deux
de chaque côté du filet. Le côté vient du signe de la coordonnée `y` après projection :
la calibration alimente donc directement le suivi. Le coût d'assignation combine la
distance à la position prédite par un modèle à vitesse constante, la signature de
couleur de la tenue et la confiance des chevilles.

Cette contrainte structurelle n'est pas décorative : **le détecteur trouve plus de
quatre personnes dans 63 % des frames** — spectateurs, ramasseurs de balle, arbitre —
et le suivi contraint retient systématiquement les quatre bonnes.

La signature de couleur a été validée par la mesure avant d'être conservée : la
dérive d'un même joueur d'une frame à l'autre vaut 0,031, contre 0,199 entre deux
partenaires. Le rapport de 6,3 confirme qu'elle distingue bien des coéquipiers
portant la même tenue, et pas seulement les deux équipes.

### Analyse tactique — contrôle du filet

Mesuré sur le match complet, 45 934 frames, dont 44 911 portent les quatre joueurs.

Les joueurs occupent deux profondeurs distinctes. Sur 182 713 positions, le mode
offensif culmine à **3,95 m** du filet et le mode défensif à **7,85 m**, contre la
vitre de fond. Le creux qui les sépare tombe à **5,85 m**, et c'est là qu'est placé le
seuil.

La ligne de service, à 6,95 m, n'est délibérément pas utilisée : c'est une règle de
service et non un marqueur de position tactique, et elle tombe du mauvais côté du
creux — elle classerait toute la bande défensive comme offensive.

| Mesure | Valeur |
|---|---|
| Contrôle côté proche | **38,0 %** |
| Contrôle côté éloigné | **20,5 %** |
| Disputé | 41,5 % |
| Frames évaluées | 44 911 |

**Le creux est large et peu profond**, donc la valeur exacte du seuil relève en partie
de la convention, et les pourcentages la suivent :

| Seuil | Proche | Éloigné | Disputé | Rapport proche/éloigné |
|---|---|---|---|---|
| 5,00 m | 28,0 % | 12,3 % | 59,8 % | 2,28 |
| 5,50 m | 33,9 % | 18,0 % | 48,2 % | 1,88 |
| **5,85 m** | **38,0 %** | **20,5 %** | **41,5 %** | **1,85** |
| 6,00 m | 39,7 % | 21,4 % | 38,9 % | 1,86 |
| 6,50 m | 43,5 % | 23,5 % | 33,0 % | 1,85 |

Les valeurs absolues dépendent donc du seuil, mais **le rapport entre les deux paires
ne bouge pratiquement pas** au-delà de 5,5 m. La conclusion robuste de ce match est
que la paire du côté proche a tenu le filet environ **1,85 fois plus souvent** que
l'autre — indépendamment de la convention retenue.

### Analyse tactique — distance et vitesse

La distance est donnée brute et lissée. L'écart entre les deux chiffre la part qu'y a
prise le bruit de position, au lieu de la masquer.

| Joueur | Moitié | Distance brute | Distance lissée | Part de bruit | Vitesse p95 | Profondeur moyenne |
|---|---|---|---|---|---|---|
| near_1 | proche | 2890 m | 2261 m | **21,8 %** | 3,50 m/s | 5,41 m |
| near_2 | proche | 2807 m | 2210 m | **21,3 %** | 3,51 m/s | 5,44 m |
| far_1 | éloignée | 3282 m | 2233 m | **32,0 %** | 3,77 m/s | 6,75 m |
| far_2 | éloignée | 3300 m | 2285 m | **30,8 %** | 3,92 m/s | 6,77 m |

**La part de bruit est une demi-fois plus élevée pour la moitié éloignée**, ce que
prédit l'asymétrie de 4,3× documentée plus haut. Les distances lissées, elles, sont
comparables entre les quatre joueurs alors que les distances brutes ne l'étaient pas :
l'écart apparent de 400 mètres entre les deux paires était du bruit, pas du jeu.

## Limites connues

**L'identité entre partenaires n'est pas encore vérifiée.** La contrainte de côté,
elle, l'est : sur 7 137 positions enregistrées, aucune n'attribue un emplacement du
côté proche à une observation du côté éloigné. Mais un échange d'identité *entre les
deux partenaires d'une même paire* ne violerait aucune contrainte, et rien ici ne le
détecterait. Les positions moyennes des quatre emplacements sont désormais bien
séparées, ce qui est encourageant sans rien prouver. Trancher demande une vérité
terrain d'identité, que le dataset ne fournit pas et qu'il faudra annoter à la main.

**Les frames incomplètes sont des échecs de détection**, pas de suivi : ce sont
exactement celles où le modèle ne trouve que trois personnes, toujours au fond du
court, lorsque deux joueuses adjacentes s'occultent mutuellement.

**Les parois latérales ne sont pas modélisées.** Seules les parois de fond le sont ;
la géométrie en paliers des côtés demande une vérification dans le règlement FIP.

**Le lissage ne retire pas tout le bruit.** L'écart entre distance brute et distance
lissée dit ce que le lissage a enlevé, pas ce qu'il reste. Les distances lissées
correspondent à environ 88 mètres par minute de **jeu effectif** — la vidéo étant
montée sur les échanges, elle ne contient aucun temps mort. Ce chiffre n'est donc pas
directement comparable aux distances par match que rapporte la littérature, qui
incluent les interruptions.

**Le seuil du filet est une convention, quoique mesurée.** Le creux entre les deux
modes est réel mais large : les pourcentages absolus de contrôle bougent de quinze
points selon l'endroit où on le place dans ce creux. Le rapport entre les deux paires,
lui, est stable — c'est cette forme-là qu'il faut citer.

## Données

Ce projet utilise le dataset PadelTracker100 (Zenodo, DOI 10.5281/zenodo.14653706),
distribué sous licence CC-BY-4.0. Il contient deux matchs des World Padel Tour
Finals 2022 en 1920×1080@30, avec annotations COCO de pose (17 keypoints),
de balle et d'événements de frappe.

Les annotations de pose n'utilisent pas l'ordre COCO standard : gauche et droite y
sont inversés pour toutes les articulations appariées sauf les oreilles. Le module
`perception/keypoints.py` effectue la conversion, et la teste.

## Licence

AGPL-3.0. Voir `LICENSE`.

Ce projet dépend d'Ultralytics, distribué sous AGPL-3.0, ce qui impose cette licence
à l'ensemble. Le code n'est donc pas réutilisable dans un produit propriétaire.
