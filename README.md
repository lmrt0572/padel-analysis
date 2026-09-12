# Padel Analysis

Analyse automatique de matchs de padel à partir de diffusions vidéo : détection et
suivi des quatre joueurs, projection de leurs positions sur une représentation du
court en mètres, et statistiques tactiques.

## État

Sous-projet A terminé : géométrie, suivi des quatre joueurs, statistiques tactiques,
évaluation chiffrée et deux ablations, validées sur un second match jamais utilisé
pour régler quoi que ce soit.

Sous-projet B en cours : détection de la balle et de ses contacts. Premier étage
livré — les candidats par mouvement et leur plafond de rappel.

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
python scripts/calibrate.py --video <video.mp4> --frame 200 --out ground_truth/calibrations/<nom>.json
```

Vérifier visuellement la calibration en superposant le modèle du court :

```bash
python scripts/overlay_court.py --video <video.mp4> --frame 200 \
    --calibration ground_truth/calibrations/<nom>.json --out outputs/overlay.png
```

Lancer la chaîne complète — vidéo annotée avec minimap, et positions mises en cache :

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

### Reproduire l'évaluation

Les métriques de suivi ajoutent deux dépendances, séparées parce qu'elles ne servent
qu'à mesurer :

```bash
pip install -e ".[eval]"
```

Le dataset ne fournit pas d'identité. Il faut la reconstruire, puis arbitrer à la main
les moments où la reconstruction est douteuse.

```bash
python scripts/build_identity_truth.py --annotations <pose.json> \
    --calibration ground_truth/calibrations/<nom>.json --out ground_truth/identity/<nom>.json

python scripts/detect_cuts.py --annotations <pose.json> \
    --calibration ground_truth/calibrations/<nom>.json --identity ground_truth/identity/<nom>.json

python scripts/review_identity.py --video <video.mp4> \
    --annotations <pose.json> --identity ground_truth/identity/<nom>.json
```

Le premier associe au plus proche voisin sur tout le match et liste les rapprochements
douteux ; le deuxième ajoute les raccords de plan, que la proximité ne voit pas ; le
troisième rejoue chaque moment douteux en boucle, les joueurs encadrés de leur couleur
d'emplacement.

La question posée n'est pas « se sont-ils croisés » mais **« le même joueur porte-t-il
la même couleur avant et après »**.

| Touche | Sur un rapprochement | Sur un raccord |
|---|---|---|
| `n` | pas de permutation | aucune permutation |
| `s` | permutation | — |
| `p` / `e` / `b` | — | la paire proche, éloignée, ou les deux ont permuté |
| `c` | — | les équipes ont changé de côté |
| `r` | revenir au clip précédent et annuler sa réponse | idem |
| `q` | quitter en conservant les réponses rendues | idem |

Le fichier est réécrit après chaque réponse, de façon atomique : une coupure de
courant coûte le clip en cours, pas l'arbitrage entier.

La campagne calcule ensuite la détection, la localisation et les deux ablations en un
seul passage sur la vidéo :

```bash
python scripts/run_evaluation.py --video <video.mp4> --annotations <pose.json> \
    --calibration ground_truth/calibrations/<nom>.json --identity ground_truth/identity/<nom>.json \
    --out outputs/<nom>_eval.json --frames 9000
```

La balle est mesurée à part, les deux méthodes de trajectoire étant calculées en un
seul passage sur la plage demandée :

```bash
python scripts/measure_trajectory.py --video <video.mp4> --annotations <ball.json>     --start 0 --stop 21472 --out outputs/<nom>_trajectory.json
```

Les contacts se mesurent de la même façon, sur le chemin reconstruit et sur la balle
annotée en un seul passage :

```bash
python scripts/measure_contacts.py --video <video.mp4> --annotations <ball.json>     --shots <shots.csv> --identity ground_truth/identity/<nom>.json     --start 0 --stop 20099 --out outputs/<nom>_contacts.json
```

Les surfaces demandent une vérité terrain qui n'existe pas : elle se produit à la main.
Le premier script dresse la liste des contacts à juger, le second les rejoue un par un.

```bash
python scripts/build_surface_tasks.py --annotations <ball.json> --poses <pose.json>     --calibration ground_truth/calibrations/<nom>.json --start 16000 --stop 20099     --video <nom> --out ground_truth/surfaces/<nom>.json

python scripts/review_surfaces.py --video <video.mp4>     --annotations <ball.json> --truth ground_truth/surfaces/<nom>.json
```

| Touche | Réponse |
|---|---|
| `s` `v` `g` `t` | le sol, une vitre, le grillage, le filet |
| `f` | une frappe, donc une raquette |
| `n` | aucun contact : la trajectoire passe tout droit |
| `x` | illisible, je ne peux pas trancher |
| `r` / `q` | revenir au clip précédent / quitter en conservant |

`n` et `x` ne disent pas la même chose et ne sont jamais additionnés. `x` est une
non-mesure ; `n` est un faux positif constaté de l'étage des contacts.

La pose de caméra se contrôle sur les repères qu'elle n'a jamais ajustés :

```bash
python scripts/check_camera_pose.py --calibration ground_truth/calibrations/<nom>.json
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

| Emplacement | Moitié | Distance brute | Distance lissée | Part de bruit | Vitesse p95 | Profondeur moyenne |
|---|---|---|---|---|---|---|
| near_1 | proche | 2890 m | 2261 m | **21,8 %** | 3,50 m/s | 5,41 m |
| near_2 | proche | 2807 m | 2210 m | **21,3 %** | 3,51 m/s | 5,44 m |
| far_1 | éloignée | 3282 m | 2233 m | **32,0 %** | 3,77 m/s | 6,75 m |
| far_2 | éloignée | 3300 m | 2285 m | **30,8 %** | 3,92 m/s | 6,77 m |

**Ces lignes sont des emplacements, pas des joueurs.** Les quatre emplacements
désignent deux positions de chaque côté du filet, et les équipes changent de côté
sept fois au cours de ce match : `near_1` est donc successivement plusieurs personnes.
Ces distances agrègent le trajet parcouru *à cet endroit du court*, ce qui reste une
mesure valide et interprétable, mais ce n'est pas une distance par joueur. La section
[Évaluation](#évaluation) explique pourquoi suivre un joueur à travers un changement
de côté est hors de portée de l'image seule.

**La part de bruit est une demi-fois plus élevée pour la moitié éloignée**, ce que
prédit l'asymétrie de 4,3× documentée plus haut. Les distances lissées, elles, sont
comparables entre les quatre emplacements alors que les distances brutes ne l'étaient
pas : l'écart apparent de 400 mètres entre les deux paires était du bruit, pas du jeu.

## Évaluation

Les mesures qui suivent comparent la sortie du pipeline aux annotations du dataset
sur 9 000 frames, soit cinq minutes de jeu. **Les deux ablations sortent du même
passage sur la vidéo** : aucune comparaison ne peut être faussée par un échantillon
différent.

### Détection

| Mesure | Valeur |
|---|---|
| Précision | 0,778 |
| **Rappel** | **0,954** |
| F1 | 0,857 |
| Personnes prédites | 44 134 |
| Personnes annotées | 36 006 |
| Appariées (IoU ≥ 0,5) | 34 338 |

La précision de 0,778 ne dit pas que le détecteur se trompe. Il trouve **8 000
personnes de plus qu'il n'y a de joueurs annotés** : arbitre, ramasseurs de balle,
premiers rangs du public. Ces détections sont correctes, elles ne sont simplement pas
des joueurs. C'est le rôle du suivi contraint de les écarter, et c'est ce que mesure
l'ablation 2.

Le rappel est donc la mesure qui compte ici : **95,4 % des joueurs annotés sont
retrouvés**.

### Ablation 1 — d'où vient le point au sol

Deux façons de décider où un joueur touche le sol : le milieu de ses chevilles, ou le
centre du bord inférieur de sa boîte englobante. Les deux implémentations coexistent
dans le code pour que le choix soit tranché par la mesure.

| Stratégie | Global | Moitié proche | Moitié éloignée |
|---|---|---|---|
| **Milieu des chevilles** | **1,89 px** | 2,01 px — 3,0 cm | 1,77 px — **11,5 cm** |
| Bas de la boîte | 19,61 px | 24,19 px — 36,5 cm | 16,64 px — **107,7 cm** |

Sur 34 338 échantillons appariés, **les chevilles font dix fois mieux**. Le bas de la
boîte englobante n'est pas l'endroit où le joueur touche le sol : c'est le point le
plus bas de l'englobant, qui inclut la raquette baissée et un pied levé.

L'écart est plus grand en pixels près de la caméra, et plus grand en mètres au fond du
court — les deux lectures sont vraies, et c'est l'asymétrie de 4,3× qui les sépare. Un
mètre d'erreur au fond avec le bas de la boîte, c'est la moitié d'une zone de service.

### Vérité terrain d'identité

Le dataset donne quatre personnes par frame mais ne dit jamais laquelle est laquelle.
Reconstruire cette information est presque gratuit : sur un match entier, deux
partenaires ne s'approchent **jamais** à moins de cinquante centimètres. La machine
associe donc au plus proche voisin sur tout le match, et un humain n'arbitre que les
moments où ça devient douteux.

Mais cette association suppose une image continue, et elle ne l'est pas. **La vidéo du
dataset est une concaténation des séquences de jeu**, temps morts retirés. Le tableau
d'affichage le prouve : entre deux frames consécutives, le score passe de 30 à 40.

À chaque raccord, les joueurs réapparaissent ailleurs. Ce n'est pas un rapprochement —
ils ne se frôlent pas, ils se téléportent — donc rien ne paraît ambigu, et l'identité
peut changer en silence.

| | rapprochements | raccords | changements de côté |
|---|---|---|---|
| Final féminine | 14 | 83 | 7 |
| Final masculine | 54 | 115 | 8 |

**266 clips arbitrés à la main**, chacun rejoué en boucle avec les quatre joueurs
encadrés de leur couleur d'emplacement.

Détecter ces raccords demande un critère contre-intuitif. Exiger que **les deux**
joueurs d'une paire bougent semble plus sûr, et c'est exactement l'inverse :
l'association au plus proche voisin minimise le déplacement apparent, donc une paire
qui échange ses places à un raccord produit le signal « ils n'ont pas bougé ». Le
critère strict est aveugle aux cas qu'il devrait attraper. Un seul joueur au-dessus du
seuil suffit donc, et la tolérance croît avec le temps écoulé : huit mètres en trois
frames manquantes est un raccord, dix mètres en soixante-dix-sept est un joueur qui
court.

Un changement de côté, lui, n'est pas une erreur à corriger. Les emplacements
désignent une moitié de court : quand les équipes changent de côté, `near_1` est
quelqu'un d'autre, et aucun échange d'étiquettes ne peut l'exprimer. **L'identité
s'arrête là et repart** — les métriques d'identité coupent des deux côtés de la
comparaison à cet endroit, et ne créditent ni ne pénalisent personne pour une
frontière qu'aucune information de l'image ne permet de franchir.

### Ablation 2 — la contrainte de court

Le suivi contraint tient exactement quatre emplacements, deux de chaque côté du filet,
et refuse toute position hors de l'enceinte. La ligne de base est ByteTrack, sans
aucune de ces contraintes.

| | Pistes moy. | Frames > 4 pistes | MOTA | IDF1 | Permutations |
|---|---|---|---|---|---|
| **Suivi contraint** | **3,98** | **0** | **0,912** | **0,819** | **4** |
| ByteTrack seul | 4,78 | 5 152 | 0,704 | 0,281 | 67 |

**ByteTrack dépasse quatre pistes sur 5 152 frames des 8 990 évaluées** — plus d'une
sur deux. Rien ne le borne, et le détecteur lui fournit huit mille personnes de trop.
L'IDF1 de 0,281 signifie que la plupart des identités de référence ne sont couvertes
par aucune piste stable : des statistiques par joueur calculées là-dessus seraient du
bruit.

### Ce que l'arbitrage change à la mesure

La vérité terrain d'identité peut être construite automatiquement, sans arbitrage
humain. Elle donne alors ceci, sur exactement les mêmes frames et le même code :

| | Sans arbitrage | Après arbitrage |
|---|---|---|
| MOTA | 0,913 | 0,912 |
| **IDF1** | **0,956** | **0,819** |
| **Permutations d'identité** | **0** | **4** |

**La version automatique annonce zéro permutation. Il y en a quatre.**

L'explication tient en une phrase : l'association au plus proche voisin qui construit
la référence est la même hypothèse que celle du tracker évalué. Aux raccords, les deux
se trompent ensemble, et la métrique compare une erreur à elle-même. L'IDF1 était
surestimé de 0,137.

MOTA ne bouge pas (0,913 → 0,912), ce qui est cohérent : il est dominé par les faux
positifs et les manques de détection, pas par l'identité. **Il fallait IDF1 pour voir
le problème, et une référence indépendante pour qu'IDF1 puisse le dire.**

Sur ces 9 000 frames il y a 18 raccords, 3 rapprochements et 1 changement de côté :
le suivi contraint décroche 4 fois sur 21 occasions, ByteTrack 67 fois.

### Phases aériennes

Un point à hauteur `h` projeté par une homographie de sol atterrit à `d × H / (H − h)`
de l'aplomb caméra au lieu de `d`. Le padel se joue en sautant — smash, bandeja,
vibora — donc la question n'est pas de savoir si le biais existe mais ce qu'il pèse.

| | Frames en phase aérienne |
|---|---|
| Final féminine | 1 640 / 183 456 — **0,89 %** |
| Final masculine | 2 293 / 211 252 — **1,09 %** |

| Hauteur du saut | Biais au filet | Biais au fond |
|---|---|---|
| 15 cm | 16 cm | 36 cm |
| 30 cm | 33 cm | **74 cm** |
| 50 cm | 56 cm | **127 cm** |

**Le biais est important quand il survient, et il survient rarement.** Un saut de
30 cm décale la position projetée de 74 cm au fond du court, soit six fois l'erreur
médiane des chevilles au même endroit. Mais sur 1 % des frames : sa contribution à une
heatmap ou à une distance cumulée est marginale, alors qu'elle domine toute position
instantanée mesurée pendant un smash.

### Match tenu à l'écart

Tout ce qui précède porte sur la finale féminine, qui a servi à régler le pipeline :
résolution d'inférence, seuil du filet, bornes du court, stratégie de point au sol.
La finale masculine n'a jamais servi à régler quoi que ce soit. Elle a été calibrée
par transfert, annotée en identité, puis évaluée une fois.

| | Finale féminine (réglage) | Finale masculine (tenue à l'écart) |
|---|---|---|
| Précision | 0,778 | **0,828** |
| Rappel | 0,954 | 0,943 |
| **F1 détection** | 0,857 | **0,882** |
| Chevilles, global | 1,89 px | 2,00 px |
| Bas de boîte, global | 19,61 px | 20,57 px |
| MOTA, suivi contraint | 0,912 | **0,856** |
| **IDF1, suivi contraint** | **0,819** | **0,764** |
| **Permutations d'identité** | **4** | **28** |
| IDF1, ByteTrack | 0,281 | 0,252 |
| Permutations, ByteTrack | 67 | 101 |

**La détection et la localisation transfèrent. Le suivi d'identité, non.**

La détection est même meilleure sur le match tenu à l'écart — F1 de 0,882 contre
0,857 — parce que sa précision monte de cinq points : le détecteur y trouve moins de
personnes qui ne jouent pas. La localisation est à onze centièmes de pixel près
identique, ce qui était attendu puisque la géométrie ne dépend pas des joueurs.

L'identité, elle, se dégrade nettement : **4 permutations deviennent 28**. Le chiffre
brut exagère l'écart, parce que le match masculin offre davantage d'occasions de
décrocher sur la même durée. Normalisé, l'écart reste :

| | Occasions | Permutations | Taux |
|---|---|---|---|
| Finale féminine | 21 — 18 raccords, 3 rapprochements | 4 | **19 %** |
| Finale masculine | 38 — 24 raccords, 14 rapprochements | 28 | **74 %** |

La cause tient dans la deuxième colonne : **14 rapprochements contre 3**, sur le même
nombre de frames. Les hommes se croisent bien plus souvent et bien plus serré — le
minimum de séparation entre partenaires descend à 0,41 m sur leur match contre 0,56 m
sur celui des femmes. Le point faible du suivi contraint est là, et un match qui le
sollicite cinq fois plus le met cinq fois plus en défaut.

Ce que le changement de match ne remet pas en cause, c'est l'ablation : le suivi
contraint garde un IDF1 trois fois supérieur à ByteTrack (0,764 contre 0,252) et ne
dépasse jamais quatre pistes, là où ByteTrack le fait sur 3 231 frames.

### Candidats de balle

La balle mesure **10 px de côté** en médiane, sur une image de deux millions de
pixels. Sur une frame figée, une ligne peinte, un logo ou un reflet lui ressemblent
exactement. Ce qui la distingue n'est pas son apparence mais son **mouvement** : la
caméra étant fixe, ce qui bouge dans l'image bouge réellement.

L'étage de détection ne tranche donc pas. Il compare chaque frame à ses deux voisines,
retient ce qui est plus clair que les deux, et renvoie une **liste de candidats
classés**. Choisir lequel est la balle revient à l'étage de trajectoire.

Mesuré sur la tranche d'évaluation du match de réglage, 3 638 balles annotées :

| Écart temporel | Rappel 5 px | 10 px | 20 px | Rang médian | Dans le top 10 | Candidats par frame |
|---|---|---|---|---|---|---|
| 1 frame | 0,445 | 0,611 | 0,658 | 4 | 52,3 % | 59 |
| **2 frames** | **0,662** | **0,912** | **0,989** | **6** | **70,2 %** | 78 |
| 3 frames | 0,661 | 0,913 | 0,991 | 7 | 63,7 % | 81 |
| 4 frames | 0,653 | 0,907 | 0,990 | 8 | 59,9 % | 81 |

**Comparer à une frame d'écart perd un tiers des balles.** À 30 images par seconde,
une balle lente parcourt moins que son propre diamètre entre deux frames
consécutives : elle se recouvre elle-même et la différence s'annule. À deux frames
d'écart elle a bougé assez pour ne plus se chevaucher, et le rappel passe de 0,611 à
0,912.

Les écarts 2, 3 et 4 se valent sur le rappel. **L'écart 2 est retenu parce qu'il place
la balle plus haut dans la liste** — rang 6 contre 7 et 8, et 70 % de présence dans
les dix premiers contre 64 % et 60 %. À rappel égal, c'est celui qui facilite le plus
l'étage suivant.

Sur le match tenu à l'écart, 19 259 balles annotées, avec l'écart retenu :

| | Rappel 5 px | 10 px | 20 px | Rang médian | Dans le top 10 | Candidats par frame |
|---|---|---|---|---|---|---|
| Match de réglage | 0,662 | 0,912 | 0,989 | 6 | 70,2 % | 78 |
| **Match tenu à l'écart** | **0,718** | **0,928** | **0,986** | 7 | 67,5 % | 87 |

**Le rappel transfère sans perte**, et se trouve même légèrement meilleur sur le match
jamais utilisé pour régler quoi que ce soit.

**Ce chiffre est un plafond, pas une performance.** Il dit que la balle est disponible
dans la liste, jamais que quoi que ce soit l'a choisie. La difficulté réelle est dans
les deux dernières colonnes : la balle est le sixième candidat parmi **78**, et une
fois sur trois elle n'est même pas dans les dix premiers. La départager est le travail
de l'étage de trajectoire, et il n'est pas entamé ici.

La chute du rappel à 5 px — 0,662 contre 0,912 à 10 px — ne vient pas d'un biais
corrigeable. Sur 512 balles, le décalage entre le centre de la tache de mouvement et
le centre annoté vaut (−0,67, +0,88) px en moyenne, et −0,37 px une fois projeté sur
la direction de déplacement. C'est de la dispersion, de norme médiane 3,3 px, pas un
décalage systématique.

### Trajectoire de la balle

L'étage précédent rend une liste de candidats classés, la balle s'y trouvant 91 % du
temps mais au sixième rang parmi 78. Cet étage doit en tirer **une position par
frame**, ou l'absence de position. Deux méthodes sont implémentées et mesurées côte à
côte.

La **croissance gloutonne** est la ligne de base. Elle part d'un candidat, extrapole à
vitesse constante, prend le candidat le plus proche de la prédiction, tolère deux
frames manquées, et s'arrête. Les segments obtenus sont ensuite départagés, et les
retenus concaténés.

L'**optimisation globale** ne décide rien frame par frame. Elle garde les 8 meilleurs
candidats de chaque frame, y ajoute un état « absent » à coût fixe, et cherche par
programmation dynamique la suite qui minimise, sur toute la fenêtre, la somme d'un
coût d'accélération et d'un coût d'émission. L'état porte le candidat courant **et le
précédent**, ce qui suffit à connaître la vitesse, donc à pénaliser un changement
brutal sans jamais avoir eu à « suivre » quoi que ce soit.

Mesuré sur les deux matchs, la vérité terrain étant l'annotation de balle du dataset :

| | | 5 px | 10 px | 20 px | Frames couvertes |
|---|---|---|---|---|---|
| **Réglage** (3 638 balles) | Croissance gloutonne | 0,074 | 0,162 | 0,203 | 2 042 / 4 100 |
| | **Optimisation globale** | **0,534** | **0,716** | **0,764** | 4 100 / 4 100 |
| **Tenu à l'écart** (19 259 balles) | Croissance gloutonne | 0,050 | 0,109 | 0,132 | 10 769 / 21 471 |
| | **Optimisation globale** | **0,576** | **0,733** | **0,774** | 21 471 / 21 471 |

Rappel ; la précision de l'optimisation globale lui est égale, le chemin répondant sur
toutes les frames. Pour la gloutonne elle vaut 0,317 et 0,214 à 10 px.

**Le facteur est de 4,4 sur le match de réglage et de 6,7 sur le match tenu à
l'écart.** Les quatre paramètres du chemin ont été balayés sur 800 frames du seul match
féminin et n'ont pas été retouchés ensuite ; le match masculin, cinq fois plus long,
donne un résultat légèrement meilleur. La fraction du plafond capturée y est la même à
un demi-point près — 79,0 % contre 78,5 %.

**Pourquoi la ligne de base plafonne.** Trois mesures enchaînées le disent sans
ambiguïté : la balle est dans la liste de candidats **93,9 %** du temps, un segment
glouton la couvre **52,5 %** du temps, et il en reste **16,8 %** après arbitrage entre
segments. La première chute est le prix de la décision locale — une extrapolation
partie sur un mauvais candidat ne revient jamais. La seconde est le prix de
l'arbitrage : il faut choisir entre des segments concurrents sans rien savoir de ce
qui se passe ailleurs dans la séquence.

**Départager les segments par leur longueur était à l'envers.** Les segments qui
suivent réellement la balle font **27 frames** en médiane ; les autres en font **31**.
Un arc de balle est court par nature — il se termine à chaque contact — tandis qu'une
fausse piste accrochée à un élément lent peut courir indéfiniment. Le critère correct
est la **vitesse** : 13,2 px/frame pour les bons segments contre 8,8 pour les autres.
Ce seul changement fait passer la précision de 0,168 à 0,405.

**Ce que le plafond d'accélération fait, et ne fait pas.** Il était présenté au départ
comme le mécanisme central, celui qui autorise les changements de direction brutaux
aux contacts. Le balayage le dément : de 40 à l'infini, le rappel bouge d'un millième.
Il ne mord quasiment jamais, et il est conservé comme garde-fou contre une frame
pathologique, pas comme le ressort de la méthode.

**Ce qui reste à gagner.** 0,912 et 0,928 étaient disponibles dans la liste de
candidats, 0,716 et 0,733 sont capturés. L'écart — un cinquième du plafond — est ce
qui justifiera, ou non, de remplacer la détection par mouvement par un réseau.

**Réserve de méthode : la métrique récompense le fait de toujours répondre.** Une frame
sans prédiction compte comme un échec de rappel, alors qu'une position produite là où
aucune balle n'est annotée n'est pas comptabilisable — 2 212 frames dans ce cas sur le
match tenu à l'écart. Le balayage a donc trouvé optimal un coût d'absence si élevé que
le chemin ne renonce jamais, ce qui est en partie un artefact de la mesure et non une
qualité propre de la méthode. Le comparatif ci-dessus reste valide, les deux méthodes
étant jugées à la même aune, mais le 0,733 ne doit pas se lire comme « la balle est
localisée trois fois sur quatre en toute circonstance ».

### Instants de contact

L'étage précédent rend une position par frame et **ne renonce jamais** : il n'y a donc
aucun trou où lire un contact. Le critère doit porter sur la forme du chemin.

Ce qui marque un contact est un changement de direction. Mesuré en pixels il n'est pas
comparable d'un lob à un smash, donc le virage est **divisé par la vitesse qui l'a
produit** : un écart de 40 px est un coude à 5 px/frame et une broutille à 30. Les
contacts retenus sont les maxima locaux de ce rapport, un seul par fenêtre de 5 frames.

Mesuré sur les deux matchs, avec le même détecteur appliqué au chemin reconstruit et à
la balle annotée — l'écart entre les deux lignes est donc imputable à la trajectoire et
à rien d'autre :

| | Contacts | Rappel des frappes | Rebonds par échange |
|---|---|---|---|
| **Réglage** (92 frappes) — balle annotée | 192 | 0,891 | 0,89 |
| — chemin reconstruit | 244 | 0,902 | 1,23 |
| **Tenu à l'écart** (475 frappes) — balle annotée | 874 | 0,806 | 0,86 |
| — **chemin reconstruit** | **1 257** | **0,895** | **1,38** |

**Le chemin reconstruit obtient un meilleur rappel que la balle annotée. Ce n'est pas
une qualité, c'est un symptôme :** il produit 44 % de contacts en plus, et détecter
davantage fait mécaniquement monter le rappel. La colonne qui compte est la troisième.

**Une trajectoire juste à 73 % ne coûte que quelques points.** Le nombre de rebonds par
échange passe de 0,86 à 1,38 — l'excédent est l'erreur de trajectoire, et il est
mesurable comme tel plutôt que caché dans un rappel flatteur.

#### La précision ne peut pas être rapportée comme une performance

L'annotation ne marque que les contacts avec une **raquette**, sous forme
d'intervalles. Ces intervalles couvrent **48,2 %** des frames annotées du match de
réglage. Un détecteur tirant ses instants **au hasard** y obtient donc une précision de
0,485 — et le détecteur de virages appliqué à la balle parfaitement annotée en obtient
0,573. L'écart est trop mince pour démontrer quoi que ce soit.

Deux corrections ont été essayées et n'ont rien changé : un appariement un pour un
entre contacts et frappes donne le même gain, et resserrer la cible autour du centre de
l'intervalle échoue parce que l'impact ne s'y concentre pas — il se disperse sur
presque toute la largeur, écart-type 0,48 en demi-largeur.

**Le match tenu à l'écart est le meilleur instrument**, ses intervalles ne couvrant que
31,0 % des frames :

| | Précision | Au hasard | Gain |
|---|---|---|---|
| Réglage, chemin reconstruit | 0,537 | 0,428 | 1,25× |
| **Tenu à l'écart, chemin reconstruit** | **0,429** | **0,285** | **1,51×** |
| Tenu à l'écart, balle annotée | 0,501 | 0,308 | 1,63× |

Sur l'instrument le moins complaisant, le détecteur bat le hasard d'un facteur 1,5.
C'est une mesure, mais faible — et elle le restera tant que la vérité terrain manquera.
Le témoin aléatoire est calculé par le code et affiché à côté de chaque précision, pour
qu'aucun de ces chiffres ne puisse être lu isolément.

#### Ce qui remplace la précision

La physique du padel. Entre deux frappes, la balle rebondit **0 fois** (volée), **1**
(sol) ou **2** (sol puis vitre, ou l'inverse). C'est un critère que l'annotation ne
fournit pas et qu'elle ne peut pas fausser. Sur le match tenu à l'écart, la
distribution obtenue est 0 : 170, 1 : 138, 2 : 78, 3 : 40, au-delà 42 — soit **82 % des
échanges dans ce que le jeu prédit**.

C'est aussi ce critère qui a fixé le réglage, et non la métrique d'événements.

#### L'encadrement du virage absolu

Le critère relatif ne connaît que des rapports, ce qui le rend aveugle à une erreur
propre au chemin reconstruit : sa vitesse au 95ᵉ centile vaut **512 px** contre **186**
pour la balle annotée. Il fait des sauts qu'aucune balle ne fait, et chaque saut
fabrique un virage. Le virage absolu est donc encadré entre 25 et 300 px.

| | Contacts | Rappel | Rebonds par échange |
|---|---|---|---|
| Seuil relatif seul | 317 | 0,957 | 1,85 |
| **Virage encadré** | **246** | **0,902** | **1,24** |

La longue traîne — jusqu'à dix contacts entre deux frappes — disparaît avec le plafond.
C'étaient des erreurs de chemin, pas des rebonds.

#### Ce qui manque

Une vérité terrain d'instants de contact, toutes surfaces confondues. Elle n'existe
dans aucun jeu de données public de padel. C'est celle que le sous-projet C doit
produire pour classer les surfaces : sa campagne d'annotation enregistrera donc
l'**instant** en plus de la surface, et servira rétroactivement de mesure de précision
à cet étage-ci. C'est la seule que ce projet pourra produire.

### Surfaces de contact

Savoir *quand* la balle a été touchée ne dit pas *contre quoi*. Un court de padel est
fermé : la balle rebondit sur le sol, sur du verre, sur du grillage et sur des
raquettes. C'est ce que cet étage doit trancher — et c'est ce qu'un pipeline de tennis
n'a pas à faire, un court ouvert n'ayant ni vitre ni grillage.

**Pourquoi l'homographie ne suffit pas.** Elle projette sur le plan du sol. Elle est
donc exacte pour un rebond au sol et fausse pour tout contact en hauteur. Mesuré sur
194 contacts réels : **un tiers se projette hors du rectangle du court**, certains à
23 m pour un court qui en fait 20. Et la distribution est presque identique entre
frappes annotées et non-frappes — 67,3 % contre 64,3 % dans le rectangle. **La position
projetée seule ne sépare rien.**

**Ce qui la remplace.** Une caméra ne donne qu'un rayon : la balle est quelque part
dessus, et rien ne dit où. C'est vrai en vol, et ça le reste. Mais **au moment d'un
contact la balle est sur une surface**, et les surfaces d'un court sont cinq plans
connus. Un rayon et un plan se coupent en un point. La hauteur, indéterminée en
général, est déterminée précisément à l'instant qui nous intéresse.

#### Retrouver la caméra

Une homographie se contente de points au sol ; une pose de caméra ne le peut pas, un
ensemble coplanaire laissant la direction verticale libre. Les repères manquants
viennent d'une annotation manuelle des panneaux de mur : haut du verre à 3 m, haut du
grillage à 4 m, haut du filet à 0,92 m.

Résultat : caméra à **x = −0,06 m, y = −26,18 m, z = +7,86 m** — centrée sur l'axe du
court, vingt-six mètres derrière le fond proche, à près de huit mètres de haut.

Le chiffre qui engage quelque chose n'est pas celui de l'ajustement mais celui des
**points de contrôle, qui n'entrent jamais dans l'ajustement** :

| Points de contrôle | Écart médian |
|---|---|
| Au sol — filet, lignes de service | **4,2 px** |
| **En hauteur — 0,92 m à 4 m, aux deux fonds** | **8,6 px** |
| Maximum, au fond éloigné | 14,4 px |

**Ce que 8,6 px valent en mètres dépend de la profondeur** : 13 cm près de la caméra,
56 cm au fond éloigné, le facteur 4,3 déjà mesuré plus haut. Sur un seuil verre /
grillage à 3 m, c'est une incertitude d'environ 20 % au pire.

#### La règle

**Raquette** — un poignet à proximité. Le dataset fournit dix-sept points par joueur,
dont les deux poignets. Mesuré : la balle est à **50 px** du poignet le plus proche
quand une frappe est annotée, contre **168 px** sinon.

**Sol ou mur** — on coupe le rayon avec les cinq plans et on ne garde que les
intersections physiquement admissibles : devant la caméra, et dans l'étendue réelle de
la surface. La marge qui absorbe l'erreur de pose est exprimée **en mètres et jamais en
pixels** — un pixel valant 1,51 cm près et 6,47 cm loin, une marge en pixels serait
quatre fois plus laxiste au fond.

**Verre ou grillage** — une table, une fois le point d'impact connu en trois
dimensions. Fonds : verre sous 3 m. Côtés : verre à moins de 4,1 m d'un fond. Aucune
heuristique.

#### La vérité terrain, qui n'existait nulle part

Aucun jeu de données public de padel n'étiquette les surfaces de contact — le dataset
utilisé ici déclare une catégorie `Wall` et ne l'a jamais remplie. Elle a donc été
produite à la main, sur les deux matchs.

| | Match de réglage | Match tenu à l'écart |
|---|---|---|
| Contacts soumis | **194**, recensement complet | **150**, tirés de 886 |
| Contacts réels | 145 | 112 |
| Illisibles | 0 | 2 |

L'outil rejoue chaque instant en boucle, la balle marquée d'une croix fixe, et
**n'affiche jamais ce que la règle prédit**. Une vérité terrain construite sur
l'hypothèse qu'elle doit juger ne mesure que deux erreurs qui s'accordent : au
sous-projet A, corriger ce défaut avait fait passer l'IDF1 de 0,956 à 0,819.

Le match masculin compte 886 contacts, soit deux heures d'arbitrage. L'échantillon est
**stratifié**, et sa taille comme sa graine sont enregistrées dans le fichier — un
tirage qu'on ne peut pas refaire ne serait pas une mesure.

#### Ce que l'annotation mesure de l'étage précédent

**Un quart des contacts détectés n'ont pas eu lieu** : la trajectoire passait tout
droit. La précision de l'étage des contacts vaut donc **0,747** sur le match de réglage
et **0,760** sur le match tenu à l'écart — sur la balle parfaitement annotée, donc hors
de toute erreur de trajectoire.

C'est la mesure que la section précédente déclarait impossible. L'annotation de frappes
ne pouvait pas la donner : ses intervalles couvrent la moitié des frames, si bien qu'un
détecteur tirant au hasard y obtenait déjà 0,480. Celle-ci est directe.

#### Ce que la règle vaut

| | Réglage | | | Tenu à l'écart | | |
|---|---|---|---|---|---|---|
| **Classe** | **n** | **Précision** | **F1** | **n** | **Précision** | **F1** |
| raquette | 75 | 0,830 | 0,896 | 58 | 0,806 | **0,892** |
| sol | 51 | 0,842 | 0,719 | 42 | 0,923 | **0,706** |
| mur | 17 | 0,789 | 0,833 | 12 | 0,714 | **0,769** |

**Exactitude globale 0,828 en réglage, 0,821 tenu à l'écart.** Sept millièmes d'écart :
les seuils n'ont pas été surajustés au match qui a servi à les choisir.

**Grillage et filet ne sont pas mesurables.** Un exemple et deux sur le match de
réglage, aucun des deux dans l'échantillon tenu à l'écart. C'était prévu — le grillage
n'occupe que le haut des fonds et le milieu des côtés. Aucun taux n'est publié pour
eux, et leurs effectifs sont donnés plutôt que tus.

#### L'arbitrage par la profondeur

Quand le sol et un mur sont tous deux admissibles, lequel choisir ? La première version
préférait le sol, systématiquement. Mesuré, ce choix coûtait **dix murs sur dix-sept**.

La règle retenue préfère le mur lorsque le point-sol candidat tombe au-delà de
`y = −7,5 m`. Ce seuil n'est pas un nombre ajusté : la caméra étant à `y = −26,18` et
`z = 7,86`, un contact sur la vitre proche se projette au sol en

| Hauteur du contact | 0,3 m | 0,5 m | 0,8 m | **1,0 m** | 1,6 m | 2,0 m |
|---|---|---|---|---|---|---|
| y projeté | −9,36 | −8,90 | −8,17 | **−7,64** | −5,86 | −4,48 |

Le seuil sépare donc les contacts de vitre **sous 1,05 m environ**. Au-delà, la
projection entre dans le court et plus rien ne la distingue d'un rebond au sol.

| Rappel des murs | Avant | Après |
|---|---|---|
| Match de réglage | 0,412 | **0,882** |
| Match tenu à l'écart | — | **0,833** |

**Le plafond n'est pas celui du seuil mais celui de la géométrie** : les murs manqués
sont les murs hauts, et une seule caméra ne peut pas les distinguer d'un rebond.

#### Ce qui n'a pas été corrigé, et pourquoi

Deux erreurs ont été mesurées, une seule est corrigible.

La seconde est que **des rebonds au sol sont pris pour des frappes** — quinze sur le
match de réglage. Le seuil de proximité au poignet a été balayé de 50 à 120 px :

| Seuil | 50 | 60 | 70 | **80** | 90 | 100 | 120 |
|---|---|---|---|---|---|---|---|
| Exactitude | 0,745 | 0,793 | 0,828 | **0,828** | 0,828 | 0,834 | 0,786 |

C'est un plateau. Le déplacer échange des frappes contre des rebonds à somme nulle : il
faudrait un autre signal, pas un autre seuil. **Le seuil est donc resté à 80 px**, et
corriger quand même, pour annoncer deux corrections plutôt qu'une, aurait été ajuster
du bruit.

#### Une strate nommée à l'envers

Les contacts où **une seule** surface est admissible avaient été étiquetés « tranchés »,
en supposant qu'une réponse unique valait confiance. Les deux campagnes disent
l'inverse :

| | Cas isolés | Dont sans contact |
|---|---|---|
| Match de réglage | 24 | **24 (100 %)** |
| Match tenu à l'écart | 16 | **14 (88 %)** |

L'explication est géométrique. Un contact réel se produit dans le volume de jeu, où le
fond proche est toujours admissible aussi, la caméra étant derrière lui — il est
candidat pour 140 des 194 contacts du match de réglage. Un rayon qui ne rencontre
qu'une seule surface est donc un rayon qui pointe hors du jeu.

Ce n'est pas une mesure de confiance mais un **détecteur de faux positifs**, et la
strate porte désormais ce nom. Il n'est pas appliqué comme filtre : 88 % sur seize cas
ne justifie pas encore de supprimer des détections, et ce serait une décision à mesurer
pour elle-même.

## Limites connues

**Un joueur ne peut pas être suivi à travers un changement de côté.** Les quatre
emplacements désignent des moitiés de court, et le suivi refuse par construction une
observation du mauvais côté du filet — c'est ce qui lui donne son « 0 frame au-dessus
de quatre ». Le prix de cette contrainte est qu'un joueur qui change de côté change
d'emplacement. Rien dans l'image ne permettrait de le rattacher : le pipeline ne lit
ni les visages ni les numéros. Les statistiques par emplacement restent valides sur le
match entier ; les statistiques **par joueur** ne le sont qu'à l'intérieur d'un
segment entre deux changements de côté.

**Le suivi d'identité ne généralise pas aussi bien que la détection.** Sur le match
de réglage il décroche 4 fois pour 21 occasions — 19 %. Sur le match tenu à l'écart,
28 fois pour 38 occasions — **74 %**. La détection, elle, transfère sans perte, et la
localisation aussi. Un pipeline jugé sur son seul F1 de détection paraîtrait
généraliser ; il ne généralise que sur la moitié de ce qu'il fait.

**Le point faible est le croisement serré entre partenaires**, pas le raccord. Sur le
même nombre de frames, le match masculin compte 14 rapprochements contre 3, et ses
partenaires descendent à 0,41 m l'un de l'autre contre 0,56 m. C'est là que se joue
l'écart entre 19 % et 74 %, et c'est la piste à travailler en priorité.

Mesurer tout cela a coûté 266 clips d'arbitrage humain. Sur les 47 clips de la finale
masculine dont la réponse a été tracée, 11 portaient une permutation réelle : **près
d'un raccord sur quatre fait décrocher l'identité**.

**La vérité terrain d'identité dépend d'un jugement humain non reproductible.** Les
266 arbitrages ont été rendus par une seule personne, sans second annotateur, donc
sans accord inter-annotateurs à rapporter. Les sept changements de côté de la finale
féminine ont en revanche été confirmés par deux voies indépendantes — la tenue des
équipes échantillonnée sur tout le match, et le tableau d'affichage sur le passage
douteux.

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

### Ce que ce dépôt versionne

Aucune image, aucune vidéo, aucun poids de modèle. `data/` — où atterrit le dataset
téléchargé — est exclu en bloc et sans exception.

`ground_truth/` en revanche est versionné, parce que sans lui les chiffres de la
section [Évaluation](#évaluation) ne seraient pas reproductibles :

| Fichier | Contenu |
|---|---|
| `calibrations/*.json` | 23 points cliqués par vidéo — 13 au sol dont 4 de contrôle, et 10 en hauteur dont 8 de contrôle |
| `identity/*.json` | assignation des 4 emplacements sur tout le match, liste des moments douteux, et les 266 arbitrages humains |
| `surfaces/*.json` | les 194 contacts à juger et les 194 jugements rendus |

**`surfaces/` est le seul de ces fichiers qui ne dérive de rien.** Les surfaces de
contact ne sont étiquetées dans aucun jeu de données public de padel : ce fichier est
la mesure elle-même, et sans lui la section sur les surfaces ne serait qu'une règle
sans juge. Les points en hauteur de `calibrations/` sont dans le même cas — ils sont
relevés à la main sur les panneaux de mur, et sans eux la pose de caméra ne se
résoudrait pas.

Ces fichiers dérivent des annotations du dataset, en CC-BY-4.0, et n'en contiennent
aucune donnée d'image. Avec eux, reproduire l'évaluation demande de télécharger le
dataset public et de lancer la campagne — pas de refaire l'arbitrage.

**Les trois calibrations sont identiques**, et c'est intentionnel. Les deux matchs
sont filmés depuis la même position au même tournoi, et l'extrait d'essai est tiré de
la finale féminine. La calibration ajustée sur cette dernière a été transférée aux
deux autres puis vérifiée par superposition du modèle de court sur une frame de
chacune : contour, lignes de service, ligne centrale et filet tombent juste. Un seul
jeu de points cliqués couvre donc tout le dataset.

## Licence

AGPL-3.0. Voir `LICENSE`.

Ce projet dépend d'Ultralytics, distribué sous AGPL-3.0, ce qui impose cette licence
à l'ensemble. Le code n'est donc pas réutilisable dans un produit propriétaire.
