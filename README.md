# Modélisation des risques opérationnels bancaires

Détection et modélisation des risques opérationnels bancaires avec le machine learning (régression logistique, Random Forest, Gradient Boosting), dans une architecture fréquence/sévérité inspirée de la Loss Distribution Approach, avec interprétabilité SHAP.

Projet issu du mémoire de recherche « Data analytics et modélisation des risques opérationnels », Master Monnaie Banque Finance Assurance parcours Économie et Gestion des Risques Financiers (EGRF), Université de Rouen Normandie, 2025-2026.

## Contexte

Le risque opérationnel est le plus difficile à mesurer des grands risques bancaires. Depuis 2017, la méthode standardisée (SMA) a remplacé les approches BIA, TSA et AMA. Elle assure la comparabilité entre banques, mais elle est instable d'une année à l'autre, pénalise les établissements répartis sur plusieurs entités et ignore la qualité des contrôles internes. En parallèle, le machine learning reste le plus souvent cantonné à des usages ponctuels comme la détection de fraude.

## Problématique et objectifs

Dans quelle mesure le machine learning permet-il de mieux détecter et caractériser le risque opérationnel qu'une approche économétrique classique, tout en gardant une interprétabilité compatible avec les exigences de gouvernance ?

Le projet cherche à comparer les capacités prédictives d'un logit, d'un Random Forest et d'un Gradient Boosting, à vérifier la stabilité des déterminants du risque face à la non-stationnarité des séries, et à tester une modélisation séparée de la fréquence et de la sévérité des pertes.

## Données

Le projet utilise la base publique P-COLD (Zhu, Chang et Li, 2024), qui recense des événements de perte opérationnelle du secteur bancaire chinois entre 1986 et 2023. Elle est disponible sur Figshare (https://doi.org/10.6084/m9.figshare.23940069.v3). Après apurement, l'échantillon compte 3 722 événements sur 3 725 au départ.

- Variables cibles : type d'événement bâlois, ligne métier et montant de la perte
- Transformation logarithmique du montant (log_loss) pour réduire l'asymétrie extrême (skewness de 53,8 avant, 0,33 après)
- Valeurs manquantes sur les dates conservées sous forme de variables dérivées plutôt que supprimées
- Variable « Causal factor » écartée, car elle détermine presque mécaniquement le type d'événement (fuite de données)

## Méthodologie

1. Analyse exploratoire et tests de stationnarité (ADF) sur les séries annuelles de fréquence et de sévérité
2. Régression logistique (fraude contre non-fraude), interprétée en odds ratios
3. Random Forest avec échantillon découpé de façon stratifiée (2 791 observations d'entraînement, 931 de test) et classes pondérées
4. Gradient Boosting en deux volets, classification pour la fréquence et régression sur log_loss pour la sévérité
5. Test de robustesse avec une variable de contrôle temporelle (décennie)
6. Interprétabilité avec SHAP (TreeSHAP)

## Résultats

| Modèle | Accuracy | AUC |
|---|---|---|
| Régression logistique | 83,0 % | 0,862 |
| Random Forest | 83,7 % | 0,867 |
| Gradient Boosting | 83,8 % | 0,873 |

- Les trois modèles convergent sur les mêmes déterminants. Le montant de la perte et l'appartenance à la banque de détail dominent dans tous les cas (odds ratio de 16,1 pour la banque de détail et de 1,45 par unité de log_loss dans le logit).
- Le gain apporté par le machine learning existe, mais il reste modeste par rapport au logit.
- Les séries de fréquence et de sévérité sont non stationnaires, et la base présente une troncature après 2019 liée à son mode de collecte (presse).
- L'ajout de la décennie améliore les trois modèles (accuracy de 84,6 % pour le logit, 85,7 % pour le Random Forest, 85,4 % pour le Gradient Boosting) sans changer la hiérarchie des facteurs.
- Volet sévérité (Gradient Boosting) : RMSE de 2,45, MAE de 1,83 et R² de 0,40 sur l'échelle logarithmique. La ligne métier banque commerciale domine (74,5 % de l'importance), suivie de la fraude interne.

## Interprétabilité

SHAP permet de décomposer chaque prédiction en contributions par variable, ce qui répond à l'exigence de justification documentée des décisions automatisées. Les résultats confirment l'importance du montant de perte et de la banque de détail pour la classification, et de la banque commerciale pour la sévérité.

## Limites

- Troncature de la base après 2019
- Peu de variables explicatives disponibles après l'exclusion de « Causal factor »
- Base limitée au secteur bancaire chinois, ce qui appelle une validation externe sur d'autres systèmes bancaires

## Pistes d'amélioration

Enrichir les variables avec des données textuelles (NLP sur la description des incidents) ou macro-financières, modéliser explicitement la troncature (correction de type Heckman ou pondération inverse), et transposer le modèle en outil de scoring où SHAP documenterait chaque alerte.

## Technologies

Python, pandas, NumPy, scikit-learn, statsmodels, SHAP, Matplotlib

## Auteur

Hamed ABIKANLOU, ingénieur statisticien-économiste (finance et actuariat), Master EGRF, Université de Rouen Normandie.
