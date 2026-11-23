### Identification du problème
Le dataset Iris contient des fleurs réparties en trois variétés : Setosa, Versicolor et Virginica. L'objectif est de prédire la variété d'une fleur à partir de quatre caractéristiques : la longueur et la largeur des sépales, ainsi que la longueur et la largeur des pétales.

Il s'agit donc d'un problème de classification supervisée, car les variétés sont déjà connues et indiquées dans le dataset. Le but est d'apprendre au modèle à reconnaître ces catégories pour pouvoir en prédire de nouvelles.

Ce problème n'est pas du clustering car les catégories ne sont pas à découvrir, ni de la régression car on cherche à prédire une catégorie et non une valeur numérique.

### Modèle choisi
Plusieurs modèles peuvent être utilisés pour classifier les variétés d'Iris, comme l'arbre de décision, la régression logistique ou le Random Forest.

L'arbre de décision est simple à comprendre, mais il peut être moins stable car il repose sur un seul arbre. La régression logistique est également adaptée à la classification, mais elle modélise moins facilement les relations complexes entre les caractéristiques.

Le modèle Random Forest Classifier a donc été choisi. Il combine plusieurs arbres de décision, ce qui permet d'obtenir des prédictions généralement plus stables qu'avec un arbre seul. Il est également capable de modéliser des relations non linéaires et de gérer les trois catégories d'Iris.

Enfin, une recherche des meilleurs hyperparamètres est réalisée avec GridSearchCV et une validation croisée afin de sélectionner une configuration adaptée aux données.


### Recherche d'hyperparamètres
Le Random Forest est optimisé par `GridSearchCV`, selon la méthode vue dans les corrections du cours. Une validation croisée à 5 plis est utilisée. Les hyperparamètres testés sont :
- `n_estimators` : nombre d'arbres 
- `max_depth` : profondeur maximale des arbres 
- `min_samples_leaf` : nombre minimal d'observations dans une feuille 
- `max_features` : variables considérées à chaque séparation.

La métrique principale est l'accuracy.

L'association d'hyperparamètres ayant obtenu la meilleure accuracy moyenne en validation croisée est conservée. Le modèle final, les noms des caractéristiques, les noms des classes et les résultats d'évaluation sont sauvegardés dans `model/iris_model.joblib`. Les résultats détaillés sont affichés dans le terminal.

Le modèle enregistré contient directement le Random Forest entraîné et optimisé. L'API utilise donc exactement cette instance pour ses prédictions.

### Résultats observés
Pour cette exécution, les meilleurs hyperparamètres sont : `n_estimators=100`, `max_depth=None`, `min_samples_leaf=2` et `max_features="sqrt"`. Le modèle obtient une accuracy moyenne de 96.67 % en validation croisée et atteint une accuracy de 96.67 % sur les 30 observations du jeu de test.

