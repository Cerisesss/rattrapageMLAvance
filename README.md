## Structure du projet

- `src/data.py` : chargement et noms des caractéristiques du dataset Iris
- `src/model.py` : chargement, sauvegarde et predit le modele
- `scripts/train.py` : entraînement du modèle, appelle les fonctions du model.py
- `api/main.py` : API
- `justification.md` : justification du probleme et choix réalisé dans la partie 1 du projet
- `architecture.md` : architecture du projet (diagramme Mermaid) pour la partie 4 du projet

## Installation 
1. Cloner le projet :

```bash
git clone "projet_url"
```

2. Créer un environnement virtuel avec le terminal :
- Ouvrez le terminal et naviguez jusqu'au répertoire du projet. 
- Exécutez les commandes suivante pour créer et activer un environnement virtuel :

```bash
python -m venv .venv
source .venv/Scripts/activate
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Entraînement

Lancer l'entraînement du modèle avec la commande suivante :
```bash
python scripts/train.py
```

Le script affiche les résultats dans le terminal et crée :

- `model/iris_model.joblib`

## Démarrer l'API

Lancer l'api avec Docker en suivant ces étapes :
- Construire l'image Docker :
```bash
docker compose build
```
- Lancer le conteneur Docker :
```bash
docker compose up -d
```

Ouvrir le lien pour accéder à la documentation (Swagger): `http://127.0.0.1:8000/` ou `http://127.0.0.1:8000/docs`

Endpoints :

- `GET /health`
- `POST /predict`


## Doc 
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html