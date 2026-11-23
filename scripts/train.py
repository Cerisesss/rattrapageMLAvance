import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import FEATURE_NAMES, TARGET_NAMES, load_iris_frame
from src.model import predict_model, save_model


def main() -> None:
    X, y = load_iris_frame()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    parameter_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 8],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2"],
    }
    search = GridSearchCV(
        model,
        parameter_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    
    test_evaluation = predict_model(
        best_model, X_test, y_test, TARGET_NAMES
    )
    test_accuracy = test_evaluation["accuracy"]
    
    model_details = {
        "best_cv_accuracy": float(search.best_score_),
        "best_params": search.best_params_,
        "test_accuracy": test_accuracy,
    }

    model_path = ROOT / "model" / "iris_model.joblib"
    save_model(
        model_path,
        best_model,
        FEATURE_NAMES,
        TARGET_NAMES,
        model_details,
    )

    print("\nResultats Random Forest")
    print("-" * 30)
    print(f"Accuracy validation croisee : {search.best_score_:.2%}")
    print(f"Accuracy sur le test         : {test_accuracy:.2%}")
    print("Meilleurs parametres         :")
    for name, value in search.best_params_.items():
        print(f"  {name}: {value}")

    print("\nMatrice de confusion")
    print("               setosa  versicolor  virginica")
    for species, row in zip(
        TARGET_NAMES,
        test_evaluation["confusion_matrix"],
    ):
        print(f"{species:<14} {row[0]:>7} {row[1]:>11} {row[2]:>10}")

    print(f"\nModele sauvegarde : {model_path}")


if __name__ == "__main__":
    main()
