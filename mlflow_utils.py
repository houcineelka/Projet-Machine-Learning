"""
MLflow Utilities Module - Heart Disease Prediction Project
==========================================================
Module réutilisable pour la configuration et le tracking MLflow.

Auteur: Projet Master ISI - Machine Learning
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
import os
import tempfile
from datetime import datetime


# =============================================================================
# 1. INITIALISATION MLFLOW
# =============================================================================

def init_mlflow(tracking_uri="file:./mlruns", experiment_name=None):
    """
    Initialise MLflow avec l'URI de tracking et l'expérience.

    Parameters:
    -----------
    tracking_uri : str
        URI pour le stockage des runs (défaut: local ./mlruns)
    experiment_name : str
        Nom de l'expérience MLflow

    Returns:
    --------
    experiment_id : str
        ID de l'expérience créée/récupérée

    Exemple:
    --------
    >>> experiment_id = init_mlflow(experiment_name="Heart_Disease_Classification")
    """
    # Configurer l'URI de tracking
    mlflow.set_tracking_uri(tracking_uri)

    if experiment_name:
        # Créer ou récupérer l'expérience
        experiment = mlflow.set_experiment(experiment_name)
        print(f"✓ MLflow initialisé")
        print(f"  - Tracking URI: {tracking_uri}")
        print(f"  - Expérience: {experiment_name}")
        print(f"  - Experiment ID: {experiment.experiment_id}")
        return experiment.experiment_id

    return None


# =============================================================================
# 2. ORGANISATION DES EXPÉRIENCES
# =============================================================================

# Structure recommandée pour les expériences
EXPERIMENTS = {
    "eda": "Heart_Disease_01_EDA",
    "feature_engineering": "Heart_Disease_02_Feature_Engineering",
    "clustering": "Heart_Disease_03_Clustering",
    "classification": "Heart_Disease_04_Classification",
    "regression": "Heart_Disease_05_Regression",
    "hyperparameter_tuning": "Heart_Disease_06_Hyperparameter_Tuning",
    "final_comparison": "Heart_Disease_07_Final_Comparison"
}


def get_experiment_name(phase):
    """
    Récupère le nom de l'expérience pour une phase donnée.

    Parameters:
    -----------
    phase : str
        Phase du projet ('eda', 'feature_engineering', 'clustering',
                        'classification', 'regression', 'hyperparameter_tuning',
                        'final_comparison')

    Returns:
    --------
    str : Nom de l'expérience
    """
    return EXPERIMENTS.get(phase, f"Heart_Disease_{phase}")


# =============================================================================
# 3. FONCTION RÉUTILISABLE POUR TRACKER LES MODÈLES SKLEARN
# =============================================================================

def track_sklearn_model(model, X_train, X_test, y_train, y_test,
                        model_name, task_type="classification",
                        log_model=True, additional_params=None,
                        additional_metrics=None, tags=None):
    """
    Fonction réutilisable pour tracker n'importe quel modèle sklearn.

    Parameters:
    -----------
    model : sklearn estimator
        Modèle sklearn (entraîné ou non)
    X_train, X_test : array-like
        Données d'entraînement et de test
    y_train, y_test : array-like
        Labels d'entraînement et de test
    model_name : str
        Nom du modèle pour le run
    task_type : str
        'classification', 'regression', ou 'clustering'
    log_model : bool
        Si True, enregistre le modèle dans MLflow
    additional_params : dict
        Paramètres supplémentaires à logger
    additional_metrics : dict
        Métriques supplémentaires à logger
    tags : dict
        Tags à ajouter au run

    Returns:
    --------
    dict : Dictionnaire contenant les métriques et le run_id

    Exemple:
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> rf = RandomForestClassifier(n_estimators=100, random_state=42)
    >>> results = track_sklearn_model(
    ...     model=rf,
    ...     X_train=X_train, X_test=X_test,
    ...     y_train=y_train, y_test=y_test,
    ...     model_name="Random_Forest",
    ...     task_type="classification"
    ... )
    """
    results = {"model_name": model_name}

    with mlflow.start_run(run_name=model_name):
        # Entraîner le modèle si pas déjà fait
        if not hasattr(model, 'classes_') and task_type == "classification":
            model.fit(X_train, y_train)
        elif not hasattr(model, 'coef_') and not hasattr(model, 'feature_importances_'):
            try:
                model.fit(X_train, y_train)
            except:
                pass  # Déjà entraîné

        # Logger les paramètres du modèle
        params = model.get_params()
        for param_name, param_value in params.items():
            try:
                mlflow.log_param(param_name, param_value)
            except:
                mlflow.log_param(param_name, str(param_value))

        # Logger les paramètres supplémentaires
        if additional_params:
            for param_name, param_value in additional_params.items():
                mlflow.log_param(param_name, param_value)

        # Logger les tags
        if tags:
            mlflow.set_tags(tags)
        mlflow.set_tag("task_type", task_type)
        mlflow.set_tag("model_type", type(model).__name__)

        # Prédictions
        y_pred = model.predict(X_test)

        # Calculer et logger les métriques selon le type de tâche
        if task_type == "classification":
            metrics = _compute_classification_metrics(model, X_test, y_test, y_pred)
        elif task_type == "regression":
            metrics = _compute_regression_metrics(y_test, y_pred)
        else:
            metrics = {}

        # Logger les métriques
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Logger les métriques supplémentaires
        if additional_metrics:
            for metric_name, metric_value in additional_metrics.items():
                mlflow.log_metric(metric_name, metric_value)

        results["metrics"] = {**metrics, **(additional_metrics or {})}

        # Logger le modèle
        if log_model:
            mlflow.sklearn.log_model(model, "model")

        results["run_id"] = mlflow.active_run().info.run_id

    return results


def _compute_classification_metrics(model, X_test, y_test, y_pred):
    """Calcule les métriques de classification."""
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    # ROC-AUC si possible
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            if len(np.unique(y_test)) == 2:
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba, multi_class='ovr')
    except:
        pass

    return metrics


def _compute_regression_metrics(y_test, y_pred):
    """Calcule les métriques de régression."""
    return {
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }


# =============================================================================
# 4. FONCTIONS POUR LOGGER LES ARTEFACTS
# =============================================================================

def log_confusion_matrix(y_test, y_pred, class_names=None, filename="confusion_matrix.png"):
    """
    Crée et log une matrice de confusion comme artefact MLflow.

    Parameters:
    -----------
    y_test : array-like
        Vraies valeurs
    y_pred : array-like
        Prédictions
    class_names : list
        Noms des classes
    filename : str
        Nom du fichier pour l'artefact

    Exemple:
    --------
    >>> with mlflow.start_run():
    ...     log_confusion_matrix(y_test, y_pred, class_names=['No Disease', 'Disease'])
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names or np.unique(y_test),
                yticklabels=class_names or np.unique(y_test))
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédiction')
    plt.ylabel('Vraie Valeur')
    plt.tight_layout()

    # Sauvegarder temporairement et logger
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(filepath)

    plt.close()


def log_roc_curve(model, X_test, y_test, filename="roc_curve.png"):
    """
    Crée et log une courbe ROC comme artefact MLflow.

    Parameters:
    -----------
    model : sklearn estimator
        Modèle avec predict_proba
    X_test : array-like
        Données de test
    y_test : array-like
        Vraies valeurs
    filename : str
        Nom du fichier pour l'artefact
    """
    if not hasattr(model, 'predict_proba'):
        print("Le modèle ne supporte pas predict_proba, courbe ROC non générée.")
        return

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    plt.tight_layout()

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(filepath)

    plt.close()


def log_feature_importance(model, feature_names, filename="feature_importance.png", top_n=15):
    """
    Log l'importance des features comme artefact MLflow.

    Parameters:
    -----------
    model : sklearn estimator
        Modèle avec feature_importances_ ou coef_
    feature_names : list
        Noms des features
    filename : str
        Nom du fichier pour l'artefact
    top_n : int
        Nombre de features à afficher
    """
    # Récupérer les importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        print("Le modèle ne supporte pas feature_importances_ ou coef_")
        return

    # Créer DataFrame et trier
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(top_n)

    # Visualisation
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(filepath)

    plt.close()


def log_dataframe(df, filename="data.csv"):
    """
    Log un DataFrame comme artefact CSV.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à logger
    filename : str
        Nom du fichier pour l'artefact
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        df.to_csv(filepath, index=False)
        mlflow.log_artifact(filepath)


def log_figure(fig, filename="figure.png"):
    """
    Log une figure matplotlib comme artefact.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure à logger
    filename : str
        Nom du fichier pour l'artefact
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(filepath)


def log_dict_as_json(data, filename="data.json"):
    """
    Log un dictionnaire comme artefact JSON.

    Parameters:
    -----------
    data : dict
        Dictionnaire à logger
    filename : str
        Nom du fichier pour l'artefact
    """
    import json
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        mlflow.log_artifact(filepath)


# =============================================================================
# 5. TRACKING POUR CLUSTERING
# =============================================================================

def track_clustering_model(model, X, model_name, true_labels=None,
                          additional_params=None, log_model=True):
    """
    Fonction pour tracker les modèles de clustering.

    Parameters:
    -----------
    model : sklearn clustering estimator
        Modèle de clustering
    X : array-like
        Données
    model_name : str
        Nom du modèle
    true_labels : array-like, optional
        Labels vrais pour métriques supervisées
    additional_params : dict
        Paramètres supplémentaires
    log_model : bool
        Si True, enregistre le modèle

    Returns:
    --------
    dict : Résultats avec métriques et run_id
    """
    results = {"model_name": model_name}

    with mlflow.start_run(run_name=model_name):
        # Fit et prédiction
        if hasattr(model, 'fit_predict'):
            labels = model.fit_predict(X)
        else:
            model.fit(X)
            labels = model.labels_

        # Logger les paramètres
        params = model.get_params()
        for param_name, param_value in params.items():
            try:
                mlflow.log_param(param_name, param_value)
            except:
                mlflow.log_param(param_name, str(param_value))

        if additional_params:
            for param_name, param_value in additional_params.items():
                mlflow.log_param(param_name, param_value)

        mlflow.set_tag("task_type", "clustering")
        mlflow.set_tag("model_type", type(model).__name__)

        # Calculer les métriques
        n_clusters = len(np.unique(labels[labels >= 0]))
        mlflow.log_metric("n_clusters", n_clusters)

        metrics = {"n_clusters": n_clusters}

        if n_clusters >= 2 and n_clusters < len(X):
            try:
                silhouette = silhouette_score(X, labels)
                mlflow.log_metric("silhouette_score", silhouette)
                metrics["silhouette_score"] = silhouette
            except:
                pass

            try:
                db_score = davies_bouldin_score(X, labels)
                mlflow.log_metric("davies_bouldin_score", db_score)
                metrics["davies_bouldin_score"] = db_score
            except:
                pass

            try:
                ch_score = calinski_harabasz_score(X, labels)
                mlflow.log_metric("calinski_harabasz_score", ch_score)
                metrics["calinski_harabasz_score"] = ch_score
            except:
                pass

        results["metrics"] = metrics
        results["labels"] = labels

        if log_model:
            mlflow.sklearn.log_model(model, "model")

        results["run_id"] = mlflow.active_run().info.run_id

    return results


# =============================================================================
# 6. COMPARAISON DES MODÈLES
# =============================================================================

def get_all_runs(experiment_name=None, experiment_id=None):
    """
    Récupère tous les runs d'une expérience.

    Parameters:
    -----------
    experiment_name : str
        Nom de l'expérience
    experiment_id : str
        ID de l'expérience

    Returns:
    --------
    pd.DataFrame : DataFrame avec tous les runs
    """
    if experiment_name:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id

    if experiment_id:
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        return runs

    return pd.DataFrame()


def compare_models(experiment_name, metric_name="accuracy", ascending=False):
    """
    Compare tous les modèles d'une expérience par une métrique.

    Parameters:
    -----------
    experiment_name : str
        Nom de l'expérience
    metric_name : str
        Nom de la métrique pour le tri
    ascending : bool
        Ordre de tri

    Returns:
    --------
    pd.DataFrame : Comparaison triée des modèles
    """
    runs = get_all_runs(experiment_name=experiment_name)

    if runs.empty:
        print(f"Aucun run trouvé pour l'expérience: {experiment_name}")
        return pd.DataFrame()

    # Filtrer les colonnes de métriques
    metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
    param_cols = [col for col in runs.columns if col.startswith('params.')]

    # Sélectionner les colonnes importantes
    cols_to_keep = ['run_id', 'tags.mlflow.runName', 'start_time'] + metric_cols

    comparison = runs[cols_to_keep].copy()
    comparison.columns = [col.replace('metrics.', '').replace('tags.mlflow.', '')
                         for col in comparison.columns]

    # Trier par la métrique spécifiée
    sort_col = metric_name
    if sort_col in comparison.columns:
        comparison = comparison.sort_values(sort_col, ascending=ascending)

    return comparison


def get_best_model(experiment_name, metric_name="accuracy", ascending=False):
    """
    Récupère le meilleur modèle d'une expérience.

    Parameters:
    -----------
    experiment_name : str
        Nom de l'expérience
    metric_name : str
        Métrique pour le classement
    ascending : bool
        Si True, le plus petit est meilleur

    Returns:
    --------
    dict : Informations sur le meilleur modèle
    """
    comparison = compare_models(experiment_name, metric_name, ascending)

    if comparison.empty:
        return None

    best_run = comparison.iloc[0]

    return {
        "run_id": best_run['run_id'],
        "run_name": best_run.get('runName', 'Unknown'),
        "metric_value": best_run.get(metric_name, None),
        "all_metrics": best_run.to_dict()
    }


def load_best_model(experiment_name, metric_name="accuracy", ascending=False):
    """
    Charge le meilleur modèle d'une expérience.

    Parameters:
    -----------
    experiment_name : str
        Nom de l'expérience
    metric_name : str
        Métrique pour le classement
    ascending : bool
        Si True, le plus petit est meilleur

    Returns:
    --------
    sklearn model : Modèle chargé
    """
    best = get_best_model(experiment_name, metric_name, ascending)

    if best:
        model_uri = f"runs:/{best['run_id']}/model"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Modèle chargé: {best['run_name']}")
        print(f"  {metric_name}: {best['metric_value']}")
        return model

    return None


# =============================================================================
# 7. SCRIPT DE COMPARAISON GLOBALE
# =============================================================================

def generate_comparison_report():
    """
    Génère un rapport de comparaison de tous les modèles du projet.

    Returns:
    --------
    dict : Rapport avec les meilleurs modèles par phase
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "experiments": {}
    }

    # Configuration des métriques par type de tâche
    metric_config = {
        "classification": ("accuracy", False),
        "regression": ("r2", False),
        "clustering": ("silhouette_score", False),
        "feature_engineering": ("n_features_selected", True),
        "hyperparameter_tuning": ("best_score", False)
    }

    for phase, experiment_name in EXPERIMENTS.items():
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            continue

        # Déterminer la métrique appropriée
        metric_name, ascending = metric_config.get(phase, ("accuracy", False))

        comparison = compare_models(experiment_name, metric_name, ascending)

        if not comparison.empty:
            best = comparison.iloc[0]
            report["experiments"][phase] = {
                "experiment_name": experiment_name,
                "n_runs": len(comparison),
                "best_model": {
                    "run_name": best.get('runName', 'Unknown'),
                    "run_id": best['run_id'],
                    f"best_{metric_name}": best.get(metric_name, None)
                },
                "all_models": comparison[['runName', metric_name]].to_dict('records') if metric_name in comparison.columns else []
            }

    return report


def print_comparison_report():
    """
    Affiche un rapport de comparaison formaté.
    """
    report = generate_comparison_report()

    print("=" * 70)
    print("RAPPORT DE COMPARAISON - Heart Disease ML Project")
    print(f"Généré le: {report['generated_at']}")
    print("=" * 70)

    for phase, data in report.get("experiments", {}).items():
        print(f"\n📊 {phase.upper().replace('_', ' ')}")
        print(f"   Expérience: {data['experiment_name']}")
        print(f"   Nombre de runs: {data['n_runs']}")
        print(f"   🏆 Meilleur modèle: {data['best_model']['run_name']}")

        # Afficher la métrique principale
        for key, value in data['best_model'].items():
            if key.startswith('best_'):
                print(f"      {key}: {value:.4f}" if isinstance(value, float) else f"      {key}: {value}")

    print("\n" + "=" * 70)
    print("Pour voir les détails dans MLflow UI, exécutez:")
    print("  mlflow ui --port 5000")
    print("Puis ouvrez: http://localhost:5000")
    print("=" * 70)


# =============================================================================
# 8. UTILITAIRES
# =============================================================================

def end_active_runs():
    """Termine tous les runs actifs (utile en cas d'erreur)."""
    while mlflow.active_run():
        mlflow.end_run()
    print("Tous les runs actifs ont été terminés.")


def get_run_info(run_id):
    """
    Récupère les informations détaillées d'un run.

    Parameters:
    -----------
    run_id : str
        ID du run

    Returns:
    --------
    dict : Informations du run
    """
    run = mlflow.get_run(run_id)
    return {
        "run_id": run_id,
        "run_name": run.data.tags.get('mlflow.runName', 'Unknown'),
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": dict(run.data.tags)
    }


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    print("""
    MLflow Utils - Heart Disease Project
    =====================================

    Exemple d'utilisation:

    # 1. Initialiser MLflow
    from mlflow_utils import init_mlflow, track_sklearn_model, log_confusion_matrix

    init_mlflow(experiment_name="Heart_Disease_04_Classification")

    # 2. Tracker un modèle
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=100)
    results = track_sklearn_model(
        model=rf,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        model_name="Random_Forest",
        task_type="classification"
    )

    # 3. Logger des artefacts (dans un run actif)
    with mlflow.start_run(run_name="Mon_Modele"):
        log_confusion_matrix(y_test, y_pred)
        log_roc_curve(model, X_test, y_test)
        log_feature_importance(model, feature_names)

    # 4. Comparer les modèles
    from mlflow_utils import compare_models, print_comparison_report

    comparison = compare_models("Heart_Disease_04_Classification", metric_name="f1_score")
    print(comparison)

    # 5. Générer un rapport global
    print_comparison_report()

    # 6. Lancer l'UI MLflow
    # Dans le terminal: mlflow ui --port 5000
    # Ouvrir: http://localhost:5000
    """)
