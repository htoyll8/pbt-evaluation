import os

try:
    import mlflow
except ImportError:
    mlflow = None


def track(experiment, run_name, params=None, metrics=None, artifact=None):
    """Log one run to MLflow.

    No-op when PBT_MLFLOW=0, when mlflow is not installed, or on any logging
    error, so a run never fails for tracking reasons. Defaults to a local
    sqlite backend (mlflow 3 deprecated the file store); point elsewhere with
    MLFLOW_TRACKING_URI. View runs with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
    """
    if mlflow is None or os.environ.get("PBT_MLFLOW", "1") == "0":
        return
    try:
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            if artifact and os.path.exists(artifact):
                mlflow.log_artifact(artifact)
    except Exception as e:
        print(f"[WARN] MLflow logging skipped: {e}")
