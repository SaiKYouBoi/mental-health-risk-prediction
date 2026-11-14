from datetime import datetime, timezone

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator


def prepare_data():
    from src.preprocessing import load_data

    X_train, X_test, y_train, y_test = load_data()
    print("Training shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Class percentages:")
    print(y_train.value_counts(normalize=True) * 100)


def train_models():
    from src.train import train_models as train

    train()


def evaluate_model():
    from src.evaluate import evaluate_model as evaluate

    evaluate()


with DAG(
    dag_id="mental_health_pipeline",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["mental-health"]
) as dag:
    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data,
        do_xcom_push=False
    )

    train = PythonOperator(
        task_id="train_models",
        python_callable=train_models,
        do_xcom_push=False
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
        do_xcom_push=False
    )

    prepare >> train >> evaluate
