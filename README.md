# Student mental health risk prediction

Simple Python versions of the preprocessing, training, and optimization notebooks.

## Run everything with Docker

```bash
docker compose up --build -d
```

- Streamlit: http://localhost:8501
- Airflow: http://localhost:8081

Airflow uses its simple local `standalone` mode with SQLite, so there are only two containers. Its database, login details, and task logs persist in a Docker volume.

Log in to Airflow as `admin`. Read its generated password with:

```bash
docker compose exec airflow cat /opt/airflow/state/simple_auth_manager_passwords.json.generated
```

Open `mental_health_pipeline`, enable it, and click **Trigger** for the first run. It runs `prepare_data → train_models → evaluate_model` and then runs weekly while enabled. Open a task's logs to inspect its output or a failure. Preprocessing is fitted inside the training pipelines.

Streamlit displays your saved model's test metrics and confusion matrix. Its prediction form passes the 15 inputs to the saved pipeline. Refresh the page after an Airflow run to see updated results. The shared `models/` and `reports/` folders persist on your machine.

```bash
docker compose logs -f airflow
docker compose down
```

`down` stops the services without removing the saved database or model files. This is a local learning setup. Ports are bound to localhost.

On Linux, the default container user is UID 1000, matching this project owner's account. If your UID differs, start with `AIRFLOW_UID=$(id -u) docker compose up --build -d` so Airflow can write to the shared folders.

## Files

- `src/preprocessing.py`: loads the CSV, makes the same 80/20 split, and builds preprocessing.
- `src/train.py`: compares three models, tunes Logistic Regression and SVM, and saves the winner.
- `src/evaluate.py`: evaluates the saved model and saves metrics, a confusion matrix, and error analysis.
- `dags/mental_health_dag.py`: calls the Python functions in order using `PythonOperator`.
- `app/streamlit_app.py`: results dashboard and prediction form.

The code follows the notebook steps, with comments marking each section. The SVM uses the calibration setup from notebook 4. Outliers are retained and no resampling is applied, as decided in the notebooks.

## Run

To run the Python scripts without Docker:

From the project root:

```bash
conda activate mhrp
python -m src.preprocessing
python -m src.train
python -m src.evaluate
```

To run Streamlit without Docker:

```bash
python -m pip install -r requirements-app.txt
streamlit run app/streamlit_app.py
```

Models are saved in `models/` and results in `reports/`. Training and evaluation replace their previous output files. Use the original CSV and split settings when evaluating a saved model.

## Import the functions

```python
from src.preprocessing import load_data, build_preprocessor
from src.train import train_models
from src.evaluate import evaluate_model
```

Preprocessing is fitted inside the model pipelines. The test set is used only for final evaluation, not for selecting settings.

The synthetic data and predictions are educational demonstrations, not clinical assessments.
