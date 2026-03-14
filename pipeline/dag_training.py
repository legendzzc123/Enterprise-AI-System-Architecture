from datetime import datetime, timedelta
import logging
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Default arguments for the DAG
default_args = {
    'owner': 'Lead-AI-Systems-Engineer',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def extract_data(**kwargs):
    logging.info("Extracting data from source database...")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    # Store data for next task (Simulated with local save)
    joblib.dump((X_train, X_test, y_train, y_test), '/tmp/training_data.pkl')
    logging.info("Data extraction complete.")

def train_model(**kwargs):
    logging.info("Training model using RandomForest...")
    X_train, X_test, y_train, y_test = joblib.load('/tmp/training_data.pkl')
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    # Store model for next task
    joblib.dump(model, '/tmp/model_candidate.pkl')
    logging.info("Model training complete.")

def evaluate_and_register(**kwargs):
    logging.info("Evaluating model candidate against production benchmarks...")
    X_train, X_test, y_train, y_test = joblib.load('/tmp/training_data.pkl')
    model = joblib.load('/tmp/model_candidate.pkl')
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model accuracy: {accuracy:.4f}")

    if accuracy >= 0.9:
        logging.info("Model passed validation. Registering to model registry...")
        # Simulated registry logic
        joblib.dump(model, '/tmp/registered_model.pkl')
    else:
        logging.warning("Model failed to meet production accuracy threshold.")
        raise ValueError("Model validation failed: accuracy below threshold.")

# Define DAG
dag = DAG(
    'enterprise_model_training_v1',
    default_args=default_args,
    description='Automated Training Pipeline for Model Serving',
    schedule_interval=timedelta(days=1),
)

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate_and_register',
    python_callable=evaluate_and_register,
    dag=dag,
)

# Task dependencies
extract_task >> train_task >> evaluate_task
