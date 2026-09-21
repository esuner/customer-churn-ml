"""Verifica que MLflow se conecta al servidor de DagsHub con las credenciales del .env."""
import mlflow
from dotenv import load_dotenv

load_dotenv()

mlflow.set_experiment("conexion-test")
with mlflow.start_run(run_name="prueba-conexion"):
    mlflow.log_param("origen", "check_mlflow.py")
    mlflow.log_metric("ok", 1.0)

print("Conexion OK. Mira la pestana Experiments en DagsHub.")
