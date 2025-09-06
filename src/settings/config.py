# src/settings/config.py
import os
from dotenv import load_dotenv

load_dotenv()  # carrega o .env

# variável de ambiente direta
ENVIRONMENT = os.getenv("ENVIRONMENT", "production").lower()

def get_env_mode():
    return ENVIRONMENT
