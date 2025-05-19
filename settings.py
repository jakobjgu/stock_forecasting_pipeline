import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

dotenv_path = find_dotenv()

load_dotenv(dotenv_path)

FRED_API_KEY = os.getenv("FRED_API_KEY")
ROOT_PATH = Path(os.getenv("ROOT_PATH")).resolve()
