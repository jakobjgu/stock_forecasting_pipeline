import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()

load_dotenv(dotenv_path)

FRED_API_KEY=os.getenv("FRED_API_KEY")
