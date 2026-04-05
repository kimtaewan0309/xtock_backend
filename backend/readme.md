## Setup

1. Create `.env` file in project root:

   ```env
   X_BEARER_TOKEN=YOUR_X_API_BEARER_TOKEN_HERE


## Create virtual environment and install dependencies:
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

## Run server:

uvicorn main:app --reload

## 25.11.26:
try yfinance

## 25.11.28:
connect mongoDB