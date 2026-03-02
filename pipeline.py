import prefect as pf
import pandas as pd
from pymongo import MongoClient
import requests
from dotenv import load_dotenv
import os
from prefect import task,flow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Llamar API

@task
def fetch_countries_data():
    url = "https://restcountries.com/v3.1/region/africa"
    params = {}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"Successfully fetched {len(data)} countries")
        return data
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
    except requests.exceptions.ConnectionError:
        print("Connection error: could not reach the API")
    except requests.exceptions.Timeout:
        print("Request timed out")
    except requests.exceptions.RequestException as e:
        print(f"Unexpected error: {e}")

    return None

# Cargar datos API a un dataframe local

@task
def load_to_dataframe(data):
    df = pd.json_normalize(data)
    print(df.head())
    print(df.shape)
    return df

# Transform datos

@task
def transform_data(df):
    # Select relevant columns
    cols_to_keep = [
        'name.common', 'cca3', 'capital', 'region', 'subregion',
        'population', 'area', 'landlocked', 'borders', 'latlng',
        'flag', 'idd.root', 'idd.suffixes'
    ]

    # Add dynamic columns (languages, currencies, gini)
    language_cols = df.filter(like='languages.').columns.tolist()
    currency_cols = df.filter(like='currencies.').columns.tolist()
    gini_cols = df.filter(like='gini.').columns.tolist()

    all_cols = cols_to_keep + language_cols + currency_cols + gini_cols

    # Keep only columns that actually exist in the df
    all_cols = [col for col in all_cols if col in df.columns]
    df_clean = df[all_cols].copy()

    # Rename for clarity
    df_clean.rename(columns={
        'name.common': 'country',
        'cca3': 'code',
        'idd.root': 'phone_root',
        'idd.suffixes': 'phone_suffixes'
    }, inplace=True)

    # Drop rows where country name is missing
    df_clean.dropna(subset=['country'], inplace=True)

    # Combine phone root and suffixes into one column
    df_clean['phone_code'] = df_clean['phone_root'] + df_clean['phone_suffixes'].apply(lambda x: x[0] if isinstance(x, list) else '')

    # Drop the separate columns
    df_clean.drop(columns=['phone_root', 'phone_suffixes'], inplace=True)

    # Collapse languages into a list
    lang_cols = [col for col in df_clean.columns if col.startswith('languages.')]
    df_clean['languages'] = df_clean[lang_cols].apply(
        lambda row: [v for v in row.dropna().values.tolist()], axis=1
    )
    df_clean.drop(columns=lang_cols, inplace=True)

    # Collapse currencies into a list of dicts
    currency_codes = set(col.split('.')[1] for col in df_clean.columns if col.startswith('currencies.'))

    def extract_currencies(row):
        result = []
        for code in currency_codes:
            name_col = f'currencies.{code}.name'
            symbol_col = f'currencies.{code}.symbol'
            if name_col in row and pd.notna(row[name_col]):
                result.append({'code': code, 'name': row[name_col], 'symbol': row.get(symbol_col)})
        return result

    currency_cols = [col for col in df_clean.columns if col.startswith('currencies.')]
    df_clean['currencies'] = df_clean.apply(extract_currencies, axis=1)
    df_clean.drop(columns=currency_cols, inplace=True)

    # Collapse gini into a single dict
    gini_cols = [col for col in df_clean.columns if col.startswith('gini.')]
    df_clean['gini'] = df_clean[gini_cols].apply(
        lambda row: {col.split('.')[1]: v for col, v in row.items() if pd.notna(v)}, axis=1
    )
    df_clean.drop(columns=gini_cols, inplace=True)

    # Reset index
    df_clean.reset_index(drop=True, inplace=True)

    print(f"Cleaned DataFrame: {df_clean.shape[0]} countries, {df_clean.shape[1]} columns")
    return df_clean

# Cargar datos a mongo, se añadió un upsert porque duplicaba datos en cada run.

@task
def load_to_mongodb(df_clean):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    records = df_clean.to_dict('records')
    for record in records:
        collection.update_one(
            {"code": record["code"]},
            {"$set": record},
            upsert=True
        )

    print(f"Upserted {len(records)} documents into MongoDB")
    client.close()

@flow(name="africa_pipeline")
def pipeline():
    countries = fetch_countries_data()
    if countries:
        df = load_to_dataframe(countries)
        df_clean = transform_data(df)
        load_to_mongodb(df_clean)

if __name__ == "__main__":
    pipeline()
