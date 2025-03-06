DATABASES={
    "319838f7f95a4c2688b63420c0325ec8": "FlyHostel1_3X_2024-12-10_16-00-00",
    "4c7c952ce5444477a13a33f58dfe124a": "FlyHostel3_3X_2024-12-10_16-00-00",
    "8e8850218cbe441b8fe038ae96a8f445": "FlyHostel3_3X_2024-12-17_16-00-00",
    "96a9fc27118e470d836053c2d38040bf": "FlyHostel3_6X_2023-08-31_13-00-00",
    "336a1cf967794d938daad8852ecc2b7b": "FlyHostel4_3X_2024-12-10_16-00-00",
}

import os
import requests
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

auth_token = os.environ["NOTION_TOKEN"]


def query_database(database_id, auth_token):

    # Notion API endpoint for querying a database
    url = f"https://api.notion.com/v1/databases/{database_id}/query"
    
    # Headers for the API request
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }


    next_cursor = None
    has_more = True
    all_results=[]
    while has_more:
        payload = {"start_cursor": next_cursor} if next_cursor else {}
    
        # Make a POST request to query the database
        response = requests.post(url, headers=headers, json=payload)
        
        # Check for a successful response
        if response.status_code != 200:
            print("Error querying the database:", response.json())
        
        # Parse the JSON response
        data = response.json()
        results = data.get("results", [])
        all_results.append(results)
        next_cursor = data.get("next_cursor")
        has_more = data.get("has_more", False)
    
    return all_results


def parse_row(row):
    """
    Read the data in one row of the database into a list of Python primitives
    """
    
    data=[]
    for prop in row:
        prop_type=row[prop]["type"]
        
        if prop_type=="number":
            value=row[prop]['number']
        elif prop_type=="rich_text":
            # print(row[prop]["rich_text"][0])
            if row[prop]["rich_text"]:
                value=row[prop]["rich_text"][0]["plain_text"]
            else:
                value=np.nan
        elif prop_type=="title":
            if row[prop]['title']:
                value=row[prop]['title'][0]["text"]["content"]
            else:
                value=np.nan

        elif prop_type=="select":
            value=row[prop]['select']
            if value is not None:
                value=value["name"]

        elif prop_type=="multi_select":
            if row[prop]['multi_select']:
                names=[token["name"] for token in row[prop]['multi_select']]
                value=";".join(names)
            else:
                value=np.nan
        else:
            raise ValueError(f"{prop} of type {row[prop]['type']} not supported")
        data.append(value)
    return data

def parse_database_page(results):

    rows=[]
    for item in results:
        row = parse_row(item["properties"])
        rows.append(row)
    
    # Get column headers from the first row (if available)
    if results:
        headers = results[0]["properties"].keys()
        df=pd.DataFrame.from_records(rows, columns=headers)
    else:
        print("No data found.")
        df=None
    
    return df

def parse_database(results):
    pages=[]
    for result in results:
        pages.append(parse_database_page(result))

    df=pd.concat(pages, axis=0).reset_index(drop=True)
    return df

def download_database(database_id, token):
    results=query_database(database_id, token)
    df=parse_database(results)
    return df


def generate_database(token, out):
    dfs={}
    for database_id, database_name in tqdm(DATABASES.items()):
        dfs[database_name]=download_database(database_id, token)

    df=[]
    for experiment in dfs:
        dff=dfs[experiment]
        dff["experiment"]=experiment
        df.append(dff)
    df=pd.concat(df, axis=0)
    df=df.loc[~df["label"].isna()]
    df.to_csv(out)