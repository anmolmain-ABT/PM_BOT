import pandas as pd
import re
import os
import openai
from datetime import date, timedelta, datetime
import requests
from slack_sdk import WebClient
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import time
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import logging
# --------------------- Load environment variables ---------------------
load_dotenv()

openai.api_type = os.getenv("openai_api_type")
openai.api_version = os.getenv("openai_api_version")
openai.api_base = os.getenv("openai_api_base")
openai.api_key = os.getenv("openai_api_key")

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
API_KEY = os.getenv("CLOCKIFY_API_KEY")
WORKSPACE_ID = os.getenv("CLOCKIFY_WORKSPACE_ID")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME")
GOOGLE_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")

slack_client = WebClient(token=SLACK_BOT_TOKEN)
app = App(token=SLACK_BOT_TOKEN)
cached_df = None

# --------------------- Slack logging helper ---------------------
def slack_log(message, channel_id=None, level="info"):
    """Send logs to Slack instead of print/logging."""
    text = f"[{level.upper()}] {message}"
    if channel_id:
        try:
            slack_client.chat_postMessage(channel=channel_id, text=text)
        except Exception:
            pass
    else:
        default_channel = os.getenv("SLACK_LOG_CHANNEL")
        if default_channel:
            try:
                slack_client.chat_postMessage(channel=default_channel, text=text)
            except Exception:
                pass

# --------------------- Keep Alive ---------------------
def keep_alive():
    url = os.getenv("RENDER_APP_URL")  # Add your Render URL in .env as RENDER_APP_URL
    if not url:
        slack_log("RENDER_APP_URL not set in .env", level="warning")
        return
    while True:
        try:
            slack_log(f"Pinging {url} to keep alive...", level="info")
            requests.get(url, timeout=10)
        except Exception as e:
            slack_log(f"Keep-alive ping failed: {e}", level="info")
        time.sleep(30 * 60)  # Ping every 30 minutes

# --------------------- Google Sheets ---------------------
def get_gsheet_client():
    creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if not creds_json:
        raise ValueError("GOOGLE_CREDENTIALS_JSON is not set in .env")
    creds_dict = json.loads(creds_json)
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client

def write_to_sheet(df):
    client = get_gsheet_client()
    sheet = client.open_by_key(GOOGLE_SHEET_ID).sheet1
    df_to_write = df.copy()
    for col in df_to_write.select_dtypes(include=['datetime', 'datetime64[ns]']):
        df_to_write[col] = df_to_write[col].dt.strftime("%m/%d/%Y")
    df_to_write = df_to_write.fillna("NA")
    df_to_write = df_to_write.replace(r'^\s*$', 'NA', regex=True)
    sheet.clear()
    sheet.update([df_to_write.columns.values.tolist()] + df_to_write.values.tolist())

def read_from_sheet():
    client = get_gsheet_client()
    sheet = client.open(GOOGLE_SHEET_NAME).sheet1
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# --------------------- Clockify Download ---------------------
def download_clockify_data(since_date=None):
    slack_log("Downloading clockify Sheet", level="info")
    today = date.today()
    start_date = (since_date.isoformat() + "T00:00:00Z") if since_date else ((today - timedelta(days=120)).isoformat() + "T00:00:00Z")
    end_date = today.isoformat() + "T23:59:59Z"
    url = f"https://reports.api.clockify.me/v1/workspaces/{WORKSPACE_ID}/reports/detailed"
    headers = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}

    all_entries = []
    page = 1
    page_size = 100
    while True:
        payload = {
            "dateRangeStart": start_date,
            "dateRangeEnd": end_date,
            "sortOrder": "DESCENDING",
            "exportType": "JSON",
            "amountShown": "HIDE_AMOUNT",
            "detailedFilter": {
                "options": {"totals": "CALCULATE"},
                "page": page,
                "pageSize": page_size,
                "sortColumn": "ID"
            }
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        entries = data.get("timeentries", data.get("timeEntries", []))
        if not entries:
            break
        all_entries.extend(entries)
        page += 1

    if not all_entries:
        return pd.DataFrame()

    formatted_rows = []
    for e in all_entries:
        start_iso = e['timeInterval']['start']
        end_iso = e['timeInterval']['end']
        start_date_str = datetime.fromisoformat(start_iso.rstrip('Z')).strftime("%m/%d/%Y") if start_iso else ""
        end_date_str = datetime.fromisoformat(end_iso.rstrip('Z')).strftime("%m/%d/%Y") if end_iso else ""
        duration_hours = (datetime.fromisoformat(end_iso.rstrip('Z')) - datetime.fromisoformat(start_iso.rstrip('Z'))).total_seconds()/3600 if start_iso and end_iso else 0
        tag_names = [tag['name'] for tag in e.get("tags", []) if 'name' in tag]
        formatted_rows.append({
            "Project": e.get("projectName", ""),
            "Client": e.get("clientName", ""),
            "Description": e.get("description", ""),
            "Task": e.get("taskName", ""),
            "User": e.get("userName", ""),
            "Tags": ", ".join(tag_names),
            "Start Date": start_date_str,
            "End Date": end_date_str,
            "Duration": round(duration_hours, 2)
        })

    return pd.DataFrame(formatted_rows)

# --------------------- Load & Cache Data ---------------------
def load_data(channel_id=None):
    slack_log("Loading clockify sheet data", channel_id)
    global cached_df
    if cached_df is not None:
        return cached_df

    if channel_id:
        slack_log("💾 Fetching and preparing Clockify data, please wait...", channel_id)

    slack_log("Preloading Clockify data...", channel_id)

    try:
        df_existing = read_from_sheet()
        if df_existing.empty:
            df_combined = download_clockify_data()
        else:
            df_existing['end date'] = pd.to_datetime(df_existing['end date'], errors='coerce')
            last_date = df_existing['end date'].max()
            df_new = download_clockify_data(since_date=last_date + timedelta(days=1))
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    except Exception:
        df_combined = download_clockify_data()

    slack_log("Standardize column names", channel_id)
    df_combined.columns = [str(col).casefold() for col in df_combined.columns]

    slack_log("Convert string columns to lowercase", channel_id)
    str_cols = df_combined.select_dtypes(include=['object']).columns
    df_combined[str_cols] = df_combined[str_cols].apply(lambda x: x.str.casefold())

    if 'user' in df_combined.columns:
        df_combined['user'] = df_combined['user'].str.replace('.', ' ', regex=False)
    slack_log("Convert date time columns", channel_id)

    for col in ['start date', 'end date']:
        if col in df_combined.columns:
            df_combined[col] = pd.to_datetime(df_combined[col], errors='coerce')

    if 'duration' in df_combined.columns:
        df_combined['duration'] = pd.to_numeric(df_combined['duration'], errors='coerce')

    if 'description' in df_combined.columns and 'task' in df_combined.columns:
        df_combined['task'] = df_combined.apply(
            lambda row: row['description'] if pd.isna(row['task']) or row['task'] == '' else row['task'],
            axis=1
        )
    slack_log("replacing null values with NA", channel_id)

    df_combined['description'] = df_combined['description'].replace('', 'NA').fillna('NA')
    df_combined['task'] = df_combined['task'].replace('', 'NA').fillna('NA')
    df_combined['tags'] = df_combined['tags'].replace('', 'NA').fillna('NA')
    df_combined['client'] = df_combined['client'].replace('', 'NA').fillna('NA')

    write_to_sheet(df_combined)
    cached_df = df_combined
    return df_combined

# --------------------- GPT Query Handling ---------------------
def gpt_response(input_str):
    # logging.info(input_str)
    response = openai.ChatCompletion.create(
        engine="gpt-4o",
        messages= [
            {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
            {"role": "user", "content": f"""
            You are a data analysis assistant.
            Your task is to generate pandas code that answers the query **precisely** using the table.
            Following is the details about each column of table:
            - project : This column will give you the name of project.
            - client : This column will give you the name of client who own this project.
            - description : The project may have different tasks , this column will giive the more informaiton about each task.
            - task : This column will give you the detail of each task that are related to the project if the Task column is empty reffer to the column Description ot find details of the task.
            - user : This column will give you the information about developer name who is working on specific Task/Project.
            - tags : The project may have different type of tags like Billable(if the hours are within threshold), Unbillable( if the hours overshoot developer post their hour in this tag), Internal(if the work done by Devloper is Internal task and it is related to own company itself), Unapproved (if the hours are not approved by anyone, then they fall into this category).
            - start date : The date when specific task is started.It is in format mm-dd-yyyy.
            - end date : The date when specific task is finished.It is in format mm-dd-yyyy.
            - duration : Total number of hours that are used to complete specific Task
            
            Rules:
            - Remember every data of the table columns are in lowercase.
            - All computations related to dates should be done in dates not datetime
            - Carefully analyze all the columns that user is expecting as output.
            - Carefully analyze the user query and output that user is expecting.
            - Use 'df' as the dataframe variable.
            - Only use the columns that are defined above.
            - Approach and create the query step by step , do not do everything in one go.
            - If using Pandas in the query Always use import pandas as pd before using it in query.
            - never use .lower() method, always use .casefold() in place of .lower() method
            - If using any other module must import it in the top, because at the end i want to execute the code.
            - Only return the exact answer to the query, nothing more.
            - Do not generate summaries, explanations, or extra text.
            - If the query asks for information not present in the table, return exactly: "Sorry i can provide you the answer from the file only".
            - Be careful with date formats, column names, and data types.
            - Always store the final output in a variable called 'answer'.
            - Do not use any ascending or descending order in code
            - If user asks for query related to last week always look for last week from Sunday to Saturday from current date.
            - If user asks for query related to last Month always look for last Month from Currnet Month, must search by first calculating the dates and then applying query, but not time only date.
            - If user asks for query related to last Qaurter always find the last quarter months acccording to indian financial year system and create query according to these months,must search by first calculating the dates and then applying query, but not time only date.
            - Always include first and last date when querying with week,month or quarter.
            - Do all date computions in MM-DD-YYYY format only.
            - If user ask to generate query based on task ,description user or project name, always use contains keyword never look for exact match.
            - If user ask to generate query based on hour related to tag , always use exact match never look for contains.
            Input:
            Query: {input_str}
            Output:
            Generate python pandas code to answer this question using 'df' as the dataframe.
            Return code that computes the answer and stores it in a variable named 'answer'.
             """}
        ]
    )
    return response.choices[0].message.content
def summarizer(table, user_query):
    response = openai.ChatCompletion.create(
        engine="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
            {"role": "user", "content": f"""
            Your job is to summarize the answer of a query. 
            Summarize the answer **precisely** according to the user's query the user query was : {user_query}. 
            The answer is provided by LLM was : {table}. 
            if this response can be represented in table, return table starting and ending with ``` adjust the table width accordingly for symmetry in column widths. otherwise return simple string
            Do not add unnecessary details. 
            """}
        ]
    )
    return response.choices[0].message.content

def clean_gpt_code(code: str,channel_id) -> str:
    # Remove backticks, python code fences, and trailing/leading whitespace
    code = re.sub(r"```python|```", "", code, flags=re.IGNORECASE).strip()
    # Remove any non-Python characters that sometimes appear
    code = re.sub(r"^\s*<.*?>\s*$", "", code, flags=re.MULTILINE)
    print(code)
    logging.info(code)
    # Ensure 'answer' variable exists
    if "answer" not in code:
        code += "\nanswer = None"
    return code


def sudo_download_file_command(channel_id):
    slack_log("Forced data refresh initiated by sudo command...", channel_id)
    try:
        df_fresh = download_clockify_data(since_date=None)
        df_fresh.columns = [str(col).casefold() for col in df_fresh.columns]
        write_to_sheet(df_fresh)
        global cached_df
        cached_df = df_fresh
        slack_log("Forced data refresh completed successfully.", channel_id)
        app.client.chat_postMessage(channel=channel_id, text="✅ All Clockify data downloaded successfully!")
    except Exception as e:
        slack_log(f"Error during forced data refresh: {e}", channel_id, "info")
        app.client.chat_postMessage(channel=channel_id, text=f"❌ Failed to refresh data: {e}")

# --------------------- Slack Bot ---------------------
@app.event("message")
def handle_message(message, say):
    user_text = message.get("text")
    channel_id = message.get("channel")
    slack_log(f"Query: {user_text}", channel_id)
    if not user_text:
        slack_log("Empty message received. Ignoring.", channel_id, "warning")
        return

    prompt = str(user_text).casefold()
    if user_text.strip().casefold() == "sudo downloadfiledatatilltoday":
        slack_log("Received sudo command to force data refresh.", channel_id)
        sudo_download_file_command(channel_id)
        return

    data = load_data(channel_id)
    slack_log("Data loaded.", channel_id)
    slack_log(data.head(10), channel_id)

    processing_message = app.client.chat_postMessage(channel=channel_id, text="💭 Processing your request... please wait.")

    retries = 5
    delay = 2
    slack_log("Attempting to generate Python Script.", channel_id)

    for attempt in range(1, retries + 1):
        try:
            slack_log(user_text, channel_id)
            raw_code = gpt_response(user_text)
            slack_log(raw_code, channel_id, "info")

            pandas_code = clean_gpt_code(raw_code, channel_id)
            local_vars = {"df": data}
            try:
                exec(pandas_code, {}, local_vars)
            except SyntaxError as e:
                slack_log("❌ SyntaxError in GPT-generated code:", channel_id, "info")
                slack_log(pandas_code, channel_id)
                raise e

            answer = local_vars.get("answer", "No answer returned.")
            result = answer.to_string(index=False) if isinstance(answer, pd.DataFrame) else str(answer)
            summarized = summarizer(result, prompt)
            slack_log(summarized, channel_id, "info")

            slack_log(summarized.capitalize(), channel_id)

            app.client.chat_update(channel=channel_id, ts=processing_message["ts"], text=f"{summarized}")
            break
        except Exception as e:
            slack_log(f"Attempt {attempt} failed: {e}", channel_id, "info")
            time.sleep(delay)
            if attempt == retries:
                app.client.chat_update(
                    channel=channel_id,
                    ts=processing_message["ts"],
                    text=f"❌ Failed to process your request: {e} {e.with_traceback}"
                )

# --------------------- Run Slack Bot ---------------------
if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
