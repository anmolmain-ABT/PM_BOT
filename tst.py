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

# --------------------- Google Sheets ---------------------
def get_gsheet_client():
    if GOOGLE_CREDENTIALS_FILE is None:
        raise ValueError("GOOGLE_CREDENTIALS_FILE path is not set in .env")
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDENTIALS_FILE, scope)
    client = gspread.authorize(creds)
    return client

def write_to_sheet(df):
    client = get_gsheet_client()
    sheet = client.open_by_key(GOOGLE_SHEET_ID).sheet1

    # Convert datetime columns to string in MM/DD/YYYY format
    df_to_write = df.copy()
    for col in df_to_write.select_dtypes(include=['datetime', 'datetime64[ns]']):
        df_to_write[col] = df_to_write[col].dt.strftime("%m/%d/%Y")

    sheet.clear()
    sheet.update([df_to_write.columns.values.tolist()] + df_to_write.values.tolist())


def read_from_sheet():
    client = get_gsheet_client()
    sheet = client.open(GOOGLE_SHEET_NAME).sheet1
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# --------------------- Clockify Download ---------------------
def download_clockify_data(since_date=None):
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
    global cached_df
    if cached_df is not None:
        return cached_df

    # Send Slack message if channel_id is provided
    if channel_id:
        try:
            app.client.chat_postMessage(
                channel=channel_id,
                text="💾 Fetching and preparing Clockify data, please wait..."
            )
        except Exception as e:
            logging.warning(f"Failed to send Slack message: {e}")

    logging.info("Preloading Clockify data...")

    try:
        df_existing = read_from_sheet()
        if df_existing.empty:
            df_combined = download_clockify_data()
        else:
            df_existing['end date'] = pd.to_datetime(df_existing['end date'], format="%m/%d/%Y", errors='coerce')
            last_date = df_existing['end date'].max()
            df_new = download_clockify_data(since_date=last_date + timedelta(days=1))
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    except Exception:
        df_combined = download_clockify_data()

    df_combined.columns = [str(col).lower() for col in df_combined.columns]
    str_cols = df_combined.select_dtypes(include=['object']).columns
    df_combined[str_cols] = df_combined[str_cols].apply(lambda x: x.str.lower())
    if 'user' in df_combined.columns:
        df_combined['user'] = df_combined['user'].str.replace('.', ' ', regex=False)
    for col in ['start date', 'end date']:
        if col in df_combined.columns:
            df_combined[col] = pd.to_datetime(df_combined[col], format="%m/%d/%Y", errors='coerce')
    if 'duration' in df_combined.columns:
        df_combined['duration'] = pd.to_numeric(df_combined['duration'], errors='coerce')
    if 'description' in df_combined.columns and 'task' in df_combined.columns:
        df_combined['task'] = df_combined.apply(
        lambda row: row['description'] if pd.isna(row['task']) or row['task'] == '' else row['task'],
        axis=1
        )

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
            - If using Pandas in the query Always use import pandas as pd before using it in query .
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

def clean_gpt_code(code: str) -> str:
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
    logging.info("Forced data refresh initiated by sudo command...")
    try:
        # Force download Clockify data from scratch
        df_fresh = download_clockify_data(since_date=None)  # Download all data
        df_fresh.columns = [str(col).lower() for col in df_fresh.columns]
        write_to_sheet(df_fresh)
        global cached_df
        cached_df = df_fresh  # Update cache
        logging.info("Forced data refresh completed successfully.")
        app.client.chat_postMessage(
            channel=channel_id,
            text="✅ All Clockify data has been downloaded and sheet is updated successfully!"
        )
    except Exception as e:
        logging.error(f"Error during forced data refresh: {e}")
        app.client.chat_postMessage(
            channel=channel_id,
            text=f"❌ Failed to refresh data: {e}"
        )




# --------------------- Slack Bot ---------------------
@app.event("message")
def handle_message(message, say):
    user_text = message.get("text")
    if not user_text:
        logging.warning("Empty message received. Ignoring.")
        return

    prompt = user_text.lower()
    channel_id = message.get("channel")

    # ---------------- Force refresh command ----------------
    if user_text.strip().lower() == "sudo downloadfiledatatilltoday":
        logging.info("Received sudo command to force data refresh.")
        sudo_download_file_command(channel_id)
        return 
    data = load_data()

    processing_message = app.client.chat_postMessage(
        channel=channel_id,
        text="💭 Processing your request... please wait."
    )

    retries = 5
    delay = 2
    for attempt in range(1, retries + 1):
        try:
            raw_code = gpt_response(user_text)
            pandas_code = clean_gpt_code(raw_code)
            local_vars = {"df": data}
            try:
                exec(pandas_code, {}, local_vars)
            except SyntaxError as e:
                logging.info("❌ SyntaxError in GPT-generated code:")
                logging.info(pandas_code)
                raise e
            answer = local_vars.get("answer", "No answer returned.")

            if isinstance(answer, pd.DataFrame):
                result = answer.to_string(index=False)
            elif isinstance(answer, pd.Series):
                result = str(answer.to_dict())
            else:
                result = str(answer)
            summarized = summarizer(result, prompt)
            logging.info(summarized.capitalize())

            app.client.chat_update(
                channel=channel_id,
                ts=processing_message["ts"],
                text=f"{summarized}"
            )
            break

        except Exception as e:
            logging.info(f"Attempt {attempt} failed: {e}")
            time.sleep(delay)
            if attempt == retries:
                app.client.chat_update(
                    channel=channel_id,
                    ts=processing_message["ts"],
                    text=f"❌ Failed to process your request: {e}"
                )


# --------------------- Run Slack Bot ---------------------
if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()