# Requirements
host_name = "192.168.137.1"
user_name = "root"
user_password = "12341234"
database_name = "trial3" #overall database name
table_name = "exit_entry_logs" #table name


topic = "sample"





import mysql.connector
from datetime import datetime, timedelta

def send_to_db_exit(bssids):
    # Define your database connection
    conn = mysql.connector.connect(
        host=host_name,
        user=user_name,
        password=user_password,
        database=database_name
    )
    cursor = conn.cursor()

    # Get current time
    curr_time = datetime.now()
    
    for bssid in bssids:
         # Check if bssid exists in mac_roll
        cursor.execute("SELECT mac FROM mac_roll WHERE mac = %s", (bssid,))
        registered = cursor.fetchone()
        
        if not registered:
            print(f"Unregistered MAC found: {bssid}")
            continue
        
        
        # Fetch the last entry for the given MAC ID
        query = """
        SELECT exit_datetime, entry_datetime 
        FROM exit_entry_logs 
        WHERE mac = %s 
        ORDER BY exit_datetime DESC 
        LIMIT 1;
        """
        cursor.execute(query, (bssid,))
        last_entry = cursor.fetchone()

        # Condition to check if we should insert or skip
        if last_entry:
            last_exit, last_entry_time = last_entry  # Unpack values

            # If entry_datetime is NULL, skip inserting
            if last_entry_time is None:
                print(f"Skipping {bssid}: Previous entry has entry_datetime = NULL.")
                continue

            # Check if either exit_datetime or entry_datetime is within the last 3 minutes
            time_threshold = curr_time - timedelta(minutes=3)
            if (last_exit and last_exit >= time_threshold) or (last_entry_time and last_entry_time >= time_threshold):
                print(f"Skipping {bssid}: Last entry was less than 3 minutes ago.")
                continue

        # If no condition met, insert a new record
        insert_query = """
        INSERT INTO exit_entry_logs (mac, exit_datetime, entry_datetime) 
        VALUES (%s, %s, %s);
        """
        cursor.execute(insert_query, (bssid, curr_time.strftime("%Y-%m-%d %H:%M:%S"), None))
        print(f"Inserted {bssid} with exit_datetime = {curr_time}")

    # Commit and close connection
    conn.commit()
    cursor.close()
    conn.close()

    print("Processing completed!")





import mysql.connector
from datetime import datetime, timedelta

def send_to_db_entry(bssids):
    # Define your database connection
    conn = mysql.connector.connect(
        host=host_name,
        user=user_name,
        password=user_password,
        database=database_name
    )
    cursor = conn.cursor()

    # Get current time
    curr_time = datetime.now()

    for bssid in bssids:
        # Check if bssid exists in mac_roll
        cursor.execute("SELECT mac FROM mac_roll WHERE mac = %s", (bssid,))
        registered = cursor.fetchone()
        
        if not registered:
            print(f"Unregistered MAC found: {bssid}")
            continue

        # Fetch the most recent log for this MAC
        query = """
        SELECT exit_datetime, entry_datetime 
        FROM exit_entry_logs 
        WHERE mac = %s 
        ORDER BY exit_datetime DESC 
        LIMIT 1;
        """
        cursor.execute(query, (bssid,))
        last_entry = cursor.fetchone()

        if not last_entry:
            print(f"Entry not found for {bssid}.")
            continue

        last_exit, last_entry_time = last_entry

        # If entry_datetime is already filled
        if last_entry_time is not None:
            print(f"Entry not found for {bssid}: Most recent record already has entry_datetime.")
            continue

        # If exit was done within the last 3 minutes
        time_threshold = curr_time - timedelta(minutes=3)
        if last_exit and last_exit >= time_threshold:
            print(f"Skipping {bssid}: Exit was done less than 3 minutes ago.")
            continue

        # Update the entry_datetime for this record
        update_query = """
        UPDATE exit_entry_logs 
        SET entry_datetime = %s 
        WHERE mac = %s AND exit_datetime = %s;
        """
        cursor.execute(update_query, (curr_time.strftime("%Y-%m-%d %H:%M:%S"), bssid, last_exit))
        print(f"Updated {bssid} with entry_datetime = {curr_time}")

    # Commit and close connection
    conn.commit()
    cursor.close()
    conn.close()

    print("Entry processing completed!")





import subprocess
import pandas as pd
import re

# Define command and file paths
command = "mosquitto_sub -t sample -h 192.168.137.1"  # Adjust if necessary
output_file = r"E:\Acads\6th_Sem\DA353_IoT\Project\terminal\output.txt"
csv_file = r"E:\Acads\6th_Sem\DA353_IoT\Project\terminal\output.csv"

responses = []  # List to store responses

# Regular expression pattern for BSSID
bssid_pattern = r"([0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2})"

try:
    with open(output_file, "w") as file:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print("Listening for messages... Press 'Ctrl+C' to terminate.")

        for line in process.stdout:
            line = line.strip()
            responses.append(line)
            file.write(line + "\n")  # Write to file

            print(line)  # Print raw line
            print("*" * 50)

            # Extract BSSIDs
            bssids = re.findall(bssid_pattern, line)
            print("Extracted BSSIDs:", bssids)

            # Decide which function to call based on first character
            if line.startswith("1"):
                print("                                                                                -----EXIT LOG----")
                send_to_db_exit(bssids)
            elif line.startswith("0"):
                print("                                                                                -----ENTRY LOG----")
                send_to_db_entry(bssids)
            else:
                print("Unknown prefix — skipping line.")

            print("=" * 50)

except KeyboardInterrupt:
    print("\nProcess interrupted. Stopping execution.")
    process.terminate()
    process.wait()  # Ensure the process fully stops

finally:
    file.close()  # Ensure the file is closed properly

    # Save responses to CSV
    df = pd.DataFrame(responses, columns=["Response"])
    # df.to_csv(csv_file, index=False)
    print(f"\nResponses saved to {csv_file}")
