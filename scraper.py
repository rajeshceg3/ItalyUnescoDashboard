import os
import datetime

LOG_FILE = "scraper_log.txt"

def write_log(message):
    """Writes a message to the log file with a timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"{timestamp} - {message}\n"
    print(f"Attempting to log: {full_message}") # Print to console for direct feedback if possible
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(full_message)
        print(f"Successfully wrote to {LOG_FILE}")
    except Exception as e:
        print(f"CRITICAL FALLBACK: Could not write to log file {LOG_FILE}. Error: {e}. Original message: {message}")

if __name__ == "__main__":
    # Attempt to clear the log file first, with console print for feedback
    print(f"Checking for existing log file: {LOG_FILE}")
    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
            print(f"Successfully cleared old log file: {LOG_FILE}")
        except Exception as e_remove:
            print(f"Could not remove old log file {LOG_FILE}: {e_remove}")
    else:
        print(f"Log file {LOG_FILE} does not exist, no need to clear.")

    write_log("Minimal scraper.py test script started and logging initialized.")
    write_log("Test log entry.")
    print("Minimal scraper.py test script finished.")
