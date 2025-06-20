import os
import subprocess
from datetime import datetime, timedelta

# Path to your local git repo
# REPO_PATH = "https://github.com/Arbazkhan-cs/Langchain"

# Dates you want to make commits on (format: YYYY-MM-DD)
dates = [
    "2025-06-18",
]

# Commit message template
commit_message = "Uploading new file"

# Change to your repo directory
# os.chdir(REPO_PATH)

# Make dummy file
filename = "contrib_file.txt"

# Make a commit on each date
for date_str in dates:
    with open(filename, "a") as f:
        f.write(f"Commit on {date_str}\n")

    subprocess.run(["git", "add", filename])
    
    # Format the date
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    formatted_date = date_obj.strftime("%a %b %d %H:%M:%S %Y +0530")

    # Make commit with date override
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = formatted_date
    env["GIT_COMMITTER_DATE"] = formatted_date

    subprocess.run(["git", "commit", "-m", commit_message.format(date_str)], env=env)

# Optional: Push to GitHub (make sure remote is set)
subprocess.run(["git", "push"])
