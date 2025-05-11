import gdown

# 1. Download CSV file from Google Drive
file_id = "1XRX94nPkVDOh2KM_uR2NNZ8rM0qYBUKd"
url = f"https://drive.google.com/uc?id={file_id}"
output = "scholar_data.csv"
gdown.download(url, output, quiet=False)