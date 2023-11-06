#  Initiate API
#  pip install kaggle, go to your kaggle 'Account' > API > Create New API Token.
#  Copy the downloaded kaggle.json file to C:\Users\your_user\.kaggle.
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# download to the project directory as a zip file
# example : https://www.kaggle.com/competitions/playground-series-s3e24
api = KaggleApi()
api.authenticate()
api.competition_download_files('playground-series-s3e24')

#  Unzip file
with zipfile.ZipFile('playground-series-s3e24.zip', 'r') as zipref:
    zipref.extractall('data/')

os.remove("playground-series-s3e24.zip")
