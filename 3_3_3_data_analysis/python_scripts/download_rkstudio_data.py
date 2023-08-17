# Imports

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

import os
import time
from datetime import datetime
import pyotp  # for 2 factor
import zipfile
import glob
import pandas as pd
import pickle

# #####################
# # Scraping function #
# #####################
# def rk_studio_login(username, password, otp_code):
#     # go to site
#     driver.get("https://rkstudio.careevolution.com/inv")

#     # enter username and password, then submit
#     driver.find_element_by_id("Username").send_keys(username)
#     driver.find_element_by_id("Password").send_keys(password)
#     driver.find_element_by_id("login-button").click()

#     # get 2 factor authorization code, then submit
#     totp = pyotp.TOTP(otp_code)
#     two_factor = totp.now()
#     driver.find_element_by_id("Code").send_keys(two_factor)
#     driver.find_element_by_id("provide-one-time-password-submit").click()

#     # navigate to project
#     try:
#         # make sure page is loaded first, then pull link
#         link = WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.LINK_TEXT, "DJW Thesis"))
#         )
#         link.click()

#     except NoSuchElementException:
#         print("Something went wrong going to Thesis project...")
#         driver.quit()

#     # navigate to downloads
#     try:
#         # make sure page is loaded first, then pull link
#         link = WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.LINK_TEXT, "Export History"))
#         )
#         link.click()

#     except NoSuchElementException:
#         print("Something went wrong going to Export History...")
#         driver.quit()

#     # -----------------------------------#
#     # find date of last downloaded file #
#     # -----------------------------------#
#     current_files = []

#     # find all files in raw_data/testing directory
#     for root, dirs, files in os.walk(download_path):
#         for filename in files:
#             if filename.endswith(".zip"):
#                 filename = filename.split("-")[1].split(".")[0]
#                 current_files.append(filename)

#     # put files in order by date
#     current_files.sort()

#     # get the last element in sorted list and convert to datetime
#     last_download = datetime.strptime(current_files[-1], "%Y%m%d")

#     # ----------------------------------#
#     # Download new data from RK Studio #
#     # ----------------------------------#

#     # get buttons of files (to click)
#     try:
#         # make sure page loads first...
#         buttons = WebDriverWait(driver, 10).until(
#             EC.presence_of_all_elements_located(
#                 (
#                     By.XPATH,
#                     "//button[@class='editor-controls-button' and @title='Download']",
#                 )
#             )
#         )
#         # buttons = driver.find_elements_by_xpath("//button[@class='editor-controls-button' and @title='Download']")

#         # get dates of files
#         dates = driver.find_elements_by_xpath("//*[contains(text(),'Incremental')]")
#     except NoSuchElementException:
#         print("Something went wrong getting buttons...")

#     # create list of dates to check against
#     file_dates = []

#     for date in dates:
#         # check if empty
#         if date.text:
#             date_text = (
#                 date.text.split(",")[1].split("to ")[-1]
#                 + " "
#                 + date.text.split(",")[2].split(" ")[1]
#             )
#             date_dt = datetime.strptime(date_text, "%b %d %Y")
#             file_dates.append(date_dt)

#     # download files more recent than `last_download`
#     for i in range(len(file_dates)):
#         if file_dates[i] > last_download:
#             buttons[i].click()

#     # give it time to download
#     time.sleep(30)

#     # output time completed
#     now = datetime.now()
#     # dd/mm/YY H:M:S
#     dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
#     print("Download completed: ", dt_string)

#     # end session
#     driver.quit()


# # Chromedriver path
# PATH = "/Users/djw/Documents/pCloud_synced/Academics/Projects/2020_thesis/thesis_experiments/3_experiments/3_3_experience_sampling/3_3_3_data_analysis/resources/chromedriver"

# # Login credentials
# user_auth = {
#     "Username": "djw",
#     "Password": "nFs3NdlXTKat65Auv!4Z!i#",
#     "OTP code": "dp2dL56du3nmoL2prv6x7ryfyowu5pLo",
# }

# # Set download path
# download_path = "/Users/djw/Documents/pCloud_synced/Academics/Projects/2020_thesis/thesis_experiments/3_experiments/3_3_experience_sampling/3_3_1_raw_data/testing"

# chrome_options = webdriver.ChromeOptions()
# prefs = {"download.default_directory": download_path}
# chrome_options.add_experimental_option("prefs", prefs)

# # Launch Driver
# driver = webdriver.Chrome(PATH, options=chrome_options)

# # Login and Scrape
# rk_studio_login(user_auth["Username"], user_auth["Password"], user_auth["OTP code"])


###################
# Saving function #
###################

# note that this goes through all files
# more efficient would be to just ADD new files


def create_df_dict(file_path, save_path):
    # createstartswithty dictionary
    app_data = {}

    # create list of zip files
    data_files = glob.glob(f"{file_path}/*.zip")
    data_files.sort(reverse=True)
    # move the first entry to the end as it has slighlty different syntax
    data_files = data_files[1:] + data_files[0:1]
    i = 1
    # extract all csv files from zip
    for zip_file in data_files:
        # track progress
        print(f"...Proccessing file {i} of {len(data_files)}...", end="\r")
        i += 1
        zf = zipfile.ZipFile(zip_file)

        # list of .csv files in zip
        file_names = zf.namelist()

        # loop through .csv files
        for file in file_names:
            # added because there seemed to be some weird files being included
            if not file.startswith("SurveyData/"):
                try:
                    df = pd.read_csv(zf.open(file), parse_dates=True, low_memory=False)
                    # clean up filename to use as key
                    sep = "_"
                    file = file.replace(".", "_")
                    file = file.split(sep, 1)[0]

                # some files are sometimes empty (e.g. notifications)
                except pd.errors.EmptyDataError:
                    continue  # will skip the rest of the block and move to next file

                # append each day to df
                try:
                    app_data[file] = app_data[file].append(df, ignore_index=True)
                except KeyError:
                    app_data[file] = df

    # save file to pickle
    out_file = open(save_path + "app_data.pkl", "wb")
    pickle.dump(app_data, out_file)
    out_file.close()

    # output time completed
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("File saved: ", dt_string)


# path to RKStudio exported data
file_path = "/Users/djw/Documents/pCloud_synced/Academics/Projects/2020_thesis/thesis_experiments/3_experiments/3_3_experience_sampling/3_3_1_raw_data/testing"
save_path = "/Users/djw/Documents/pCloud_synced/Academics/Projects/2020_thesis/thesis_experiments/3_experiments/3_3_experience_sampling/3_3_2_processed_data/"

# create df from all files
create_df_dict(file_path, save_path)
