###########
# Imports #
###########
import pandas as pd
import numpy as np
import pickle

# Google API
from googleapiclient.discovery import build
from google.oauth2 import service_account

# Twilio API
import os
from twilio.rest import Client

from dotenv import load_dotenv  # for loading .env file with account info

#############
# Load Data #
#############
save_path = "/Users/djw/Documents/pCloud_synced/Academics/Projects/2020_thesis/thesis_experiments/3_experiments/3_3_experience_sampling/3_3_2_processed_data/"

# load pickle
in_file = open(save_path + "app_data.pkl", "rb")
app_data = pickle.load(in_file)

# just need the indication of survey task completion
df = app_data["SurveyTasks"]

# for tasks/surveys pull out dates after March 1
df["DueDate"] = pd.to_datetime(df.DueDate)
cols = df["DueDate"] > pd.to_datetime("28/2/2021").tz_localize("US/Eastern")
df = df[cols]

####################
# Participant Info #
####################
# path to google sheets
cred_path = "/Users/djw/Documents/pCloud_synced/Academics/Projects/2020_thesis/thesis_experiments/3_experiments/3_3_experience_sampling/3_3_3_data_analysis/resources/client_secret.json"

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SERVICE_ACCOUNT_FILE = cred_path

creds = None
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

# The spreadsheet ID
SPREADSHEET_ID = "1NXMvrji3WeLm5JvVO2BJNkoEpj2HmDXuvqTenDsi1i4"

service = build("sheets", "v4", credentials=creds)

# Call the Sheets API
sheet = service.spreadsheets()
result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range="Sheet1!A:D").execute()

values = result.get("values", [])
contacts = pd.DataFrame(values[1:], columns=values[0])

#################
# Response Rate #
#################
# Dataframe giving counts of each status (Complete, Incomplete, Closed)
dfg = pd.DataFrame(
    df.groupby("ParticipantIdentifier")["Status"].apply(lambda x: x.value_counts())
).reset_index()

# Find completed tasks by subject
# rename columns (notice that I am effectively "moving" the status column...)
dfg = dfg.rename(columns={"Status": "CompleteCount", "level_1": "Status"})
# df that only has the complete count
completes = dfg.loc[
    dfg.Status == "Complete", ["ParticipantIdentifier", "CompleteCount"]
]

# Find total tasks by subject
totals = pd.DataFrame(
    dfg.groupby("ParticipantIdentifier")["CompleteCount"].agg(np.sum)
).reset_index()
# df that has the total assigned tasks for each participant
totals = totals.rename(columns={"CompleteCount": "TotalCount"})

# left join complete and total columns
response_rate = totals.merge(completes, how="left", on="ParticipantIdentifier")

# Calculate completion rate by subject
response_rate["CompRate"] = response_rate.CompleteCount / response_rate.TotalCount

# remove unneccesary columns
response_rate = response_rate[["ParticipantIdentifier", "CompRate"]]

###########
# Streaks #
###########

# simplify timestamp
df = df.assign(DueDate2=df.DueDate.apply(lambda x: str(x).split(" ")[0]))

# get status by participant and day
dfg = pd.DataFrame(
    df.groupby(["ParticipantIdentifier", "DueDate2"])["Status"].apply(
        lambda x: x.value_counts()
    )
).reset_index()

# remove and rename columns
dfg = dfg.drop(columns="Status")
dfg = dfg.rename(columns={"level_2": "Status"})

# clean things up and get the first version of status
dfg = dfg.drop_duplicates(
    ["ParticipantIdentifier", "DueDate2"], keep="first"
).reset_index(drop=True)

# replace Status with 1 for complete and 0 otherwise
di = {"Complete": 1, "Closed": 0, "Incomplete": 0}
dfg = dfg.replace({"Status": di})

# find start of streak by comparing with next row (is it the same?)
dfg["StartOfStreak"] = dfg.groupby("ParticipantIdentifier")["Status"].apply(
    lambda x: x.ne(x.shift())
)

# indicate whether given row is "start" of streak
dfg["StreakID"] = dfg["StartOfStreak"].cumsum()

# sum up length of streak
dfg["StreakCounter"] = dfg.groupby("StreakID").cumcount() + 1

# find each subjects current streak
streaks = (
    dfg.groupby("ParticipantIdentifier")
    .apply(lambda x: x.iloc[-1])
    .reset_index(drop=True)
)

# remove unneccessary columns
streaks = streaks[["ParticipantIdentifier", "Status", "StreakCounter"]]

###########
# Combine #
###########
# join adherence and response rate
adherence = streaks.merge(response_rate, how="left", on="ParticipantIdentifier")

# Merge w/ contacts dataframe (imported from Google Sheet)
adherence = contacts.merge(adherence, how="left", on="ParticipantIdentifier")

###########
# Rewards #
###########
# define thresholds and payments for response
response_rates = [50, 60, 70, 80, 90, 95]
reward_rates = [25, 35, 50, 70, 100, 150]

# define reward for streak (they could select this on input...)
streak_reward = "$5 Amazon gift card"

# response
adherence["Reward"] = 0
adherence["NextRate"] = response_rates[0]
adherence["NextReward"] = reward_rates[0]
# streak
adherence["StreakText"] = ""

# Calculate rewards
for subject in range(len(adherence)):
    # response
    for i in range(len(response_rates)):
        if adherence.loc[subject, "CompRate"] * 100 >= response_rates[i]:
            adherence.loc[subject, "Reward"] = reward_rates[i]
            adherence.loc[subject, "NextRate"] = response_rates[i + 1]
            adherence.loc[subject, "NextReward"] = reward_rates[i + 1]

    # streak
    if adherence.loc[subject, "Status"] == 1:
        current_streak = adherence.loc[subject, "StreakCounter"] % 7
        if current_streak == 0:
            adherence.loc[
                subject, "StreakText"
            ] = f"Congrats, you just completed a 7 day streak!\n\nThis means you win a {streak_reward}\n\nYou can start another streak today to win more!"
            # need to automatically send out gift card, and perhaps keep track...(e.g. streaks_complete +=1)
        elif current_streak >= 5:
            adherence.loc[
                subject, "StreakText"
            ] = f"Congrats, you have completed {current_streak} days in a row!\n\nJust {7-current_streak} more day(s) and you win {streak_reward}!"
        else:
            adherence.loc[
                subject, "StreakText"
            ] = f"Good job, you have a {current_streak} day streak going..."
    if adherence.loc[subject, "Status"] == 0:
        current_streak = adherence.loc[subject, "StreakCounter"]
        if current_streak > 1 and current_streak < 6:
            adherence.loc[
                subject, "StreakText"
            ] = f"Hi, just a reminder that you have missed the last {current_streak} days in a row.\n\nIf you miss 6 days in a row we will no longer be able to use your data and will have to remove you from the experiment."
        if current_streak == 6:
            adherence.loc[
                subject, "StreakText"
            ] = f"You have missed {current_streak} days in a row. We will no longer be able to use you data. You will be contacted by the experimenter and removed from the study."
            # email/text to me
        if current_streak > 6:
            adherence.loc[subject, "StreakText"] = ""

#############
# Save File #
#############
save_path = "/Users/djw/Documents/pCloud_synced/Academics/Projects/2020_thesis/thesis_experiments/3_experiments/3_3_experience_sampling/3_3_2_processed_data/"

# save file to pickle
out_file = open(save_path + "adherence.pkl", "wb")
pickle.dump(adherence, out_file)
out_file.close()

#################
# Notifications #
#################
def adherence_notification(df):
    """Send messages to all subjects about their adherence rates."""

    for i in range(len(df)):
        # Get info on subject
        tel_number = df.loc[i, "PhoneNumber"]

        current_rate = df.loc[i, "CompRate"] * 100  # convert to %
        current_reward = df.loc[i, "Reward"]
        next_rate = df.loc[i, "NextRate"]
        next_reward = df.loc[i, "NextReward"]
        streak = df.loc[i, "StreakText"]

        # create message text
        body1 = f"STREAK\n{streak}\n\n"
        body2 = f"COMPLETION RATE\nYour current rate is {current_rate:.0f}%...\nBased on your current rate you will receive a ${current_reward} bonus when you finish.\n\n"
        body3 = f"If you can increase your rate to {next_rate}% you will receive a bonus of ${next_reward}!\n\n"

        # Send message using Twilio
        load_dotenv()  # take environment variables from .env file

        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
        client = Client(account_sid, auth_token)

        message = client.messages.create(
            body=body1 + body2 + body3,
            from_="+12897242427",  # my Twilio number
            to=tel_number,
        )

        # print(message.sid)


adherence_notification(adherence)