import gspread
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name('ForPatternLab.json', scope)
client = gspread.authorize(credentials)
sheet = client.open('Test Set').sheet1
data_From_First_Column = sheet.col_values(1)
data_From_Second_Column = sheet.col_values(2)

for i in range(len(data_From_Second_Column)):
    data_From_First_Column[i] = int(data_From_First_Column[i])
    data_From_Second_Column[i] = int(data_From_Second_Column[i])

list_to_covert_to_array = [(key, value) for (key, value) in zip(data_From_First_Column, data_From_Second_Column)]

matrix = np.array(list_to_covert_to_array)
print(matrix)