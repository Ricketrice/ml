# import pandas as pd

# df = pd.read_excel("Scats Data October 2006.xls", sheet_name="Data", header=1)
# volColumns = []
# for col in df.columns:
#     if str(col).startswith("V"):
#         volColumns.append(col)
# filterData = df[["SCATS Number", "Date"] + volColumns]

# #Test 1: Data Processing test 
# def testDataProcessing(data):
#     checkColumn = ["SCATS Number", "Date", "TimeInterval", "Volume", "Time"]
#     for col in checkColumn:
#         if col not in data.columns:
#             print("TC01 FAIL: ", col, "is missing")
#             return 
#     print("Test case(Data Processing): Pass")

# testDataProcessing(longData)