import json
from os.path import expanduser
from pathlib import Path
import pickle
import os


# Path(os.environ["ICHOR_OUTPUT_DATASET"]).mkdir(exist_ok=True, parents=True)
# Path(os.environ["ICHOR_LOGS"]).mkdir(exist_ok=True, parents=True)


fil = Path(os.environ["ICHOR_OUTPUT_DATASET"]) / "new" / "di.json"
# x = Path(os.environ["ICHOR_OUTPUT_DATASET"])
# print(x)


with open(fil, 'w') as f:
    print("The json file is created")

# fil1 = Path(os.environ["ICHOR_OUTPUT_DATASET"]) / "new" / "dict3.json"

# # Data to be written
# dictionary = {
#     "name": "sathiyajith",
#     "rollno": 56,
#     "cgpa": 8.6,
#     "phonenumber": "9976770500"
# }

# with open(fil, "w") as outfile:
#     json.dump(dictionary, outfile)


# # Data to be written
# dictionary = {
#     "name": "sathiyajith",
#     "rollno": 56,
#     "cgpa": 8.6,
#     "phonenumber": "9976770500"
# }


# with open(fil1, "w") as outfile:
#     json.dump(dictionary, outfile)