import json
from os.path import expanduser
from pathlib import Path
import pickle
import os


# Path(os.environ["ICHOR_OUTPUT_DATASET"]).mkdir(exist_ok=True, parents=True)
# Path(os.environ["ICHOR_LOGS"]).mkdir(exist_ok=True, parents=True)


fil = Path(os.environ["ICHOR_OUTPUT_DATASET"]) / "di.json"
# x = Path(os.environ["ICHOR_OUTPUT_DATASET"])
# print(x)


with open(fil, 'w') as f:
    print("The json file is created")


# Data to be written
dictionary = {
    "name": "sathiyajith",
    "rollno": 56,
    "cgpa": 8.6,
    "phonenumber": "9976770500"
}

# load json module
# import json

# create json object from dictionary
json = json.dumps(dictionary)

# open file for writing, "w" 
f = open(fil,"w")

# write json object to file
f.write(json)

# close file
f.close()
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