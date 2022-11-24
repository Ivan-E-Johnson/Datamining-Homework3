# This is a sample Python script.
import os
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import json
def read_data():
    path = "restaurants/train/"
    data_dict = {}
    all_files = {}
    reviews = {}
    for file in os.listdir(path):
        with open(path + file) as project_file:
            data = json.load(project_file)
            data_dict["Reviews"] =  pd.json_normalize(data)
            reviews[file] = pd.json_normalize(data)
            data_dict["Resturaunt"] = data["RestaurantInfo"]
            #data_dict["Reviews"] = pd.DataFrame.from_dict(data["Reviews"])
            all_files[file] = data_dict
        #df = pd.read_json(path + file , lines=True , orient="columns")
        #df = pd.read_json(path + file)
    return all_files


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    all_files = read_data()
    print("Finished")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
