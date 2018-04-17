import pickle

with open('annotated_train_set.p', "rb") as input_file:
    e = pickle.load(input_file)