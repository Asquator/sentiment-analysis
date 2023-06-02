import sys

import os
import tensorflow as tf

# suppressing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from model import SentimentVerboseModel

# SentimentUI defines basic operations on the sentiment neural network model and a simple
class SentimentUI:
    def __init__(self):
        self.model = SentimentVerboseModel()
        self.print_help()

    def check_trained(self):
        if not self.model.is_trained():
            raise RuntimeError("Model is not trained yet")

    def retrain_model(self):
        self.model.verbose_train()
        print("Model was retrained and updated")

    def test_predict(self):
        results = {0: "Negative", 1: "Positive"}  # result to string

        self.check_trained()
        res = self.model.predict([input("Enter your sentence: >")])
        print(results[res[0]])  # printing the result in human-readable form

    def predict(self):
        self.check_trained()

        # getting file names
        inp = input("Input file: >")
        out = input("Output file: >")

        # opening input and output files, making predictions and writing results
        with open(inp, "r") as inf, open(out, "w") as outf:
            lines = inf.read().split('\n')  # separate lines from input file
            pred = self.model.predict(lines)
            # writing each prediction on a new line to the output
            outf.writelines([str(val) + '\n' for val in pred])

    def load_model(self):
        path = input("Enter path : >") # path to the model
        self.model.load_model(path)
        print("Model was loaded")

    def save_model(self):
        self.check_trained()
        path = input("Enter path : >") # path to the model
        self.model.save(path)
        print("Model was saved")

    def parse_loop(self):

        # Infinite parser loop, in case of a thrown exception, print it to console
        while True:
            try:
                self.parse_command(input(">").split())

            except Exception as e:
                print(e)

    def print_help(self):
        print("Available commands:")
        print("help: print this menu")
        print("test: test the current model with one sentence")
        print("predict: make predictions from input file and save them to output file")
        print("retrain: discard the current model and train a new one in a verbose mode")
        print("save: save the current model")
        print("load: load model")
        print("exit: exit the user cli")

    # get a command and apply the corresponding operation
    def parse_command(self, command):
        if len(command) == 0:
            return

        elif command[0] == "help":
            self.print_help()

        elif command[0] == "test":
            self.test_predict()

        elif command[0] == "retrain":
            self.retrain_model()

        elif command[0] == "predict":
            self.predict()

        elif command[0] == "save":
            self.save_model()

        elif command[0] == "load":
            self.load_model()

        elif command[0] == "exit":
            sys.exit(0)

        else:
            print("Invalid command")
            self.print_help()


if __name__ == '__main__':
    cli = SentimentUI()
    cli.parse_loop()
