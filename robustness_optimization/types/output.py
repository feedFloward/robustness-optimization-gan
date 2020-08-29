import io
import sys
import time
import os
import matplotlib.pyplot as plt

DIR_NAME = r'output/run_'+time.strftime('%H_%M_%S', time.localtime())+'/'

HISTORY_FILE = DIR_NAME + 'history.txt'

if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)


class DualOutput(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.logfile = HISTORY_FILE
    def write(self, message):
        self.terminal.write(message)
        with open(self.logfile, 'a') as f:
            f.write(message)
    def flush(self):
        self.terminal.flush()
        # self.logfile.flush()

    


def write_to_output(func):
    def output_wrapper():
        with open(HISTORY_FILE, 'a') as f:
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout
            func()
            buffer = new_stdout.getvalue()
            sys.stdout = old_stdout
            print(buffer)
            f.write(buffer)
    return output_wrapper


def write_results(history):
    print("best factor config:")
    print(history['best_factor_config'])
    print('worst noise design:')
    print(history['worst_noise_design'].state)

    plt.ylabel("SN ratio")
    plt.xlabel("# iteration")
    plt.plot(history['sn_history'])
    plt.savefig(DIR_NAME+'sn_ratio.png')