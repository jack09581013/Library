import sys
import pickle
import datetime
from colorama import Fore, Style


def print_progress(message, rate):
    '''Pring progress'''
    if rate < 0: rate = 0
    if rate > 1: rate = 1
    percent = rate * 100
    sys.stdout.write('\r')
    sys.stdout.write('{} {:.2f} % [{:<50s}]'.format(message, percent, '=' * int(percent / 2)))
    sys.stdout.flush()


def save(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def load(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


class Timer:
    def __init__(self):
        self.current_time = None

    def tic(self):
        self.current_time = datetime.datetime.now()

    def toc(self, return_timespan=False):
        if return_timespan:
            return datetime.datetime.now() - self.current_time
        else:
            print('Elapsed:', datetime.datetime.now() - self.current_time)

    @staticmethod
    def timespan_str(timespan):
        total = timespan.seconds
        second = total % 60 + timespan.microseconds / 1e+06
        total //= 60
        minute = int(total % 60)
        total //= 60
        return f'{minute:02d}:{second:05.2f}'


class ThresholdColor:
    def __init__(self, th1, th2, th3, th4):
        self.th1 = th1
        self.th2 = th2
        self.th3 = th3
        self.th4 = th4

    def get_color(self, value):
        if value >= self.th1:
            return Fore.RED
        elif value >= self.th2:
            return Fore.YELLOW
        elif value >= self.th3:
            return Fore.BLUE
        elif value >= self.th4:
            return Fore.CYAN
        else:
            return Fore.GREEN
