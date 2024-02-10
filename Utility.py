import sys
import os
import time
import numpy as np
import inspect
import random
from colorama import Fore, Style


def debug(variable: str):
    print(Fore.RED + Style.BRIGHT + "DEBUG: " + Style.RESET_ALL + Fore.RED + f"Name is '{variable}', " + f"Value is '{eval(variable)}'" + Fore.RESET)
