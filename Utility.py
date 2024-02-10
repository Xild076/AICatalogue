import sys
import os
import time
import numpy as np
import random
from datetime import datetime
from colorama import Fore, Style


def debug(variable: str):
    print(Fore.RED + Style.BRIGHT + "DEBUG: " + Style.RESET_ALL + Fore.RED + f"Name is '{variable}', " + f"Value is '{eval(variable)}'" + Fore.RESET)

def random_name_gen():
    adjective = ['ordinary', 'wanting', 'spurious', 'fierce', 'lucky', 'groovy', 'ashamed', 'jolly', 'fearful', 'aberrant']
    noun = ['homework', 'negotiation', 'decision', 'contribution', 'tension', 'drawer', 'distribution', 'version', 'basket', 'mud']
    name = f"{random.choice(adjective)}_{random.choice(noun)}_{datetime.now().timestamp()}"
    return name