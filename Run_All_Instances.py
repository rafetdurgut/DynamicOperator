from itertools import product
import os
import threading
import time

def thread_function(conf):
    time.sleep(1)
    print(conf)
    os.system(f"Python ./Run_Problem.py {conf}")

for p_no in range(5):
    x = threading.Thread(target=thread_function, args=(p_no,))
    x.start()