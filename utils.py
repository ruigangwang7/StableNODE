import torch 
import logging

class Dict2Class(object): 
    def __init__(self, dict):   
        for key in dict: 
            setattr(self, key, dict[key]) 

class Logger():
    def __init__(self, logdir):
        format_ = "[%(asctime)s] %(message)s"
        filename = f'{logdir}/log.txt'
        f = open(filename, "a")
        logging.basicConfig(filename=filename, level=20, format=format_, datefmt='%H:%M:%S')

    def __call__(self, msg):
        print(msg)
        logging.info(msg)