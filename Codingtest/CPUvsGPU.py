import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import os
import time

# Main function
def main():
    # Set GPU device
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    a = torch.rand(50,30)
    b = torch.rand(50,30)
    c = torch.tensor(30)

    # starttime = time.time()
    # res1=a*30
    # print("res_1: ", time.time()-starttime)

    starttime = time.time()
    res2=torch.mul(a,c)
    print("res_2: ", time.time()-starttime)

    starttime = time.time()
    res3=a.mul_(c)
    print("res_3: ", time.time()-starttime)



if __name__ == "__main__":
    main()
