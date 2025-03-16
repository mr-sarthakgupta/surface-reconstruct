import os
import glob
import re

def find_lowest_j():
    # Get all mesh files in the intermediates directory
    mesh_files = os.listdir("intermediates")
    
    chamferlist = []

    for name in mesh_files:
        chamferlist.append(name.split("_")[2].split(".")[1])

    lowest_j = min(chamferlist)

    lowest_index = chamferlist.index(lowest_j)
    print(mesh_files[lowest_index])


if __name__ == "__main__":
    lowest_file = find_lowest_j()