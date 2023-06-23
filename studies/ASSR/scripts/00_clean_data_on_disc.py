# %%
from pathlib import Path
import fooof
import shutil
from pandas.core.frame import DataFrame
import scipy.io  as sio
import numpy as np
import pandas as pd
import os 
def copy_without_patterns(input_path, destination_path, list_of_patterns):
    shutil.copytree(input_path, destination_path, ignore=shutil.ignore_patterns(*list_of_patterns))

def copy_all_data(input_path, destination_path):
    """ Copy the data from backup to fresh folder"""
    patterns = ["@raw*", "matrix_scout*", "results*"] #pattern to exclude
    copy_without_patterns(input_path,destination_path, patterns)

def remove_brainstorm():
    """After copying data into Brainstorm project, Brainstorm config need to be deleted, not doing that will cause error when running pipeline

    """
    def remove_folder(folder_path):
        """Function for removing folders 
        
        """    
        try:
            shutil.rmtree(folder_path)
        except OSError as e:
            print("Error: %s : %s" % (folder_path, e.strerror))

    brainstorm_path = "/home/daniel/.brainstorm"
    remove_folder(brainstorm_path)
BaseFolder = "/Volumes/ExtremePro/Brainstorm"
input_path = f"{BaseFolder}/tDCS_MEG_Marinazzo/" # source path 
destination_path = f"{BaseFolder}/tDCS_MEG/" # destination path
# %% copy all data

def rename_sub3(destination_path):
    """Rename sub3 to sub03"""
    for source_file in Path(destination_path).glob("data/Subject03*/*/*bl.mat"):
        if source_file.is_file():
            print("OLD:", source_file)
            new_name = source_file.name.replace("_bl", "")
            target_file = source_file.with_name(new_name)
            print("NEW:", target_file)
            os.rename (source_file, target_file)

if __name__ == "__main__":
   copy_all_data(input_path, destination_path)
#  rename_sub3(destination_path)

# %%

# %%
