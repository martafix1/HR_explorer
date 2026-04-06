

import numpy as np
import json


filePath = "Projects/TestProject/analysis_modules/plot_radar_ampl/data/meas7_unrosed.npz"

def loadNPZ(path) -> dict : 

    # outputDict = dict()

    print(f"Loading file {path}")
    with np.load(path, allow_pickle=True) as data:
        for key in data.files:
            value = data[key]
            if type(value).__name__ == "ndarray":
                print(f"key: {key}, type: {type(value).__name__}, shape {value.shape}, dtype: {value.dtype}, value: {str(value)[:40] :40}")
            else:
                print(f"key: {key}, type: {type(value).__name__}, value: {str(value)[:40] :40}")
        
            if key == "__metadata__":
                metadata = json.loads(value.item())
                for k, v in metadata.items():
                    print(f"METADATA: {k}, value: {v}")

        outputDict = dict(data) # needed for two reasons. 1st, the NpzDataFile must be closed, so this must be witihn the with () as data: structure which does that. So this copies it into another dict, which is irrelevant of the file and can be used however the user wants. Will take up memory ofc.
        
    return outputDict


if __name__ == "__main__":
     loadNPZ(filePath)