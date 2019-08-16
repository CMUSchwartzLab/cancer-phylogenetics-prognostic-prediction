""" Prepare and preprocess various feature types: survival, clinical, driver,
  two-node and multi-node.
"""

import pickle

from utils_feature import (Survival_Data_Processor, get_clinical_feature,
  get_driver_feature, get_twonode_feature, get_multinode_feature)


print("Extracting OS/DFS time of sampes...")
for cancer in ["BRCA", "LUAD", "LUSC"]:
  sdp = Survival_Data_Processor("", cancer)
  refined = sdp.run()
  refined.to_csv(
      "data/survival/"+cancer.lower()+".csv",
      sep=",", header=True)

print("Extracting clinical feature of samples...")
for cancer in ["BRCA", "lung"]:
  data, feature_lut = get_clinical_feature("", cancer)
  data.to_csv(
      "data/clinical/"+cancer.lower()+".csv",
      sep=",", header=True)
  pickle.dump(feature_lut, open("data/clinical/"+cancer.lower()+".pkl","wb"), protocol=2)

# Below is for feature extraction of ICGC data: driver, two-node and multi-node feature.
# TCGA data is similar with minor revisions.
# Specifically, TCGA uses hg38 instead of hg19.
print("Extracting driver feature of samples...")
for cancer in ["BRCA", "lung"]:
  data = get_driver_feature(cancer)
  data.to_csv(
      "data/driver/"+cancer.lower()+".csv",
      sep=",", header=True)

print("Extracting two-node feature of samples...")
for cancer in ["BRCA", "lung"]:
  data = get_twonode_feature(cancer)
  data.to_csv(
      "data/twonode/"+cancer.lower()+".csv",
      sep=",", header=True)

print("Extracting multi-node feature of samples...")
for cancer in ["BRCA", "lung"]:
  data = get_multinode_feature(cancer)
  data.to_csv(
      "data/multinode/"+cancer.lower()+".csv",
      sep=",", header=True)


