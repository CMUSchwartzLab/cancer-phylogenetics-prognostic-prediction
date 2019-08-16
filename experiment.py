""" After the preprocessed data are loaded, feature filtering, step-wise feature
  selection and leave-one-out cross-validation are conducted.
"""

import pickle
import copy
import warnings

from utils_exp import (get_experiment_src_data, filter_features, cv_single_feat_type)


warnings.filterwarnings("ignore")

PENALIZER = 0.3

for cancer in ["brca", "lung"]:
  for task in ["os", "dfs"]:

    # Load preprocessed data.
    src_data = get_experiment_src_data(cancer, task)
    with open("data/result/"+cancer+"_"+task+"/src_data.pkl", "wb") as f:
      pickle.dump(src_data, f)

    # Note here for ICGC WGS samples, we use LOOCV, but in WES, we use k-fold CV (k=10).
    k = len(src_data["tumors"])

    # Step 1: fitering features
    print("Filtering features...")
    list_clinical = filter_features(
        k, tumors=src_data["tumors"], survival=src_data["survival"],
        data=src_data["data"][src_data["list_clinical"]], penalizer=PENALIZER)
    list_driver = filter_features(
        k, tumors=src_data["tumors"], survival=src_data["survival"],
        data=src_data["data"][src_data["list_driver"]], penalizer=PENALIZER)
    list_twonode = filter_features(
        k, tumors=src_data["tumors"], survival=src_data["survival"],
        data=src_data["data"][src_data["list_twonode"]], penalizer=PENALIZER)
    list_multinode = filter_features(
        k, tumors=src_data["tumors"], survival=src_data["survival"],
        data=src_data["data"][src_data["list_multinode"]], penalizer=PENALIZER)

    # Step 2: step-wise feature selection
    # for single feature types
    print("Step-wise feature selection...")
    results_clinical = cv_single_feat_type(
        k=k, tumors=src_data["tumors"], survival=src_data["survival"],
        data=src_data["data"], feats=list_clinical, penalizer=PENALIZER)
    with open("data/result/"+cancer+"_"+task+"/clinical.pkl", "wb") as f:
      pickle.dump(results_clinical, f)

    results_driver = cv_single_feat_type(
        k=k, tumors=src_data["tumors"], survival=src_data["survival"],
        data=src_data["data"], feats=list_driver, penalizer=PENALIZER)
    with open("data/result/"+cancer+"_"+task+"/driver.pkl", "wb") as f:
      pickle.dump(results_driver, f)

    results_twonode = cv_single_feat_type(
        k=k, tumors=src_data["tumors"], survival=src_data["survival"],
        data=src_data["data"], feats=list_twonode, penalizer=PENALIZER)
    with open("data/result/"+cancer+"_"+task+"/twonode.pkl", "wb") as f:
      pickle.dump(results_twonode, f)

    results_multinode = cv_single_feat_type(
        k=k, tumors=src_data["tumors"], survival=src_data["survival"],
        data=src_data["data"], feats=list_multinode, penalizer=PENALIZER)
    with open("data/result/"+cancer+"_"+task+"/multinode.pkl", "wb") as f:
      pickle.dump(results_multinode, f)

    # for feature types in addition to clinical features
    survival_clinical = copy.deepcopy(src_data["survival"])
    for feat in results_clinical["feats"]:
      survival_clinical[feat] = src_data["data"][feat]

    results_clinical_driver = cv_single_feat_type(
        k=k, tumors=src_data["tumors"], survival=survival_clinical,
        data=src_data["data"], feats=list_driver, penalizer=PENALIZER)
    with open("data/result/"+cancer+"_"+task+"/clinical_driver.pkl", "wb") as f:
      pickle.dump(results_clinical_driver, f)

    results_clinical_twonode = cv_single_feat_type(
        k=k, tumors=src_data["tumors"], survival=survival_clinical,
        data=src_data["data"], feats=list_twonode, penalizer=PENALIZER)
    with open("data/result/"+cancer+"_"+task+"/clinical_twonode.pkl", "wb") as f:
      pickle.dump(results_clinical_twonode, f)

    results_clinical_multinode = cv_single_feat_type(
        k=k, tumors=src_data["tumors"], survival=survival_clinical,
        data=src_data["data"], feats=list_multinode, penalizer=PENALIZER)
    with open("data/result/"+cancer+"_"+task+"/clinical_multinode.pkl", "wb") as f:
      pickle.dump(results_clinical_multinode, f)

    results_clinical_genomic = cv_single_feat_type(
        k=k, tumors=src_data["tumors"], survival=survival_clinical,
        data=src_data["data"], feats=list_driver+list_twonode+list_multinode, penalizer=PENALIZER)
    with open("data/result/"+cancer+"_"+task+"/clinical_genomic.pkl", "wb") as f:
      pickle.dump(results_clinical_genomic, f)


