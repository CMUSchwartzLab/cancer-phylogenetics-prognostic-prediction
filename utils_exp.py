""" Utilities for experiment.py.
"""

from collections import defaultdict as dd
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter


def get_survival(cancer, task):
  """ Get the survival data of specific cancer type and prediction task.

  Parameters
  ----------
  cancer: str
    cancer type, can be `brca` or `lung`
  task: str
    prognostic prediction task, can be `os` or `dfs`

  Returns
  -------
  data: dataframe
    each row is a sample, two columns are status and months respectively

  """

  if cancer == "brca":
    df = pd.read_csv("data/survival/brca.csv", index_col=0)
  elif cancer == "lung":
    df_luad = pd.read_csv("data/survival/luad.csv", index_col=0)
    df_lusc = pd.read_csv("data/survival/lusc.csv", index_col=0)
    df = pd.concat([df_luad,df_lusc])

  data = pd.DataFrame(
      data=np.zeros((df.shape[0],3)),
      index=df.index,
      columns=["tumor","status","months"]
      )
  data["tumor"] = df["tumor"]
  data["status"] = df[task+"_status"]
  data["months"] = df[task+"_months"]
  data.index = data["tumor"]
  data.drop(columns=["tumor"],inplace=True)

  return data


def get_clinical(cancer, quiet_mode=True):
  """ Get the clinical data of specific cancer type.
  Continuous features will be normalized in this function (categorical feature will be kept).
  The features that are too sparse will be dropped.

  Parameters
  ----------
  cancer: str
    cancer type, can be `brca` or `lung`
  quiet_mode: bool
    if True, will print verbose features that are dropped.

  Returns
  -------
  data: dataframe
    each row is a sample, each column is a specific clinical feature

  """

  data = pd.read_csv("data/clinical/"+cancer+".csv", index_col=0)

  if cancer == "brca":
    for feat in [
        "gender",
        "history_of_neoadjuvant_treatment",
        "cytokeratin_immunohistochemistry_staining_method_micrometastasis_indicator"]:
      data[feat] = 1 - data[feat]

    for feat in data.columns:
      if np.std(data[feat]) <= 1e-2:
        data.drop(columns=[feat], inplace=True)
      elif np.sum(data[feat]) <= int(0.05*np.shape(data)[0]):
        data.drop(columns=[feat], inplace=True)
        if not quiet_mode:
          print(feat)
    for feat in [
        "age_at_initial_pathologic_diagnosis",
        "her2_immunohistochemistry_level_result",
        "number_of_lymphnodes_positive_by_he"]:
      data[feat] = StandardScaler().fit_transform(data[feat].values.reshape(-1, 1))

  elif cancer == "lung":
    for feat in [
        "history_of_neoadjuvant_treatment|no"]:
      data[feat] = 1 - data[feat]
    for feat in data.columns:
      if np.std(data[feat]) <= 1e-2:
        data.drop(columns=[feat], inplace=True)
      elif np.sum(data[feat]) <= int(0.05*np.shape(data)[0]):
        data.drop(columns=[feat], inplace=True)
        if not quiet_mode:
          print(feat)
    for feat in [
        "age_at_initial_pathologic_diagnosis"]:
      data[feat] = StandardScaler().fit_transform(data[feat].values.reshape(-1, 1))

  return data


def get_driver(cancer, quiet_mode=True):
  """ Get the driver data of specific cancer type.
  The features that are too sparse will be dropped.

  Parameters
  ----------
  cancer: str
    cancer type, can be `brca` or `lung`
  quiet_mode: bool
    if True, will print verbose features that are dropped.

  Returns
  -------
  data: dataframe
    each row is a sample, each column is a specific driver feature

  """

  data = pd.read_csv("data/driver/"+cancer+".csv", index_col=0)
  for feat in data.columns:
    if np.std(data[feat]) <= 1e-2:
      data.drop(columns=[feat], inplace=True)
      if not quiet_mode:
        print("all zeros",feat)
    elif np.sum(data[feat] != 0) <= int(0.01*np.shape(data)[0]):
      data.drop(columns=[feat], inplace=True)
      if not quiet_mode:
        print("many zeros",feat)

  return data


def get_twonode(cancer):
  """ Get the two-node data of specific cancer type.
  All the features will be normalized in this function.

  Parameters
  ----------
  cancer: str
    cancer type, can be `brca` or `lung`

  Returns
  -------
  data: dataframe
    each row is a sample, each column is a specific two-node feature

  """

  data = pd.read_csv("data/twonode/"+cancer+".csv", index_col=0)

  for feat in data.columns:
    data[feat] = StandardScaler().fit_transform(data[feat].values.reshape(-1, 1))

  return data


def get_multinode(cancer, quiet_mode=True):
  """ Get the multi-node data of specific cancer type.
  All the features will be normalized in this function.
  The features that are too sparse will be dropped.

  Parameters
  ----------
  cancer: str
    cancer type, can be `brca` or `lung`
  quiet_mode: bool
    if True, will print verbose features that are dropped.

  Returns
  -------
  data: dataframe
    each row is a sample, each column is a specific multi-node feature

  """

  data = pd.read_csv("data/multinode/"+cancer+".csv", index_col=0)

  for feat in data.columns:
    if np.std(data[feat]) <= 1e-2:
      data.drop(columns=[feat], inplace=True)
      if not quiet_mode:
        print("all zeros",feat)
    else:
      data[feat] = StandardScaler().fit_transform(data[feat].values.reshape(-1, 1))

  return data


def align_meta_data(survival, clinical, driver, twonode, multinode, quiet_mode=True):
  """ Align multiple data and features together and shuffle the samples.

  Parameters
  ----------
  survival: dataframe
    survival data, each row a sample, two columns are event status and time
  clinical: dataframe
    clinical data, each row a sample, each column a clinical feature
  driver: dataframe
    driver data, each row a sample, each column a driver feature
  twonode: dataframe
    two-node data, each row a sample, each column a two-node feature
  multinode: dataframe
    multi-node data, each row a sample, each column a multi-node feature
  quiet_mode: bool
    if True, will print verbose features that are dropped.

  Returns
  -------
  src_data: dict
    `src_data` include the aligned data and list of features, specifically, it
    contains 7 keys:

    tumors: list of str
      aligned list of tumor barcodes
    survival: dataframe
      aligned survival data, each row a sample, two columns are event status and time
    data: dataframe
      aligned features, each row is a sample, each column is a specific feature
    list_clinical: list of str
      list of clinical features in the data
    list_driver: list of str
      list of driver features in the data
    list_twonode: list of str
      list of two-node features in the data
    list_multinode: list of str
      list of multi-node features in the data

  """

  # Find the union of samples
  tumors = [s for s in survival.index \
            if (s in clinical.index) and \
            (s in driver.index) and \
            (s in twonode.index) and \
            (s in multinode.index)]
  # Remove samples that have unknow survival time (no decease time nor follow-up time)
  tumors = [s for s in tumors if survival.loc[s]["status"] != -1]

  #np.random.seed(0)
  np.random.shuffle(tumors)

  survival = survival.loc[tumors]
  clinical = clinical.loc[tumors]
  driver = driver.loc[tumors]
  twonode = twonode.loc[tumors]
  multinode = multinode.loc[tumors]

  data = pd.DataFrame(index=tumors,columns=[],dtype=float)

  list_clinical, list_driver, list_twonode, list_multinode = [], [], [], []
  NUM_NON_DUP = 3
  # To filter the features that have too small variance.
  # At least 3 samples should have different features from the mode.
  for feat in clinical.columns:
    if np.sum(clinical[feat] != clinical[feat].mode().values[0]) >= NUM_NON_DUP:
      data[feat] = clinical[feat]# + 1e-3*(np.random.rand(np.shape(clinical[feat])[0])-0.5)
      list_clinical.append(feat)
    else:
      if not quiet_mode:
        print(feat)

  for feat in driver.columns:
    if np.sum(driver[feat] != driver[feat].mode().values[0]) >= NUM_NON_DUP:
      data[feat] = driver[feat]# + 1e-3*(np.random.rand(np.shape(driver[feat])[0])-0.5)
      list_driver.append(feat)
    else:
      if not quiet_mode:
        print(feat)

  for feat in twonode.columns:
    if np.sum(twonode[feat] != twonode[feat].mode().values[0]) >= NUM_NON_DUP:
      data[feat] = twonode[feat]# + 1e-3*(np.random.rand(np.shape(twonode[feat])[0])-0.5)
      list_twonode.append(feat)
    else:
      if not quiet_mode:
        print(feat)

  for feat in multinode.columns:
    if np.sum(multinode[feat] != multinode[feat].mode().values[0]) >= NUM_NON_DUP:
      data[feat] = multinode[feat]# + 1e-3*(np.random.rand(np.shape(multinode[feat])[0])-0.5)
      list_multinode.append(feat)
    else:
      if not quiet_mode:
        print(feat)

  src_data = {"tumors":tumors, "survival":survival, "data":data,
              "list_clinical":list_clinical, "list_driver":list_driver,
              "list_twonode":list_twonode, "list_multinode":list_multinode}

  return src_data


def get_experiment_src_data(cancer, task):
  """ Prepare the source data for the experiment. Specifically,
  extract the survival, clinical, driver, two-node and multi-node data,
  drop the features that are too sparse and normalize the conintuous features,
  align these different data types and shuffle the order of samples.

  Parameters
  ----------
  cancer: str
    cancer type, can be `brca` or `lung`
  task: str
    prognostic prediction task, can be `os` or `dfs`

  Returns
  -------
  src_data: dict
    `src_data` include the aligned data and list of features, specifically, it
    contains 7 keys:

    tumors: list of str
      aligned list of tumor barcodes
    survival: dataframe
      aligned survival data, each row a sample, two columns are event status and time
    data: dataframe
      aligned features, each row is a sample, each column is a specific feature
    list_clinical: list of str
      list of clinical features in the data
    list_driver: list of str
      list of driver features in the data
    list_twonode: list of str
      list of two-node features in the data
    list_multinode: list of str
      list of multi-node features in the data

  """

  survival = get_survival(cancer, task)
  clinical = get_clinical(cancer)
  driver = get_driver(cancer)
  twonode = get_twonode(cancer)
  multinode = get_multinode(cancer)

  src_data = align_meta_data(
      survival, clinical, driver, twonode, multinode)

  return src_data


def get_assignments(n, k):
  """ assign the test set for k-fold cross-validation

  Parameters
  ----------
  n: int
    total number of samples for cross-validation
  k: int
    number of fold for cross-validation

  Returns
  -------
  assignments: list of int
    [1, 2, 3, ..., k, 1, 2, 3, ..., k, ...] it have the length of `n`

  """

  assignments = np.array((n // k + 1) * list(range(1, k + 1)))
  assignments = assignments[:n]

  return assignments


def cv_add_single_feat(k, tumors, survival, data, feat, penalizer):
  """ k-fold cross-validation by using an additional feature specified by `feat`.

  Parameters
  ----------
  k: int
    number of fold for cross-validation
  tumors: list of str
    the list of sample names (TCGA barcode in our case)
  survival: dataframe
    survival data, each row a sample, the columns include event status, time of last follow-up
    Note: the existing features are also included in this dataframe (survival.columns rather than data.columns)
  data: dataframe
    all the candidate features of samples, each row a sample, each column a candidate feature
  feat: str
    a feature that is to be added to the exiting features in `survival` for CV
    feat have to be included in the data.cloumns
  penalizer: float
    l2 penalizer coefficient for CV

  Returns
  -------
  ci: float
    concordance index of cross-validation
  T_concat: list of float
    ground truth last follow-up time
  E_concat: list of 0/1
    list of event states
  T_pred_concat: list of float
    predicted survival time (negative hazard)

  """

  # https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/utils/__init__.py#L548
  survival = survival.copy()
  T_concat = []
  E_concat = []
  T_pred_concat = []

  assignments = get_assignments(len(tumors), k)

  cph = CoxPHFitter(penalizer=penalizer)

  # add feat to the existing features
  survival[feat] = data[feat]

  feats_columns = survival.columns.drop(["status", "months"])

  for i in range(1, k + 1):
    ix = (assignments == i)
    train_data = survival.loc[~ix]
    test_data = survival.loc[ix]

    # fit the fitter to the training data
    if np.sum(train_data["status"].values) == 0:
      print("error!")
    else:
      cph.fit(train_data, duration_col="months", event_col="status",
              show_progress=False, step_size=0.1) #0.1

      test_X = test_data[feats_columns]
      test_T = test_data["months"].values
      test_E = test_data["status"].values

      T_pred = -cph.predict_partial_hazard(test_X).values

      T_concat.append(test_T)
      E_concat.append(test_E)
      T_pred_concat.append(T_pred)

  T_concat = np.concatenate(T_concat)
  E_concat = np.concatenate(E_concat)
  T_pred_concat = np.concatenate(T_pred_concat)

  if np.sum(E_concat) == 0:
    print("error")
  else:
    ci = concordance_index(T_concat, T_pred_concat, E_concat)

  return ci, T_concat, E_concat, T_pred_concat


def remove_single_redundant_feature(feats, data, threshold, quiet_mode):
  """ Filtering out the a single redundant feature that are highly correlated.

  Parameters
  ----------
  feats: list of str
    candidate features to be removed, have to be in ascending order of relevance
  data: dataframe
    each row a sample, each column a feature
  threshold: float
    if two features have correlation magnitude larger than `threshold`, the one
    with lower relevance will be removed
  quiet_mode: bool
    if True, intermediate results will not be printed out

  Returns
  -------
  first element: bool
    if no feature is remove, return False, else True
  second element: list of str
    a list of the unremoved feats

  """

  for idx in range(len(feats)):
    for idy in range(idx+1, len(feats)):
      corr = np.corrcoef(data[[feats[idx],feats[idy]]].values.T)[0,1]
      if np.abs(corr) >= threshold:
        if not quiet_mode:
          print(corr, feats[idx], feats[idy])
        del feats[idx]
        return True, feats

  return False, feats


def filter_features(k, tumors, survival, data, penalizer, quiet_mode = True):
  """ Filter out features using univariate Cox regression based on
  max-relevance rule and min-redundancy rule

  Parameters
  ----------
  k: int
    number of fold for cross-validation
  tumors: list of str
    the list of sample names (TCGA barcode in our case)
  survival: dataframe
    survival data, each row a sample, the columns include event status, time of last follow-up
  data: dataframe
    all the candidate features of samples, each row a sample, each column a candidate feature
  penalizer: float
    l2 penalizer coefficient for CV
  quiet_mode: bool
    if True, intermediate results will not be printed out

  Returns
  -------
  feats: list of str
    the list of features that pass both max-relevance rule and min-redundancy rule
    top features are chosen and are sorted in the descending order of relevance

  """

  # Step 1: implement the max-relevance rule to get the most informative single features.
  feat2ci = dd(float)
  list_feat_ci = []
  for feat in data.columns:
    ci, _, _, _  = cv_add_single_feat(
        k=k, tumors=tumors, survival=survival,
        data=data, feat=feat, penalizer=penalizer)

    list_feat_ci.append([feat, ci])
    feat2ci[feat] = ci
    if not quiet_mode:
      print(feat, ci)

  # sort in the ascending order of max-relevance
  list_feat_ci = sorted(list_feat_ci, key=lambda pair:pair[1], reverse=False)
  feats = [t[0] for t in list_feat_ci]

  # Step 2: implement the min-redundancy rule to remove the redundant features
  while True:
    flag, feats = remove_single_redundant_feature(feats, data, threshold = 0.8, quiet_mode=quiet_mode)
    if flag == False:
      break

  feats = feats[::-1]

  # if the feature has univariate regression ci > 0.5, it will be kept
  # otherwise, the top five features will be kept
  for idx, feat in enumerate(feats):
    if feat2ci[feat] > 0.5:
      continue
    elif idx < 4:
      continue
    else:
      break

  feats = feats[:idx+1]

  return feats


def cv_stepwise_selection(k, tumors, survival, data, feats, penalizer, quiet_mode = True):
  """ Step-wise feature selection cross-validation

  Parameters
  ----------
  k: int
    number of fold for cross-validation
  tumors: list of str
    the list of sample names (TCGA barcode in our case)
  survival: dataframe
    survival data, each row a sample, the columns include event status, and time of last follow-up
  data: dataframe
    all the data, each row a sample, each column a feature
  feats: list of str
    all the candidate features of samples
  penalizer: float
    l2 penalizer coefficient for CV
  quiet_mode: bool
    if True, intermediate results will not be printed out

  Returns
  -------
  ci_best: float
    best concordance index of cross-validation
  feats_best: list of str
    list of features selected
  T_best: list of float
    ground truth last follow-up time
  E_best: list of 0/1
    list of event states
  T_pred_best: list of float
    best predicted survival time (negative hazard)

  """

  ci_best = -1.0
  feats_best = []
  T_best, E_best, T_pred_best = [], [], []

  while True:
    if len(feats) == 0:
      break

    cis = []
    T_s, E_s, T_preds = [], [], []

    # add a feature at a time
    for feat in feats:
      ci, T, E, T_pred = cv_add_single_feat(
          k=k, tumors=tumors, survival=survival,
          data=data, feat=feat, penalizer=penalizer)
      cis.append(ci)
      T_s.append(T)
      E_s.append(E)
      T_preds.append(T_pred)

    # get the feature that has largest CI
    idx_del = np.argmax(cis)

    ci = cis[idx_del]
    feat = feats[idx_del]

    if ci > ci_best:
      survival[feat] = data[feat]
      feats_best.append(feat)
      ci_best = ci
      T_best = T_s[idx_del]
      E_best = E_s[idx_del]
      T_pred_best = T_preds[idx_del]
      if not quiet_mode:
        print(feat + ": %.3f"%ci )
      del cis[idx_del]
      del feats[idx_del]
    else:
      break

  return ci_best, feats_best, T_best, E_best, T_pred_best


def cv_single_feat_type(k, tumors, survival, data, feats, penalizer, quiet_mode = True):
  """ Step-wise feature selection cross-validation of specific feature type

  Parameters
  ----------
  k: int
    number of fold for cross-validation
  tumors: list of str
    the list of sample names (TCGA barcode in our case)
  survival: dataframe
    survival data, each row a sample, the columns include event status, and time of last follow-up
    it can all include features that are already selected, e.g., in the case when addition genomic
    features are added to selected clinical features
  data: dataframe
    all the data, each row a sample, each column a feature
  feats: list of str
    all the candidate features of samples
  penalizer: float
    l2 penalizer coefficient for CV
  quiet_mode: bool
    if True, intermediate results will not be printed out

  Returns
  -------
  results: dict
    ci: float
      concordance index of cross-validation
    feats: list of str
      list of features selected
    T: list of float
      ground truth last follow-up time
    E: list of 0/1
      list of event states
    T_pred: list of float
      predicted survival time (negative hazard)
    all_feats: list of str
      all the candidate features

  """

  all_feats = copy.deepcopy(feats)
  survival = copy.deepcopy(survival)
  ci_best, feats_best, T, E, T_pred = cv_stepwise_selection(
      k=k, tumors=tumors, survival=survival, data=data,
      feats=feats, penalizer=penalizer, quiet_mode=quiet_mode)
  results = {"ci":ci_best, "feats":feats_best, "T":T, "E":E, "T_pred":T_pred, "all_feats":all_feats}

  return results


