""" Utilities for prepare_feature.py.
"""

import os
import math
from collections import defaultdict as dd
import numpy as np
import pandas as pd
import vcf


# Part 1: OS/DFS data
class Survival_Data_Processor:
  """ Prepare data related to survival analysis:
    * Overall survival (OS)
    * Disease free survival (DFS) / Recurrence free survival (RFS)

  """

  def __init__(self, path_home, cancer_type):
    """ Initialize the data processor with cancer type and proper constant.

    """

    self.path_home = path_home
    self.cancer_type = cancer_type
    self.path = self.path_home + "data/TCGA/"+self.cancer_type+"/clin.merged.txt"
    # 1 year = 365.2425 days
    # 1 month = 30.4369 days
    self.RATIO = 30.4369
    self.df = None

  def get_df(self, path):
    """ Read the TCGA clinical data and set it to self.df.

    """

    df = pd.read_csv(path, sep="\t")
    df.set_index(["admin.batch_number"], inplace = True)
    df.columns = df.loc["patient.bcr_patient_barcode"]
    self.df = df

  def check_num(self, status, months):
    """ To check the statistic of OS/DFS in the dataset.

    """

    print("#observed = %d; #unobserved = %d; #no-records = %d; #total = %d;sum_months = %.1f" % (
          len([1 for s in status if s == 1]),
          len([1 for s in status if s == 0]),
          len([1 for s in status if s == -1]),
          len(status),
          sum(months)))

  def get_os_brca_lung(self, df, RATIO, tumors):
    """ Extract the overall survival (OS) data from the data `df`.

    Parameters
    ----------
    df: dataframe
      clinical data from TCGA.
    RATIO: float
      number of days in a month.
    tumors: list of str
      barcode of tumor samples.

    Returns
    -------
    status: list of 0/1/-1
      whether a tumor sample deceased (1), lost follow-up (0), or data not available (-1)
    months: list of float
      last follow-up time (in unit of month).

    """

    status = list() # 1 for dead, 0 for alive, -1 if not records.
    months = list() # last followup time, -1 if not records.
    flups = [
        "follow_ups.follow_up-4.",
        "follow_ups.follow_up-3.",
        "follow_ups.follow_up-2.",
        "follow_ups.follow_up.",
        ""]

    for tumor in tumors:
      cur_status = -1
      cur_month = -1
      for flup in flups:
        days_death = float(df.loc["patient."+flup+"days_to_death",tumor])
        days_follow = float(df.loc["patient."+flup+"days_to_last_followup",tumor])
        if not math.isnan(days_death):
          cur_status = 1
          cur_month = days_death/RATIO
          break
        elif not math.isnan(days_follow):
          cur_status = 0
          cur_month = days_follow/RATIO
          break
        else: # both records missing
          continue
      if cur_month < 0:
        cur_status = -1
        cur_month = -1
      status.append(cur_status)
      months.append(cur_month)
    #self.check_num(status, months)
    return status, months

  def get_dfs_lung(self, df, RATIO, tumors):
    """ Extract the disease-free survival (DFS) data from the data `df` for lung cancer.

    Parameters
    ----------
    df: dataframe
      clinical data from TCGA.
    RATIO: float
      number of days in a month.
    tumors: list of str
      barcode of tumor samples.

    Returns
    -------
    status: list of 0/1/-1
      whether a tumor sample deceased (1), lost follow-up (0), or data not available (-1)
    months: list of float
      last follow-up time (in unit of month).

    """

    status = list() # 1 for dead, 0 for alive, -1 if not records.
    months = list() # last followup time, -1 if not records.

    flups = [
        "new_tumor_events.",
        "follow_ups.follow_up.",
        "follow_ups.follow_up-2.",
        "follow_ups.follow_up-3.",
        "follow_ups.follow_up-4."
        ]

    for tumor in tumors:
      cur_status = -1
      cur_month = -1
      for flup in flups:
        yon = df.loc[
            "patient."+flup+"new_tumor_event_after_initial_treatment",
            tumor]

        flupd = flup
        flupf = flup
        if flup == "new_tumor_events.":
          flupd = flup+"new_tumor_event."
          flupf = ""

        days_death = float(df.loc[
            "patient."+flupd+"days_to_new_tumor_event_after_initial_treatment",
            tumor])
        days_follow = float(df.loc["patient."+flupf+"days_to_last_followup",tumor])
        days_real_death = float(df.loc["patient."+flupf+"days_to_death",tumor])

        if (yon == "yes") and (not math.isnan(days_death)):
          cur_status = 1
          cur_month = days_death/RATIO
          break
        elif (yon == "yes") and (math.isnan(days_death)):
          # Won't treat such weired samples with broken records.
          break
        elif (yon == "no") and (not math.isnan(days_death)):
          # This case doesn't exist in LUAD.
          pass
        elif (yon == "no") and (math.isnan(days_death)):
          if not math.isnan(days_follow):
            cur_status = 0
            cur_month = days_follow/RATIO
          elif not math.isnan(days_real_death):
            cur_status = 0
            cur_month = days_real_death/RATIO
          else:
            # won't treat these cases
            pass
          continue
        else:#(yon == nan) and (math.isnan(days_death))
          # No records at this follow up.
          continue

      if cur_month < 0:
        cur_status = -1
        cur_month = -1

      status.append(cur_status)
      months.append(cur_month)
    #self.check_num(status, months)
    return status, months

  def get_dfs_brca(self, df, RATIO, tumors):
    """ Extract the disease-free survival (DFS) data from the data `df` for breast cancer.

    Parameters
    ----------
    df: dataframe
      clinical data from TCGA.
    RATIO: float
      number of days in a month.
    tumors: list of str
      barcode of tumor samples.

    Returns
    -------
    status: list of 0/1/-1
      whether a tumor sample deceased (1), lost follow-up (0), or data not available (-1)
    months: list of float
      last follow-up time (in unit of month).

    """

    status = list() # 1 for dead, 0 for alive, -1 if not records.
    months = list() # last followup time, -1 if not records.

    flups = [
        "patient.days_to_last_followup",
        "patient.follow_ups.follow_up.days_to_last_followup",
        "patient.follow_ups.follow_up-2.days_to_last_followup",
        "patient.follow_ups.follow_up-3.days_to_last_followup",
        "patient.follow_ups.follow_up-4.days_to_last_followup"
        ]

    recs = [
        [["patient.new_tumor_events.new_tumor_event_after_initial_treatment",
        "patient.new_tumor_events.new_tumor_event.days_to_new_tumor_event_after_initial_treatment",
        "patient.new_tumor_events.new_tumor_event-2.days_to_new_tumor_event_after_initial_treatment",
        "patient.new_tumor_events.new_tumor_event-3.days_to_new_tumor_event_after_initial_treatment",
        "patient.new_tumor_events.new_tumor_event-4.days_to_new_tumor_event_after_initial_treatment",
        "patient.new_tumor_events.new_tumor_event-5.days_to_new_tumor_event_after_initial_treatment"]],
        [["patient.follow_ups.follow_up.new_tumor_event_after_initial_treatment",
        "patient.follow_ups.follow_up.days_to_new_tumor_event_after_initial_treatment"],
        ["patient.follow_ups.follow_up.new_tumor_events.new_tumor_event_after_initial_treatment",
        "patient.follow_ups.follow_up.new_tumor_events.new_tumor_event.days_to_new_tumor_event_after_initial_treatment",
        "patient.follow_ups.follow_up.new_tumor_events.new_tumor_event-2.days_to_new_tumor_event_after_initial_treatment",
        "patient.follow_ups.follow_up.new_tumor_events.new_tumor_event-3.days_to_new_tumor_event_after_initial_treatment",
        "patient.follow_ups.follow_up.new_tumor_events.new_tumor_event-4.days_to_new_tumor_event_after_initial_treatment"]],
        [["patient.follow_ups.follow_up-2.new_tumor_event_after_initial_treatment",
        "patient.follow_ups.follow_up-2.days_to_new_tumor_event_after_initial_treatment"],
        ["patient.follow_ups.follow_up-2.new_tumor_events.new_tumor_event_after_initial_treatment",
        "patient.follow_ups.follow_up-2.new_tumor_events.new_tumor_event.days_to_new_tumor_event_after_initial_treatment",
        "patient.follow_ups.follow_up-2.new_tumor_events.new_tumor_event-2.days_to_new_tumor_event_after_initial_treatment",
        "patient.follow_ups.follow_up-2.new_tumor_events.new_tumor_event-3.days_to_new_tumor_event_after_initial_treatment"]],
        [["patient.follow_ups.follow_up-3.new_tumor_events.new_tumor_event_after_initial_treatment",
        "patient.follow_ups.follow_up-3.new_tumor_events.new_tumor_event.days_to_new_tumor_event_after_initial_treatment",
        "patient.follow_ups.follow_up-3.new_tumor_events.new_tumor_event-2.days_to_new_tumor_event_after_initial_treatment",
        "patient.follow_ups.follow_up-3.new_tumor_events.new_tumor_event-3.days_to_new_tumor_event_after_initial_treatment",
        "patient.follow_ups.follow_up-3.new_tumor_events.new_tumor_event-4.days_to_new_tumor_event_after_initial_treatment"]],
        [["patient.follow_ups.follow_up-4.new_tumor_events.new_tumor_event_after_initial_treatment",
        "patient.follow_ups.follow_up-4.new_tumor_events.new_tumor_event.days_to_new_tumor_event_after_initial_treatment"]]
        ]

    flups_death = [
        "patient.days_to_death",
        "patient.follow_ups.follow_up.days_to_death",
        "patient.follow_ups.follow_up-2.days_to_death",
        "patient.follow_ups.follow_up-3.days_to_death",
        "patient.follow_ups.follow_up-4.days_to_death"
        ]

    for tumor in tumors:
      cur_status = -1
      cur_month = -1
      flag_complete = False
      for idx_flup, flup in enumerate(flups):
        # consider the idx_flup-th followup
        days_follow = float(df.loc[flup, tumor])

        for rec in recs[idx_flup]:
          # rec is a list of [new_tumor_event, days_to_new_tumor_event s]
          yon = df.loc[rec[0],tumor]

          for idx_days in range(1,len(rec)):
            days_death = float(df.loc[rec[idx_days], tumor])
            if (yon == "yes") and (not math.isnan(days_death)):
              cur_status = 1
              cur_month = days_death/RATIO
              flag_complete = True
              break
            elif (yon == "yes") and (math.isnan(days_death)):
              # We ignore such minor cases
              flag_complete = True
              break
            elif (yon == "no") and (not math.isnan(days_death)):
              # Such event doesn't exist
              flag_complete = True
              break
            elif (yon == "no") and (math.isnan(days_death)):
              days_real_death = float(df.loc[flups_death[idx_flup], tumor])
              if not math.isnan(days_follow):
                cur_status = 0
                cur_month = days_follow/RATIO
                continue
              elif not math.isnan(days_real_death):
                cur_status = 0
                cur_month = days_real_death/RATIO
                flag_complete = True
                break
              else:
                # won't treat these cases
                continue
            else:
              # if yon == nan. No records at this follow up.
              continue
          if flag_complete == True:
            break
        if flag_complete == True:
          break

      if cur_month < 0:
        cur_status = -1
        cur_month = -1
      status.append(cur_status)
      months.append(cur_month)

    #self.check_num(status, months)
    return status, months

  def run(self):
    """ The main function to extract OS/DFS time (in unit of months) of smaples.

    Returns
    -------
    refined: pandas matrix
      each row means a sample, with index of TCGA barcode
      four columns represent the status and time of OS/DFS
    """

    self.get_df(self.path)
    tumors = list(self.df.columns)

    #print(self.cancer_type+" OS:")
    os_status, os_months = self.get_os_brca_lung(self.df, self.RATIO, tumors)

    #print(self.cancer_type+" DFS:")
    if self.cancer_type == "BRCA":
      dfs_status, dfs_months = self.get_dfs_brca(self.df, self.RATIO, tumors)
    elif (self.cancer_type == "LUAD") or (self.cancer_type == "LUSC"):
      dfs_status, dfs_months = self.get_dfs_lung(self.df, self.RATIO, tumors)

    refined = pd.DataFrame(
        data=np.zeros((len(tumors), 5),"float"),
        columns=["tumor","os_status", "os_months","dfs_status","dfs_months"]
        )
    refined["tumor"] = [t.upper() for t in tumors]
    refined["os_status"] = os_status
    refined["os_months"] = os_months
    refined["dfs_status"] = dfs_status
    refined["dfs_months"] = dfs_months

    return refined



# Part 2: Clinical data
def get_df_clinical(path):
  """ Read the TCGA clinical data.

  """

  df = pd.read_csv(path, sep="\t")
  df.set_index(["admin.batch_number"], inplace = True)
  df.columns = [t.upper() for t in list(df.loc["patient.bcr_patient_barcode"])]
  return df

def get_clinical_feature(path_home, cancer_type):
  """ Extract the clinical features of specific cancer type.

  Parameters
  ----------
  path_home: str
    path to the `data/` directory
  cancer_type: str
    cancer type can be `BRCA` or `lung`

  Returns
  -------
  data: dataframe matrix
    each row a sample, each col a feature
    the categorical features are mapped into sparse one-hot features
  feature_lut: dict
    {categorical_feature:{subfeature:index, ...}, ...}
    the subfeature of categorical features and their order in the data matrix.

  """

  if cancer_type == "BRCA":

    features_continuous = [
        "patient.age_at_initial_pathologic_diagnosis",
        "patient.her2_neu_chromosone_17_signal_ratio_value",#<0.5
        "patient.her2_immunohistochemistry_level_result",
        "patient.number_of_lymphnodes_positive_by_he",
        "patient.her2_erbb_pos_finding_cell_percent_category",#<0.5
        "patient.fluorescence_in_situ_hybridization_diagnostic_procedure_chromosome_17_signal_result_range"]#<0.5
    features_bernoulli_str = [
        "patient.gender",
        "patient.history_of_neoadjuvant_treatment",
        "patient.ethnicity",
        "patient.person_neoplasm_cancer_status",
        "patient.cytokeratin_immunohistochemistry_staining_method_micrometastasis_indicator",
        "patient.lab_procedure_her2_neu_in_situ_hybrid_outcome_type"]#<0.5
    feature_multinomial_str = [
        "patient.race_list.race",
        "patient.menopause_status",
        "patient.margin_status",
        "patient.histological_type",
        "patient.breast_carcinoma_estrogen_receptor_status",
        "patient.breast_carcinoma_progesterone_receptor_status",
        "patient.lab_proc_her2_neu_immunohistochemistry_receptor_status",
        "patient.stage_event.pathologic_stage"]

  elif cancer_type == "lung":

      features_continuous = [
          "patient.age_at_initial_pathologic_diagnosis"]
      features_bernoulli_str = [
          "patient.gender",
          "patient.ethnicity",
          "patient.person_neoplasm_cancer_status",
          "type"]# cancer type
      feature_multinomial_str = [
          "patient.history_of_neoadjuvant_treatment",
          "patient.histological_type",
          "patient.race_list.race",
          "patient.stage_event.pathologic_stage",
          "patient.anatomic_neoplasm_subdivision"]

  features = features_continuous+features_bernoulli_str+feature_multinomial_str

  if cancer_type == "BRCA":
    df = get_df_clinical(path_home+"data/TCGA/BRCA/clin.merged.txt")
    df_survival = pd.read_csv(path_home+"data/survival/BRCA.csv", index_col=0)
    data = df.loc[features, list(df_survival["tumor"])]
    data = data.T

  elif cancer_type == "lung":
    df_luad = get_df_clinical(path_home+"data/TCGA/LUAD/clin.merged.txt")
    df_survival_luad = pd.read_csv(path_home+"data/survival/LUAD.csv", index_col=0)
    data_luad = df_luad.loc[features, list(df_survival_luad["tumor"])]
    data_luad = data_luad.T
    data_luad["type"] = ["LUAD"] * data_luad.shape[0]

    df_lusc = get_df_clinical(path_home+"data/TCGA/LUSC/clin.merged.txt")
    df_survival_lusc = pd.read_csv(path_home+"data/survival/LUSC.csv", index_col=0)
    data_lusc = df_lusc.loc[features, list(df_survival_lusc["tumor"])]
    data_lusc = data_lusc.T
    data_lusc["type"] = ["LUSC"] * data_lusc.shape[0]

    data = pd.concat([data_luad, data_lusc])

  n = data.notnull()
  data = data.loc[:, n.mean() >= .5]

  # redefine the features, after removing ineffective ones
  features_continuous = [f for f in features_continuous if f in data.columns]
  features_bernoulli_str = [f for f in features_bernoulli_str if f in data.columns]
  feature_multinomial_str = [f for f in feature_multinomial_str if f in data.columns]
  features_pre = features_continuous+features_bernoulli_str+feature_multinomial_str

  #features_continuous_pre = features_continuous
  features_continuous = [f[8:] if f.startswith("patient.") else f for f in features_continuous]
  features_bernoulli_str = [f[8:] if f.startswith("patient.") else f for f in features_bernoulli_str]
  feature_multinomial_str = [f[8:] if f.startswith("patient.") else f for f in feature_multinomial_str]
  feature_multinomial_str = [f[10:] if f.startswith("race_list.") \
                             else f[12:] if f.startswith("stage_event.") \
                             else f for f in feature_multinomial_str]
  features = features_continuous+features_bernoulli_str+feature_multinomial_str

  data = data.rename(columns={f_pre:f for f_pre,f in zip(features_pre,features)})

  # Mapping 1: map pseudo continuous feature into continuous.

  remap_table = {
      "0":0.5,
      "1+":1.5,
      "2+":2.5,
      "3+":3.5}
  if "her2_immunohistochemistry_level_result" in features:
      data["her2_immunohistochemistry_level_result"] = data["her2_immunohistochemistry_level_result"].apply(
          lambda f: np.NaN if pd.isnull(f) else remap_table[f] )

  remap_table = {
      "<10%":0.05,
      "10-19%":0.15,
      "20-29%":0.25,
      "30-39%":0.35,
      "40-49%":0.45,
      "50-59%":0.55,
      "60-69%":0.65,
      "70-79%":0.75,
      "80-89%":0.85,
      "90-99%":0.95}
  if "her2_erbb_pos_finding_cell_percent_category" in features:
      data["her2_erbb_pos_finding_cell_percent_category"] = data["her2_erbb_pos_finding_cell_percent_category"].apply(
          lambda f: np.NaN if pd.isnull(f) else remap_table[f] )

  # Mapping 2: map features into median/mode

  data[features_continuous] = data[features_continuous].convert_objects(
      convert_numeric=True)
  for f in features_continuous:
      data[f].fillna( data[f].median(), inplace=True )

  for f in (features_bernoulli_str+feature_multinomial_str):
      data[f].fillna( data[f].mode().values[0], inplace=True )

  # Mapping 3: map categorical features into 0/1

  feature_lut = {}

  for feature in features_bernoulli_str:
      f2idx = {f:idx for idx, f in enumerate(list(set(data[feature].values)))}
      feature_lut[feature] = f2idx

      data[feature] = data[feature].apply(
        lambda f: f2idx[f])

  for feature in feature_multinomial_str:
      f2idx = {f:idx for idx, f in enumerate(list(set(data[feature].values)))}
      feature_lut[feature] = f2idx

      for sub_f in f2idx.keys():
          data[feature+"|"+sub_f] = data[feature].apply(
              lambda f: 1 if f == sub_f else 0)

  data = data.drop(feature_multinomial_str, axis=1)

  return data, feature_lut



# Part 3: Driver data.
def get_gene2info(path_gene_marker):
  """ Get the information of driver genes, specifically, positions on the genome.

  Parameters
  ----------
  path_gene_marker: str
    path to the file that stores the gene information

  Returns
  -------
  gene2info: dict
    {gene:[chromosome_number, starting_position, end_position]}

  """

  gene2chr = dd(str)
  gene2start = dd(list)
  gene2end = dd(list)
  with open(path_gene_marker,"r") as f:
    next(f)
    for line in f:
      line = line.strip().split("\t")
      gene, chrom, start, end = str(line[4]), str(line[0]), int(line[1]), int(line[2])
      gene2chr[gene] = chrom
      gene2start[gene].append(start)
      gene2end[gene].append(end)

  # calculate the median of all the positions to reduce noise
  for gene, starts in gene2start.items():
    gene2start[gene] = int(np.median(starts))

  for gene, ends in gene2end.items():
    gene2end[gene] = int(np.median(ends))

  gene2info = {gene:[chrom, gene2start[gene], gene2end[gene]] for gene, chrom in gene2chr.items()}

  return gene2info


def get_samples_icgc(cancer):
  """ Return the barcode of ICGC samples.

  Parameters
  ----------
  cancer: str
    cancer type, can be `BRCA` or `lung`

  Returns
  -------
  samples: list of str
    each element is a TCGA barcode of the sample

  """

  samples = list({f.split(".")[0] for f in os.listdir("data/vcf-data") if f.startswith("TCGA-")})
  df = pd.read_csv("data/clinical/"+cancer+".csv", index_col=0)
  cancer_samples = list(df.index)
  samples = [s for s in samples if s in cancer_samples]#92#90

  return samples


def get_cancer_driver(cancer):
  """ Get a list of potential drivers for specific cancer type.

  Parameters
  ----------
  cancer: str
    cancer type, can be `BRCA` or `lung`

  Returns
  -------
  samples: list of str
    each element is a driver gene name

  """

  path_genes = "data/intogen/"+cancer+"_drivers.txt"

  genes = []
  with open(path_genes, "r") as f:
    for line in f:
      line = line.strip()
      genes.append(line)

  return genes


def get_single_driver_feature(sample, genes, gene2info, feat2idx):
  """ Get the driver features of specific sample.

  Parameters
  ----------
  sample: str
    barcode of specific tumor sample
  genes: list of str
    list of potential drivers
  gene2info: dict
    {driver:[chrom, start, end],...}
  feat2idx: dict
    {driver:index_of_driver}

  Returns
  -------
  feats: list of int
    each element is the count of SNVs/CNAs/SVs happened in a driver

  """

  feats = np.zeros(len(feat2idx), int)

  # Read in SVs.
  vcf_file = "data/vcf-data/"+sample+".sngr.sv.vcf.gz"
  reader = vcf.Reader(open(vcf_file,"r"))

  for idx, rec in enumerate(reader):
    for gene in genes:
      info = gene2info.get(gene, ["chrUNKNOWN",0,0])
      chrom, start, end = info[0][3:], info[1], info[2]
      if rec.CHROM == chrom:
        if (rec.POS < end) and (start < rec.POS):
          feats[feat2idx[gene]] += 1
          break

  # Read in CNVs.
  vcf_file = "data/vcf-data/"+sample+".sngr.cnv.vcf.gz"
  reader = vcf.Reader(open(vcf_file,"r"))

  for idx, rec in enumerate(reader):
    if rec.samples[1]["TCN"] == 2:
      continue
    for gene in genes:
      info = gene2info.get(gene, ["chrUNKNOWN",0,0])
      chrom, start, end = info[0][3:], info[1], info[2]

      if rec.CHROM == chrom:
        if (end < rec.POS) or (rec.INFO["END"] < start):
          continue
        else:
          feats[feat2idx[gene]] += 1

  # Read in SNVs.
  vcf_file = "data/vcf-data/"+sample+".sngr.snv.vcf.gz"
  reader = vcf.Reader(open(vcf_file,"r"))

  for idx, rec in enumerate(reader):
    for gene in genes:
      info = gene2info.get(gene, ["chrUNKNOWN",0,0])
      chrom, start, end = info[0][3:], info[1], info[2]
      if rec.CHROM == chrom:
        if (rec.POS < end) and (start < rec.POS):
          feats[feat2idx[gene]] += 1
          break

  return feats


def get_driver_feature(cancer):
  """ Get the driver features of specific cancer type.

  Parameters
  ----------
  cancer: str
    cancer type, can be `BRCA` or `lung`

  Returns
  -------
  data: dataframe matrix
    each row a sample, each col a driver feature, each element is the count that
    a driver have been perturbed by SNVs/CNAs/SVs.

  """

  # Note: not true for TCGA data (should use hg38 instead)
  path_gene_marker = "data/intogen/driver_hg19.txt"
  gene2info = get_gene2info(path_gene_marker)

  samples = get_samples_icgc(cancer)
  sample2idx = {sample:idx for idx, sample in enumerate(samples)}
  idx2sample = {idx:sample for sample, idx in sample2idx.items()}

  genes = get_cancer_driver(cancer)
  feat2idx = {gene:idx for idx, gene in enumerate(genes)}
  idx2feat = {idx:gene for gene, idx in feat2idx.items()}

  matrix = np.zeros((len(sample2idx), len(feat2idx)), int)

  for idx_s, sample in enumerate(samples):
    feats = get_single_driver_feature(sample, genes, gene2info, feat2idx)
    matrix[sample2idx[sample]] = feats
    # This step can be slow. The line below is for monitoring purpose.
    if idx_s % 10 == 0:
      print("%s, %s: %d/%d"%(cancer, sample, idx_s, len(samples)))

  data = pd.DataFrame(
      data=matrix,
      index=[idx2sample[idx] for idx in range(len(idx2sample))],
      columns=[idx2feat[idx] for idx in range(len(idx2feat))],
      dtype=int)

  return data



# Part 4: Two-node data.
def get_single_twonode_feature(sample, feat2idx):
  """ Get the two-node features of specific sample.

  Parameters
  ----------
  sample: str
    barcode of specific tumor sample
  feat2idx: dict
    {twonode_feature:index_of_twonode}

  Returns
  -------
  feats: list of int
    each element is the mutation rate of specific twonode feature

  """

  feats = np.zeros(len(feat2idx), int)

  vcf_file = "data/vcf-data/"+sample+".sngr.sv.vcf.gz"
  reader = vcf.Reader(open(vcf_file,"r"))

  for idx, rec in enumerate(reader):
    feats[feat2idx["sv_rate"]] += 1

  vcf_file = "data/vcf-data/"+sample+".sngr.cnv.vcf.gz"
  reader = vcf.Reader(open(vcf_file,"r"))
  CNV_SIZE_CUTOFF = 500000

  for idx, rec in enumerate(reader):
    feats[feat2idx["cnv_rate"]] += 1
    if (rec.INFO["END"] - rec.POS) > CNV_SIZE_CUTOFF:
      feats[feat2idx["cnv_lg_rate"]] += 1
    else:
      feats[feat2idx["cnv_sm_rate"]] += 1

    if rec.samples[1]["TCN"] >= 3:
      feats[feat2idx["cnv_amp_rate"]] += 1
    elif rec.samples[1]["TCN"] <= 1:
      feats[feat2idx["cnv_del_rate"]] += 1

  vcf_file = "data/vcf-data/"+sample+".sngr.snv.vcf.gz"
  reader = vcf.Reader(open(vcf_file,"r"))

  for idx, rec in enumerate(reader):
    feats[feat2idx["snv_rate"]] += 1
    r = rec.REF
    a = str(rec.ALT[0])
    nucleotides = ["A", "T", "C", "G"]
    if (r in nucleotides) and (a in nucleotides) and (r != a):
      feats[feat2idx[r + "->" + a]] += 1

  return feats

def get_twonode_feature(cancer):
  """ Get the two-node features of specific cancer type.

  Parameters
  ----------
  cancer: str
    cancer type, can be `BRCA` or `lung`

  Returns
  -------
  data: dataframe matrix
    each row a sample, each col a two-node feature

  """

  # Note: we show the WGS (ICGC) data here.
  # WES (TCGA) is similar but sv_rate should not be considered.
  samples = get_samples_icgc(cancer)
  sample2idx = {sample:idx for idx, sample in enumerate(samples)}
  idx2sample = {idx:sample for sample, idx in sample2idx.items()}

  feat2idx = {
      "snv_rate" : 0, "cnv_rate" : 1, "sv_rate" : 2,
      "cnv_sm_rate": 3, "cnv_lg_rate" : 4, "A->T" : 5,
      "A->C" : 6, "A->G" : 7, "T->A" : 8,
      "T->C" : 9, "T->G" : 10, "C->A" : 11,
      "C->T" : 12, "C->G" : 13, "G->A" : 14,
      "G->T" : 15, "G->C" : 16, "cnv_amp_rate": 17,
      "cnv_del_rate": 18}

  idx2feat = {idx:gene for gene, idx in feat2idx.items()}

  matrix = np.zeros((len(sample2idx), len(feat2idx)), int)

  for idx_s, sample in enumerate(samples):
    feats = get_single_twonode_feature(sample, feat2idx)
    matrix[sample2idx[sample]] = feats
    # This step can be slow. The line below is for monitoring purpose.
    if idx_s % 10 == 0:
      print("%s, %s: %d/%d"%(cancer, sample, idx_s, len(samples)))

  data = pd.DataFrame(
      data=matrix,
      index=[idx2sample[idx] for idx in range(len(idx2sample))],
      columns=[idx2feat[idx] for idx in range(len(idx2feat))],
      dtype=int)

  return data



# Part 5: Multi-node data.
def get_samples_icgc_tusv(cancer):
  """ Return the barcode of ICGC samples that have TUSV output.

  Parameters
  ----------
  cancer: str
    cancer type, can be `BRCA` or `lung`

  Returns
  -------
  samples: list of str
    each element is a TCGA barcode of the sample

  """

  samples = [f for f in os.listdir("data/tusv/") if f.startswith("TCGA-") \
             and ("U.tsv" in os.listdir("data/tusv/"+f))]
  df = pd.read_csv("data/clinical/"+cancer+".csv", index_col=0)
  cancer_samples = list(df.index)
  samples = [s for s in samples if s in cancer_samples]

  return samples


def get_tree(file):
  """ Get the cancer phylogeny from the `file`.

  Parameters
  ----------
  file: str
    path to the file that stores the phylogeny

  Returns
  -------
  root: TreeNode
    root of the phylogeny

  """

  txt = file.read()
  lines = txt.split("\n")
  lines.pop(0)
  lines.pop(len(lines)-1)
  lines = [ line.replace("\t", "") for line in lines ]
  lines = [ line.replace(" ", "") for line in lines ]
  edges = {} # key is tuple (parent, child). val is edge label
  for line in lines:
    if "->" in line:
      i, j = line.split("->")
      if "[" in j:
        j, edge_label = j.split("[")
        edge_label = "[" + edge_label
        edge_label = edge_label.split("\"")[1]
        edges[(i, j)] = edge_label

  nodes = {}
  for (i, j), label in edges.iteritems():
    if i not in nodes.keys():
      nodes[i] = TreeNode(i)
    if j not in nodes.keys():
      nodes[j] = TreeNode(j, data = label)
    else:
      nodes[j].data = label
    nodes[i].children.append(nodes[j])
    nodes[j].parent = nodes[i]

  # find root
  root = nodes[nodes.keys()[0]]
  while root.parent is not None:
    root = root.parent

  return root


class TreeNode:
  """ Class of the tree node in the phylogeny (a type of binary tree).

  """

  def __init__(self, label, data = ""):
    """
    Parameters
    ----------
    label: str
      the name of the node
    data: str
      the change of CNV/SV from its parent to the current node, in the format
      of a string: "SV/CNV".

    """

    self.label = label
    self.parent = None
    self.children = []
    self.data = data
    self.height = -1
    self.height_sv = -1
    self.height_cnv = -1

  def __str__(self):
    """ Example:
      treenode = TreeNoe("node")
      print(str(treenode))

    """

    return self.as_str()

  def as_str(self, level = 1):
    """ Print out the tree structure and data.

    """

    indent = "\t".join([ "" for i in xrange(0, level) ])
    s = indent + "name:\t" + self.label + "\n"
    s += indent + "edge:\t" + self.data + "\n"
    s += indent + "children:\n"
    for child in self.children:
      s += child.as_str(level + 1) + "\n"
    return s


def get_edge_lengths(T):
  """ Collect the edge lengths in both SV rates and CNA rates.
  The algorithm traverses the tree using BFS.

  Parameters
  ----------
  T: TreeNode
    root of the phylogeny

  Returns
  -------
  list_sv: list of float
    list of edge lengths in SV rates
  list_cnv: list of float
    list of edge lengths in CNA rates

  """

  list_sv, list_cnv = [], []
  # BFS
  traversed = [T]
  while True:
    cur = traversed.pop(0)
    if cur.data != "0/0" and cur.data != "":
      list_sv.append(float(cur.data.split("/")[0]))
      list_cnv.append(float(cur.data.split("/")[1]))

    for ch in cur.children:
      traversed.append(ch)
    if len(traversed) == 0:
      break

  return list_sv, list_cnv


def get_tree_heights(T):
  """ Get the tree height in the unit of both SV and CNA rates.

  Parameters
  ----------
  T: TreeNode
    root of the phylogeny

  Returns
  -------
  tree_height: float
    height of the phylogeny in topology (each edge has a constant edge length of 1)
  tree_height_sv: float
    height of the phylogeny in SV rates
  tree_height_cnv: float
    height of the phylogeny in CNA rates

  """

  tree_height = 0
  tree_height_sv = 0
  tree_height_cnv = 0
  traversed = [T]
  while True:
    cur = traversed.pop(0)
    if cur.parent == None:
      cur.height = 0
      cur.height_sv = 0
      cur.height_cnv = 0
    else:
      cur.height = cur.parent.height + 1
      cur.height_sv = cur.parent.height_sv + float(cur.data.split("/")[0])
      cur.height_cnv = cur.parent.height_cnv + float(cur.data.split("/")[1])
      tree_height = max(tree_height, cur.height)
      tree_height_sv = max(tree_height_sv, cur.height_sv)
      tree_height_cnv = max(tree_height_cnv, cur.height_cnv)

    for ch in cur.children:
      traversed.append(ch)
    if len(traversed) == 0:
      break

  return tree_height, tree_height_sv, tree_height_cnv


def get_largest_clone_height(U, T):
  """ Get the height of the largest clone in the unit of both SV and CNA rates.
  Implemented in BFS algorithm.

  Parameters
  ----------
  U: float matrix of size (1, k)
    matrix representing the fractions of corresponding subclones
  T: TreeNode
    root of the phylogeny

  Returns
  -------
  (float, float):
    (height of largest clone in unit of SV rates, height of largest clone in unit of CNA rates)

  """

  idx_max = str(np.argmax(U))

  traversed = [T]
  while True:
    cur = traversed.pop(0)
    if cur.label == idx_max:
      return cur.height_sv, cur.height_cnv

    for ch in cur.children:
      traversed.append(ch)
    if len(traversed) == 0:
      break

  return 0, 0


def get_tusv_features(T, U, amp_ratio=1.0):
  """ Get the multi-node features based on the phylogeny and fraction matrix.

  Parameters
  ----------
  T: TreeNode
    root of the phylogeny
  U: matrix of size (1,k)
    matrix representing the fractions of corresponding subclones
  amp_ratio: float
    used when subsampling is used when running TUSV

  Returns
  -------
  feats: list of float
    each element is a specific multi-node feature

  """

  lg_clone_portion = np.max(U)
  diversity = 1.0 - np.sum(U ** 2)
  num_clone = np.sum(U != 0)

  list_sv, list_cnv = get_edge_lengths(T)
  list_sv = [s*amp_ratio for s in list_sv]
  list_cnv = [s*amp_ratio for s in list_cnv]

  list_variants = [x+y for x, y in zip(list_sv, list_cnv)]

  height_topology, height_sv, height_cnv = get_tree_heights(T)
  height_sv = height_sv*amp_ratio
  height_cnv = height_cnv*amp_ratio

  height = height_sv+height_cnv

  lg_clone_sv, lg_clone_cnv = get_largest_clone_height(U, T)
  lg_clone_sv = lg_clone_sv*amp_ratio
  lg_clone_cnv = lg_clone_cnv*amp_ratio

  branch_mean_sv = np.mean(list_sv)
  branch_var_sv = np.std(list_sv)
  branch_max_sv = max(list_sv)
  branch_mean_cnv = np.mean(list_cnv)
  branch_var_cnv = np.std(list_cnv)
  branch_max_cnv = max(list_cnv)
  branch_mean = np.mean(list_variants)
  branch_var = np.std(list_variants)
  branch_max = max(list_variants)
  branch_num = len(list_variants)
  branch_len = np.sum(list_variants)

  feats = [num_clone, lg_clone_portion, diversity, height_topology, height_sv, height_cnv, \
    height, lg_clone_sv, lg_clone_cnv, branch_mean_sv, branch_var_sv, branch_max_sv, \
    branch_mean_cnv, branch_var_cnv, branch_max_cnv, \
    branch_mean, branch_var, branch_max, branch_num, branch_len]

  return feats


def get_single_multinode_feature(sample):
  """ Get the multi-node features of specific sample.

  Parameters
  ----------
  sample: str
    barcode of specific tumor sample

  Returns
  -------
  feats: list of float
    each element is a specific multi-node feature

  """

  U = np.genfromtxt("data/tusv/"+sample+"/U.tsv", dtype = float)
  T = get_tree(open("data/tusv/"+sample+"/T.dot"))

  feats = get_tusv_features(T, U)

  return feats


def get_multinode_feature(cancer):
  """ Get the multi-node features of specific cancer type.

  Parameters
  ----------
  cancer: str
    cancer type, can be `BRCA` or `lung`

  Returns
  -------
  data: dataframe matrix
    each row a sample, each col a multi-node feature

  """

  samples = get_samples_icgc_tusv(cancer)
  sample2idx = {sample:idx for idx, sample in enumerate(samples)}

  feat_list = [
      "num_clone", "lg_clone_portion", "diversity", "height_topology", "height_sv", "height_cnv", \
      "height", "lg_clone_sv", "lg_clone_cnv", "branch_mean_sv", "branch_var_sv", "branch_max_sv", \
      "branch_mean_cnv", "branch_var_cnv", "branch_max_cnv", \
      "branch_mean", "branch_var", "branch_max", "branch_num", "branch_len"]

  matrix = np.zeros((len(samples), len(feat_list)),float)

  for idx_s, sample in enumerate(samples):
    feats = get_single_multinode_feature(sample)
    matrix[sample2idx[sample]] = feats
    # This step can be slow. The line below is for monitoring purpose.
    #print(idx_s, len(samples), cancer, sample)

  data = pd.DataFrame(
      data = matrix,
      index=samples,
      columns=feat_list,
      dtype=float)

  return data


