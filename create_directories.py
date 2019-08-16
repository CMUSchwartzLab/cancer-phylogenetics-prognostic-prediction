import os


try:
  os.mkdir("data/TCGA/")
  os.mkdir("data/TCGA/BRCA/")
  os.mkdir("data/TCGA/LUAD/")
  os.mkdir("data/TCGA/LUSC/")
  os.mkdir("data/vcf-data/")
  os.mkdir("data/tusv/")
  os.mkdir("data/canopy/")
  os.mkdir("data/survival/")
  os.mkdir("data/clinical/")
  os.mkdir("data/driver/")
  os.mkdir("data/twonode/")
  os.mkdir("data/multinode/")
  os.mkdir("data/result/")
  os.mkdir("data/result/brca_os/")
  os.mkdir("data/result/brca_dfs/")
  os.mkdir("data/result/lung_os/")
  os.mkdir("data/result/lung_dfs/")
except OSError:
  print ("Creation of the directory failed")


