from pathlib import Path

# Available datasets
CELEBA = '/nfs/students/summer-term-2018/project_2/data/CelebA/'
CELEBA_REDUCED = '/nfs/students/summer-term-2018/project_2/data/CelebA_reduced/'  # DEBUG
CAR = '/nfs/students/summer-term-2018/project_2/data/YT_CAR_DRIVING/'
# DeepFakes
MEGA_MERKEL_TRUMP = "/nfs/students/summer-term-2018/project_2/data/MEGA_Merkel_Trump"
SIMONE_MERKEL = "/nfs/students/summer-term-2018/project_2/data/Simone_Merkel"


# ==================================================
# =====
# ===== Set here the directory you want to use
# =====
# ==================================================
ROOT = Path(CELEBA)

# Preprocessing
CONVERTER_BASE = "/nfs/students/summer-term-2018/project_2/data/converter"
CONVERTER_INPUT = "/test_converter_images"
CONVERTER_OUTPUT = "/converter_output"
IMAGE_DOWNLOADER = "/nfs/students/summer-term-2018/project_2/data/ImageDownloader"
VIDEO_DOWNLOADER = "/nfs/students/summer-term-2018/project_2/data/VideoDownloader"

# ===== Settings for preprocessing
RESOLUTIONS = [2, 4, 8, 16, 32, 64, 128]
LOW_RESOLUTIONS = [2, 4, 8]

# Trainer
MOST_RECENT_MODEL = "."

# ===== Names
# Folder names
RAW = "raw"
A = "A"
B = "B"
PREPROCESSED = "preprocessed"

# Buffer names
LANDMARKS_BUFFER = "landmarks.txt"
HISTO_BUFFER = "histo.txt"

# File names
IMAGES_2_NPY = 'images2.npy'
IMAGES_4_NPY = 'images4.npy'
IMAGES_8_NPY = 'images8.npy'
IMAGES_16_NPY = 'images16.npy'
IMAGES_32_NPY = 'images32.npy'
IMAGES_64_NPY = 'images64.npy'
IMAGES_128_NPY = 'images128.npy'

LANDMARKS_NPY = "landmarks.npy"
LANDMARKS_5_NPY = "landmarks5.npy"
LANDMARKS_10_NPY = "landmarks10.npy"
LANDMARKS_28_NPY = "landmarks28.npy"
LANDMARKS_MEAN_NPY = "landmarks_mean.npy"
LANDMARKS_MEAN_5_NPY = "landmarks5_mean.npy"
LANDMARKS_MEAN_10_NPY = "landmarks10_mean.npy"
LANDMARKS_MEAN_28_NPY = "landmarks28_mean.npy"
LANDMARKS_COV_NPY = "landmarks_cov.npy"
LANDMARKS_COV_5_NPY = "landmarks5_cov.npy"
LANDMARKS_COV_10_NPY = "landmarks10_cov.npy"
LANDMARKS_COV_28_NPY = "landmarks28_cov.npy"

LOWRES_2_NPY = 'lowres2.npy'
LOWRES_4_NPY = 'lowres4.npy'
LOWRES_8_NPY = 'lowres8.npy'
LOWRES_MEAN_2_NPY = 'lowres2_mean.npy'
LOWRES_MEAN_4_NPY = 'lowres4_mean.npy'
LOWRES_MEAN_8_NPY = 'lowres8_mean.npy'
LOWRES_COV_2_NPY = 'lowres2_cov.npy'
LOWRES_COV_4_NPY = 'lowres4_cov.npy'
LOWRES_COV_8_NPY = 'lowres8_cov.npy'

# ===== Paths
# To folders
RAW_FOLDER = ROOT / RAW
PREPROCESSED_FOLDER = ROOT / PREPROCESSED
# To buffers
LANDMARKS_BUFFER_PATH = ROOT / LANDMARKS_BUFFER
# To arrays
ARRAY_IMAGES_2 = ROOT / IMAGES_2_NPY
ARRAY_IMAGES_4 = ROOT / IMAGES_4_NPY
ARRAY_IMAGES_8 = ROOT / IMAGES_8_NPY
ARRAY_IMAGES_16 = ROOT / IMAGES_16_NPY
ARRAY_IMAGES_32 = ROOT / IMAGES_32_NPY
ARRAY_IMAGES_64 = ROOT / IMAGES_64_NPY
ARRAY_IMAGES_128 = ROOT / IMAGES_128_NPY
ARRAY_LANDMARKS = ROOT / LANDMARKS_NPY
ARRAY_LANDMARKS_5 = ROOT / LANDMARKS_5_NPY
ARRAY_LANDMARKS_10 = ROOT / LANDMARKS_10_NPY
ARRAY_LANDMARKS_28 = ROOT / LANDMARKS_28_NPY
ARRAY_LANDMARKS_MEAN = ROOT / LANDMARKS_MEAN_NPY
ARRAY_LANDMARKS_5_MEAN = ROOT / LANDMARKS_MEAN_5_NPY
ARRAY_LANDMARKS_10_MEAN = ROOT / LANDMARKS_MEAN_10_NPY
ARRAY_LANDMARKS_28_MEAN = ROOT / LANDMARKS_MEAN_28_NPY
ARRAY_LANDMARKS_COV = ROOT / LANDMARKS_COV_NPY
ARRAY_LANDMARKS_5_COV = ROOT / LANDMARKS_COV_5_NPY
ARRAY_LANDMARKS_10_COV = ROOT / LANDMARKS_COV_10_NPY
ARRAY_LANDMARKS_28_COV = ROOT / LANDMARKS_COV_28_NPY
ARRAY_LOWRES_2 = ROOT / LOWRES_2_NPY
ARRAY_LOWRES_4 = ROOT / LOWRES_4_NPY
ARRAY_LOWRES_8 = ROOT / LOWRES_8_NPY
ARRAY_LOWRES_2_MEAN = ROOT / LOWRES_MEAN_2_NPY
ARRAY_LOWRES_4_MEAN = ROOT / LOWRES_MEAN_4_NPY
ARRAY_LOWRES_8_MEAN = ROOT / LOWRES_MEAN_8_NPY
ARRAY_LOWRES_2_COV = ROOT / LOWRES_COV_2_NPY
ARRAY_LOWRES_4_COV = ROOT / LOWRES_COV_4_NPY
ARRAY_LOWRES_8_COV = ROOT / LOWRES_COV_8_NPY
