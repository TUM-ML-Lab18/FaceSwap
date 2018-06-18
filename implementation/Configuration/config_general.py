# Preprocessing
CONVERTER_BASE = "/nfs/students/summer-term-2018/project_2/data/converter"
CONVERTER_INPUT = "/test_converter_images"
CONVERTER_OUTPUT = "/converter_output"
IMAGE_DOWNLOADER = "/nfs/students/summer-term-2018/project_2/data/ImageDownloader"
VIDEO_DOWNLOADER = "/nfs/students/summer-term-2018/project_2/data/VideoDownloader"

RAW = "raw"
A = "A"
B = "B"
PREPROCESSED = "preprocessed"
LANDMARKS_JSON = "landmarks.json"
LANDMARKS_BUFFER = "landmarks.txt"
LANDMARKS_NPY = "landmarks.npy"
HISTO_BUFFER = "histo.txt"
HISTO_JSON = "histo.json"
HISTO_NPY = "histo.npy"
FACE_EMBEDDINGS_BUFFER = "embeddings.txt"
FACE_EMBEDDINGS_JSON = "embeddings.json"
FACE_EMBEDDINGS_NPY = "embeddings.npy"
ANNOTATIONS = "list_attr_celeba.txt"
RESOLUTIONS = [4, 8, 16, 32, 64, 128]

# Trainer
MOST_RECENT_MODEL = "."

# ===================================== MODELS
# DeepFakes
MEGA_MERKEL_TRUMP = "/nfs/students/summer-term-2018/project_2/data/MEGA_Merkel_Trump"
SIMONE_MERKEL = "/nfs/students/summer-term-2018/project_2/data/Simone_Merkel"

# CelebA
# ROOT_CELEBA = '/nfs/students/summer-term-2018/project_2/data/CelebA/'
ROOT_CELEBA = '/nfs/students/summer-term-2018/project_2/data/CelebA_reduced/'  # DEBUG
ARRAY_CELEBA_IMAGES_2 = ROOT_CELEBA + 'data2.npy'
ARRAY_CELEBA_IMAGES_4 = ROOT_CELEBA + 'data4.npy'
ARRAY_CELEBA_IMAGES_8 = ROOT_CELEBA + 'data8.npy'
ARRAY_CELEBA_IMAGES_16 = ROOT_CELEBA + 'data16.npy'
ARRAY_CELEBA_IMAGES_32 = ROOT_CELEBA + 'data32.npy'
ARRAY_CELEBA_IMAGES_64 = ROOT_CELEBA + 'data64.npy'
ARRAY_CELEBA_IMAGES_128 = ROOT_CELEBA + 'data128.npy'
ARRAY_CELEBA_LANDMARKS = ROOT_CELEBA + 'landmarks.npy'
ARRAY_CELEBA_LANDMARKS_MEAN = ROOT_CELEBA + 'landmarks_mean.npy'
ARRAY_CELEBA_LANDMARKS_COV = ROOT_CELEBA + 'landmarks_cov.npy'
ARRAY_CELEBA_LANDMARKS_5 = ROOT_CELEBA + 'landmarks5.npy'
ARRAY_CELEBA_LANDMARKS_5_MEAN = ROOT_CELEBA + 'landmarks5_mean.npy'
ARRAY_CELEBA_LANDMARKS_5_COV = ROOT_CELEBA + 'landmarks5_cov.npy'
ARRAY_CELEBA_LANDMARKS_10 = ROOT_CELEBA + 'landmarks10.npy'
ARRAY_CELEBA_LANDMARKS_10_MEAN = ROOT_CELEBA + 'landmarks10_mean.npy'
ARRAY_CELEBA_LANDMARKS_10_COV = ROOT_CELEBA + 'landmarks10_cov.npy'
ARRAY_CELEBA_LANDMARKS_28 = ROOT_CELEBA + 'landmarks28.npy'
ARRAY_CELEBA_LANDMARKS_28_MEAN = ROOT_CELEBA + 'landmarks28_mean.npy'
ARRAY_CELEBA_LANDMARKS_28_COV = ROOT_CELEBA + 'landmarks28_cov.npy'
ARRAY_CELEBA_HISTO = ROOT_CELEBA + 'histo.npy'
ARRAY_CELEBA_HISTO_8 = ROOT_CELEBA + 'histo8.npy'
ARRAY_CELEBA_HISTO_16 = ROOT_CELEBA + 'histo16.npy'
ARRAY_CELEBA_HISTO_32 = ROOT_CELEBA + 'histo32.npy'
ARRAY_CELEBA_HISTO_64 = ROOT_CELEBA + 'histo64.npy'
ARRAY_CELEBA_ATTRIBUTES = ROOT_CELEBA + 'attributes.npy'
ARRAY_CELEBA_LOWRES = ROOT_CELEBA + 'lowres.npy'
ARRAY_CELEBA_ULTRA_LOWRES = ROOT_CELEBA + 'ultra_lowres.npy'
ARRAY_CELEBA_ULTRA_LOWRES_MEAN = ROOT_CELEBA + 'ultra_lowres_mean.npy'
ARRAY_CELEBA_ULTRA_LOWRES_COV = ROOT_CELEBA + 'ultra_lowres_cov.npy'

# YouTube
ROOT_CAR = '/nfs/students/summer-term-2018/project_2/data/YT_CAR_DRIVING/'
ARRAY_CAR_LOWRES = ROOT_CAR + 'lowres.npy'
ARRAY_CAR_IMAGES_8 = ROOT_CAR + 'data8.npy'
ARRAY_CAR_IMAGES_64 = ROOT_CAR + 'data64.npy'
ARRAY_CAR_IMAGES_128 = ROOT_CAR + 'data128.npy'
ARRAY_CAR_LANDMARKS = ROOT_CAR + 'landmarks.npy'
