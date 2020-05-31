################ZSL_GAN configs################################
device = "cpu"
gpu = '0' #index of GPU to use
splitmode = 'easy' #the way to split train/test data: easy/hard
manualSeed = 11 # manual seed
resume  = None #the model to resume
disp_interval = 20
save_interval = 200
evl_interval = 40
""" hyper-parameter for training """
GP_LAMBDA = 10  # Gradient penalty lambda
CENT_LAMBDA = 1
REG_W_LAMBDA = 0.001
REG_Wz_LAMBDA = 0.0001
lr = 0.0001
batchsize = 48
""" hyper-parameter for testing"""
nSample = 60  # number of fake feature for each class
Knn = 20  # knn: the value of K


rdc_text_dim = 1000
z_dim = 100
h_dim = 4096

VGG_FEATURES_SIZE = 14
IMG_SIZE = 16
INCEPTION_V3_OUTPUT_SIZE = 17


###############AttenGAN configs#################################
cfg_file = "cfg/DAMSM/bird.yml"
data_dir = ""
UPDATE_INTERVAL = 200
checkpoint_model_ZSL = ""





