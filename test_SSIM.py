# code to compare some SSIM implementations against original SSIM author results at https://www.cns.nyu.edu/~lcv/ssim/
# Chris Lomont 2023
# SSIM testing
from skimage.metrics import structural_similarity, mean_squared_error
from skimage.metrics import mean_squared_error
# import tensorflow as tf # crashed...
import torch
import torchmetrics
import numpy as np
from PIL import Image,ImageOps
from SSIM_PIL import compare_ssim
import IQA_pytorch
import cv2 as cv


def err_IQA(img1,img2):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img1 =torch.tensor(img1, device=device).float()
    img2 =torch.tensor(img2, device=device).float()

    mean_squared_error = torchmetrics.MeanSquaredError()
    mse_score = mean_squared_error(img1, img2)

    img1 /= 255.0
    img2 /= 255.0

    def expand_img(img):
        img = img.unsqueeze(0) # add dimension
        img = img.repeat(3,1,1)
        img = img.unsqueeze(0) # add dimension
        return img
    
    img1 = expand_img(img1)
    img2 = expand_img(img2)

    D = IQA_pytorch.SSIM()
    ssim_score = D(img1, img2, as_loss=False)

    return (mse_score.item(),ssim_score.item())

def err_scikit(img1, img2):
    # see https://stackoverflow.com/questions/58604326/which-ssim-is-correct-skimage-metrics-structural-similarity
    ssim_score = structural_similarity(
        img1, img2, 
        multichannel=False,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        data_range=255.0)
    mse_score = mean_squared_error(img1,img2)
    return (mse_score,ssim_score)

def err_PIL(image1,image2):
    ssim_score = compare_ssim(image1, image2, GPU=False)
    mse_score = 0.0
    return (mse_score,ssim_score)

# [get-mssim]
# from https://docs.opencv.org/3.4/d5/dc4/tutorial_video_input_psnr_ssim.html
def getMSSISM(i1, i2):
    C1 = 6.5025
    C2 = 58.5225
    # INITS
    I1 = np.float32(i1) # cannot calculate on one byte large values
    I2 = np.float32(i2)
    I2_2 = I2 * I2 # I2^2
    I1_2 = I1 * I1 # I1^2
    I1_I2 = I1 * I2 # I1 * I2
    # END INITS
    # PRELIMINARY COMPUTING
    mu1 = cv.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
    sigma2_2 = cv.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
    sigma12 = cv.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2 # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2 # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    ssim_map = cv.divide(t3, t1) # ssim_map = t3./t1;
    mssim = cv.mean(ssim_map) # mssim = average of ssim map
    return mssim

def err_opencv(image1,image2):
    ssim_score = getMSSISM(image1, image2)[0]
    mse_score = 0.0
    return (mse_score,ssim_score)


# images and test values from https://www.cns.nyu.edu/~lcv/ssim/
def process(fn1, fn2, mse_truth, ssim_truth):

    image1 = Image.open(fn1)
    image2 = Image.open(fn2)

    # applying grayscale method if color images
    if image1.mode == 'RGB':
        # print('ty',image1)
        image1 = ImageOps.grayscale(image1) # todo - match other code?
        image2 = ImageOps.grayscale(image2) # todo - match other code?

    np_image1 = np.array(image1) # opencv cannot read GIF!
    np_image2 = np.array(image2)

    print('MSE {:.0f} SSIM {:.3f} '.format(round(mse_truth,1),round(ssim_truth,3)),end='')

    def write1(name,errs):
        mse,ssim = errs
        print('{} MSE {:.0f} SSIM {:.3f} '.format(name,round(mse,1),round(ssim,3)),end='')
        return ssim

    ssim_pil     = write1('PIL',err_PIL(image1,image2))
    ssim_scikit  = write1('Scikit',err_scikit(np_image1,np_image2))
    ssim_pytorch = write1('Pytorch IQA',err_IQA(np_image1,np_image2))
    ssim_opencv  = write1('OpenCV ',err_opencv(np_image1,np_image2))

    print()

    return np.array([ssim_pil,ssim_scikit,ssim_pytorch,ssim_opencv])


def process_SSIM_test():
    # test images at 
    ssimPath = 'C:/ImageFilterTools/datasets/SSIM/'
    files = ["einstein","meanshift","contrast","impulse","blur","jpg"]
    ssim_truth = [1.0, 0.988, 0.913, 0.840, 0.694, 0.662]
    mse_truth = [0.0,144.0,144.0,144.0,144.0,144.0,142.0]

    
    print('Each is MSE and SSIM truth, then values from some libs')
    for i in range(len( files)):
        f = files[i]
        fn1 = ssimPath + "einstein.gif"
        fn2 = ssimPath + f + ".gif"
        print(f"{f}: ",end='')
        process(fn1,fn2, mse_truth[i],ssim_truth[i])
    

def process_LIVE():
    # LIVE dataset
    livePath = 'C:/ImageFilterTools/datasets/LIVE/'

    with open(livePath+"allcompares.txt") as pairs_file:
        i = 0
        ssim_max_err = np.zeros(4)
        ssim_sum_err = np.zeros(4)
        for line in pairs_file:
            f1,f2,sc=line.split()
            ssim_truth = float(sc)
            fn1 = livePath + f1
            fn2 = livePath + f2
            print(i+1,'/',982,':',end='')
            ssimN = process(fn1, fn2, 0.0, ssim_truth)
            ssimErr = np.abs(ssimN - ssim_truth)
            ssim_max_err = np.maximum(ssim_max_err, ssimErr)
            ssim_sum_err = ssim_sum_err + ssimErr
            i = i + 1        
        ssim_avg_err = ssim_sum_err / i
        print(' ssim_pil, ssim_scikit, ssim_pytorch, ssim_opencv')
        print('SSIM max err',ssim_max_err)
        print('SSIM avg err',ssim_avg_err)


process_SSIM_test()    
process_LIVE()