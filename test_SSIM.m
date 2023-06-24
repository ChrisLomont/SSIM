# Chris Lomont 2023
# Sample to test in Octave to see if SSIM code
# gives the same results as the original implementation
K = [0.01 0.03];
window = ones(8);
L = 256;
path = "C:\\ImageFilterTools\\datasets\\SSIM\\";

filepath1 = strcat(path,"einstein.gif");
img1 = imread(filepath1);

# files from https://www.cns.nyu.edu/~lcv/ssim/
filenames = {'einstein.gif','meanshift.gif','contrast.gif','impulse.gif','blur.gif','jpg.gif'};
truth = {1.0,0.988,0.913,0.840,0.694,0.662};
for index = 1:length(filenames)
  filename = filenames{index};
  truthMssim = truth{index};
  filepath2 = strcat(path,filename);
  img2 = imread(filepath2);
  [mssim,ssim_map] = ssim(img1,img2,K,window,L);
  mseVal = immse(img1,img2);
  fprintf('%10s MSE: %0.0f SSIM: %f (= truth %f?)\n',filename,mseVal,mssim,truthMssim);
end
