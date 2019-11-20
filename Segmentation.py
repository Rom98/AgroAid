import cv2 as cv
import numpy as np
img = cv.imread('URL')
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (50,50,200,200)
res = cv.resize(img,(256,256), interpolation = cv.INTER_CUBIC)
mask = np.zeros(res.shape[:2], np.uint8)
cv.grabCut(res,mask,rect,bgdModel,fgdModel,6,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

res = res*mask2[:,:,np.newaxis]
cv.imwrite('URL',res)