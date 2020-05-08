from sklearn.feature_extraction import image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import savez_compressed, savez

p = 12

# one_image = Image.open("LC08_L1TP_184029_20160616_20170324_01_T1_B4.tif")
one_image = Image.open("LC08_L1TP_184029_20170806_20170813_01_T1_B4.TIF")

#one_image = one_image.resize((2800,2800), Image.NEAREST)
one_image = np.array(one_image)
# A = one_image[1500:6000, 1500:6000]
A = one_image[2200:2500, 3350:3650]
#A=one_image
# A = (A-np.amin(A))/(np.amax(A)-np.amin(A))

plt.gray()
plt.imshow(A)
plt.show()

[L1, L2] = np.shape(A)
l1 = int(L1//p)
l2 = int(L2//p)


# for i in range(l1):
#     for j in range(l2):
#         patch = A[i*p:(i+1)*p, j*p:(j+1)*p]
#         patch = (patch - patch.min()) / (patch.max() - patch.min())
#         np.save('./patches_inainte_alunecare_12x12/patch'+str(i)+'_'+str(j)+'.npy', patch)

# B = np.load('./patches/patch0_0.npy')

'''
print(L1, L2)
print(l1,l2)

L1 = 100
L2 = 100
'''

for i in range(int(p/2),L1-int(p/2),2):
    for j in range(int(p/2),L2-int(p/2),2):
        #print(i-p/2, i+p/2, j-p/2, j+p/2)
        patch = A[int(i-p/2):int(i+p/2), int(j-p/2):int(j+p/2)]
        patch = (patch - patch.min()) / (patch.max() - patch.min())
        np.save('./patches_img300x300_dupa_12x12/patch'+str(i)+'_'+str(j)+'.npy', patch)


# print('Image shape: {}'.format(one_image.shape))
# print('Image type: {}'.format(type(one_image)))

# patches = image.extract_patches_2d(one_image, (28, 28))

# print('Patches shape: {}'.format(patches.shape))
# print('Patches len: {}'.format(len(patches)))

#print(patches[0])
#print(patches[800])
'''
for i in range(len(patches)):
    data = asarray(patches[i])
    savez_compressed('patch'+str(i)+'.npz', data)
'''
