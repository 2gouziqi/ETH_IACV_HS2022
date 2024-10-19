# Note: You are not allowed to import additional python packages except NumPy
import numpy as np

K = 16

class ImageCompressor:
    # This class is responsible to i) learn the codebook given the training images
    # and ii) compress an input image using the learnt codebook.
    def __init__(self):
        # You can modify the init function to add / remove some fields
        self.mean_image = np.array([])
        self.principal_components = np.array([])
        
    def get_codebook(self):
        # This function should return all information needed for compression
        # as a single numpy array
        
        # TODO: Modify this according to you algorithm
        # self.mean_image [1,27648]
        # self.principal_components [27648, K]
        mean_image_re = np.array(np.reshape(self.mean_image, (1,27648)))
        principal_components_re = np.array(self.principal_components.T) 
        codebook = np.vstack((mean_image_re, principal_components_re)) # (K+1,27648)
        return codebook
    
    def train(self, train_images):
        # Given a list of training images as input, this function should learn the 
        # codebook which will then be used for compression
        
        # ******************************* TODO: Implement this ***********************************************#
        x_m = np.reshape(train_images, (100,27648))
        self.mean_image = x_m.mean(0) # (27648,)
        x = self.mean_image - x_m
        x = x.T
        U, s, Vh = np.linalg.svd(x, full_matrices=False)          
        U = U[:, :K] # (27648*K)
        U = np.array(U)    
        self.principal_components = np.float16(U)
        self.mean_image = np.float16(self.mean_image)
        return

    def compress(self, test_image):
        # Given a test image, this function should return the compressed representation of the image
        # ******************************* TODO: Implement this ***********************************************#
        test_image = np.reshape(test_image, (1,-1)) #(1,27648)
        dm = test_image - self.mean_image     
        test_image_compressed = np.matmul(self.principal_components.T,dm.T)
        return np.float16(test_image_compressed) 


class ImageReconstructor:
    # This class is used on the client side to reconstruct the compressed images.
    def __init__(self, codebook):
        # The codebook learnt by the ImageCompressor is passed as input when
        # initializing the ImageReconstructor
        
        self.mean_image = codebook[0] # (27648,)
        self.mean_image = np.reshape(self.mean_image,(27648,1))
        self.principal_components = codebook[1:K+1]


    def reconstruct(self, test_image_compressed):
        # Given a compressed test image, this function should reconstruct the original image
        # ******************************* TODO: Implement this ***********************************************#
        test_image_recon = np.matmul(self.principal_components.T,test_image_compressed)
        test_image_recon = test_image_recon + self.mean_image
        test_image_recon = test_image_recon.astype(int) # (d,n)
        rec = np.reshape(test_image_recon, (96,96,3))

        for i in range(96):
          for j in range(96):
            pixel = rec[i,j]
            if pixel[0]<150 and pixel[1]<127.5 and pixel[2]<127.5:
              pixel = [0,0,0]
            elif pixel[0]>150 and pixel[1]<127.5 and pixel[2]<127.5:
              pixel = [255,0,0]
            elif pixel[0]<150 and pixel[1]>127.5 and pixel[2]<127.5:
              pixel = [0,255,0]
            else:
              pixel = [255,255,255]
              rec[i,j] = pixel
        return rec
