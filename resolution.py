from cv2 import dnn_superres 
import cv2

#create SR object
sr = dnn_superres.DnnSuperResImpl_create()

#read image
image = cv2.imread("output/2D_image_grayscale.jpg")

#read model
path = "models/EDSR_x2.pb"
sr.readModel(path)

#set desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 2)

#upscale the image 
result = sr.upsample(image)


padding = (0,0,0)
new_shape = (450,450)
original_shape = (result.shape[1], result.shape[0])
ratio = float(max(new_shape)) / max(original_shape)
new_size = tuple([int(x*ratio) for x in original_shape])
image = cv2.resize(result, new_size)
delta_w = new_shape[0] - new_size[0]
delta_h = new_shape[1] - new_size[1]
top, bottom = delta_h//2, delta_h-(delta_h//2)
left, right = delta_w//2, delta_w-(delta_w//2) 

image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding)

#create SR object
sr = dnn_superres.DnnSuperResImpl_create()

#read model
path = "models/EDSR_x2.pb"
sr.readModel(path)

#set desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 2)

#upscale the image 
result = sr.upsample(image)

cv2.imwrite("output/2D_Image_grayscale_upscaled.jpg", result)
