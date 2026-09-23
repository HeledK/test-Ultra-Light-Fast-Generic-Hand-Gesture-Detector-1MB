import cv2, numpy as np

img = cv2.imread("00c601ff-0b34-4500-8710-8d367ceabcc7.jpg")  # BGR

# Version A: feed as BGR
img_bgr = cv2.resize(img, (320, 240))
img_bgr = (img_bgr.astype(np.float32) - 127) / 128

# Version B: feed as RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb = cv2.resize(img_rgb, (320, 240))
img_rgb = (img_rgb.astype(np.float32) - 127) / 128

# Run both through your .pth model, compare confidences