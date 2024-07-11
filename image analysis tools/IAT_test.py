from imageAnalysisTools import *

# Initial image
image = cv2.imread("animals/leopard.jpg")
cv2.imwrite('rendered/0_initial.jpg', image)

# Gray conversion
image_gray = grayscale_converter(image)
cv2.imwrite('rendered/1_gray.jpg', image_gray)

# Gaussian blur
image_blur = gaussian_blur(image, 10, 3)
cv2.imwrite('rendered/2_blur.jpg', image_blur)

# Canny
image_gray_blur = gaussian_blur(image_gray)
cv2.imwrite('rendered/Canny_0_blur_grey.jpg', image_gray_blur)

grad_x, grad_y, magnitudes, angles = xy_gradients(image_gray_blur)
cv2.imwrite('rendered/Canny_1_grad_x.jpg', grad_x)
cv2.imwrite('rendered/Canny_1_grad_y.jpg', grad_y)
cv2.imwrite('rendered/Canny_2_magnitudes.jpg', magnitudes)

outlined_matrix = only_maxima(magnitudes, angles)
cv2.imwrite('rendered/Canny_3_maxima.jpg', outlined_matrix)

hysteresis_matrix = hysteresis(outlined_matrix)
cv2.imwrite('rendered/Canny_4_hystersis_FINAL_CANNY.jpg', hysteresis_matrix)

# Contours
cv2.imwrite('rendered/countours.jpg', contour(image))