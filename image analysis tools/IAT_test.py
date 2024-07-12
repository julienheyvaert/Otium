from imageAnalysisTools import *
import time

# Initial image
image = cv2.imread("animals/zebre.jpg")
cv2.imwrite('rendered/0_initial.jpg', image)

def test(image = image):
    # Canny
    start = time.time()
    canny_matrix = canny(image)
    cv2.imwrite('rendered/1_Canny.jpg', canny_matrix)
    end = time.time()
    print(f"-- Canny computed in {end - start} seconds.")

    # Extrimities
    start = time.time()
    extrimities = edges_extrimity_finder(canny_matrix)
    cv2.imwrite('rendered/2_Extrimities.jpg', extrimities)
    end = time.time()
    print(f"-- Extrimities computed in {end - start} seconds.")


def test_all():
    # Gray conversion
    start = time.time()
    image_gray = grayscale_converter(image)
    cv2.imwrite('rendered/1_gray.jpg', image_gray)
    end = time.time()
    print(f"-- Graysale conversion in {end - start} seconds.")

    # Black and White conversion
    start = time.time()
    image_b_w = black_white_converter(image)
    cv2.imwrite('rendered/3_blackAndWhite.jpg', image_b_w)
    end = time.time()
    print(f"-- Black and White conversion in {end - start} seconds.")

    # Gaussian blur
    start = time.time()
    image_blur = gaussian_blur(image, 10, 3)
    cv2.imwrite('rendered/4_blur.jpg', image_blur)
    end = time.time()
    print(f"-- Blured in {end - start} seconds.")

    # Canny
    start_canny = time.time()

    image_gray_blur = gaussian_blur(image_gray)
    cv2.imwrite('rendered/Canny_0_blur_grey.jpg', image_gray_blur)

    start = time.time()
    grad_x, grad_y, magnitudes, angles = xy_gradients(image_gray_blur)
    cv2.imwrite('rendered/Canny_1_grad_x.jpg', grad_x)
    cv2.imwrite('rendered/Canny_1_grad_y.jpg', grad_y)
    cv2.imwrite('rendered/Canny_2_magnitudes.jpg', magnitudes)
    end = time.time()
    print(f"-- gradients, magnitudes, angles computed in {end - start} seconds.")

    start = time.time()
    outlined_matrix = only_maxima(magnitudes, angles)
    cv2.imwrite('rendered/Canny_3_maxima.jpg', outlined_matrix)
    end = time.time()
    print(f"-- Outlined in {end - start} seconds.")

    start = time.time()
    hysteresis_matrix = hysteresis(outlined_matrix)
    cv2.imwrite('rendered/Canny_4_hystersis_FINAL_CANNY.jpg', hysteresis_matrix)
    end = time.time()
    print(f"-- Hysteresis computed in {end - start} seconds.")

    end_canny = time.time()
    print(f"-- Canny algorithm computed in {end_canny - start_canny} seconds.")

    # Extrimity finder
    start = time.time()
    extrimities = edges_extrimity_finder(hysteresis_matrix)
    cv2.imwrite('rendered/Canny_5_Extrimities.jpg', extrimities)
    end = time.time()
    print(f"-- Extrimities computed in {end - start} seconds.")

    # Contours
    start = time.time()
    canny = contour(image)
    cv2.imwrite('rendered/Z_10_countours.jpg', canny)
    end = time.time()
    print(f"-- Contours drawed in {end - start} seconds.")

test_all()