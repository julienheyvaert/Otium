from imageAnalysisTools import *
import cv2
import time

def aggregate(edges_matrix, threshold, kernel_dim):
    """
    edges_matrix est une matrice comprenant les bordures détectées sur une image.
    Le but est de calculer le point moyen de bordure sur une zone de dimension kernel_dim
    et de placer un nouveau point de bordure moyen (en supprimant les anciens).
    """
    aggregate_matrix = np.copy(edges_matrix)
    border_locations = np.argwhere(edges_matrix == 255)
    kernel_radius = kernel_dim // 2

    i = 0
    for border in border_locations:
        i+=1
        row, col = border
        row_start = max(row - kernel_radius, 0)
        row_end = min(row + kernel_radius + 1, edges_matrix.shape[0])
        col_start = max(col - kernel_radius, 0)
        col_end = min(col + kernel_radius + 1, edges_matrix.shape[1])

        zone = aggregate_matrix[row_start:row_end, col_start:col_end]
        borders_in_zone = np.argwhere(zone == 255)

        if len(borders_in_zone) >= threshold:
            mean_x, mean_y = np.mean(borders_in_zone, axis=0).astype(int)
            mean_x += row_start
            mean_y += col_start

            aggregate_matrix[row_start:row_end, col_start:col_end] = 0
            aggregate_matrix[mean_x,mean_y] = 255

    return aggregate_matrix

# Initial image
image = cv2.imread("animals/zebre.jpg")
cv2.imwrite('rendered/0_initial.jpg', image)

def test(image = image):
    start_test = time.time()
    print('Test launched.')

    # Canny
    canny_matrix = canny(image)
    cv2.imwrite('rendered/1_Canny.jpg', canny_matrix)
    end = time.time()

    end_test = time.time()
    print(f'Test done ({round(end_test-start_test, 2)}s).')

def test_all():
    start_test = time.time()

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
    image_blur = gaussian_blur(image, 20, 10)
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
    print(f"-- -- gradients, magnitudes, angles computed in {end - start} seconds.")

    start = time.time()
    outlined_matrix = only_maxima(magnitudes, angles)
    cv2.imwrite('rendered/Canny_3_maxima.jpg', outlined_matrix)
    end = time.time()
    print(f"-- -- Maximas in {end - start} seconds.")

    start = time.time()
    hysteresis_matrix = hysteresis(outlined_matrix)
    cv2.imwrite('rendered/Canny_4_hystersis_FINAL_CANNY.jpg', hysteresis_matrix)
    end = time.time()
    print(f"-- -- Hysteresis computed in {end - start} seconds.")

    end_canny = time.time()
    print(f"-- Canny algorithm computed in {end_canny - start_canny} seconds.")

    # Outliner
    start = time.time()
    outlined_matrix = outliner(hysteresis_matrix,15, 31)
    cv2.imwrite(f'rendered/Z_0_Outlined.jpg', outlined_matrix)
    end = time.time()
    print(f"-- Outliner computed in {end - start} seconds.")

    # Draw
    start = time.time()
    draw_matrix = binary_reverse(hysteresis_matrix)
    cv2.imwrite(f'rendered/Z_01_Draw.jpg', draw_matrix)
    end = time.time()
    print(f"-- Draw_it computed in {end - start} seconds.")

    # Contours
    start = time.time()
    contour_matrix = contour(image)
    cv2.imwrite('rendered/Z_10_countours.jpg', contour_matrix)
    end = time.time()
    print(f"-- Contours drawed in {end - start} seconds.")
       
    
    print(f"== TEST DONE IN {round(time.time()-start_test,2)}s.")

test_all()