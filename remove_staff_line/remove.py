import cv2
from skimage.filters import threshold_otsu

from remove_staff_line.segmenter import Segmenter


def remove(img):
    """
    Remove staff lines from a grayscale image.

    Parameters:
    - img: Grayscale input image.

    Returns:
    - imgs_with_staff: List of images with staff lines.
    - imgs_without_staff: List of images without staff lines.
    - segmenter: Segmenter object for further processing.
    """
    _, binary_img = cv2.threshold(img, threshold_otsu(img), 1, cv2.THRESH_BINARY)
    segmenter = Segmenter(binary_img)
    imgs_with_staff = segmenter.regions_with_staff
    imgs_without_staff = segmenter.regions_without_staff

    return imgs_with_staff, imgs_without_staff, segmenter


# if __name__ == "__main__":
#     imgs = read_data("./data/package_aa/*/*.png")
#     imgs_with_staff, imgs_without_staff, segmenter = remove(imgs)

#     # Define the output directory
#     output_path = "./output"
#     os.makedirs(output_path, exist_ok=True)

#     normalized_region = (imgs_with_staff[0] * 255).astype(np.uint8)
#     staff_output_path = os.path.join(output_path, f"region_with_staff_{i}.png")
#     io.imsave(staff_output_path, normalized_region)

#     normalized_region = (imgs_without_staff[0] * 255).astype(np.uint8)
#     no_staff_output_path = os.path.join(output_path, f"region_without_staff_{i}.png")
#     io.imsave(no_staff_output_path, normalized_region)
