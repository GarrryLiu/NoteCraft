import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, square

from extract_symbol.staff import coordinator


def get_connected_components(img_without_staff, img_with_staff):
    """
    Extract connected components from the binary image without staff lines.

    Parameters:
    - img_without_staff: Binary image with staff lines removed.
    - img_with_staff: Original image with staff lines.

    Returns:
    - components: List of extracted components.
    - comp_with_staff: List of corresponding components from the original image.
    - boundary: Bounding boxes of the components.
    """
    components = []
    boundary = []
    bw = 1 - img_without_staff
    label_img = label(bw)
    for region in regionprops(label_img):
        if region.area >= 100:
            boundary.append(region.bbox)

    boundary = sorted(boundary, key=lambda b: b[1])

    comp_with_staff = []
    for bbox in boundary:
        minr, minc, maxr, maxc = bbox
        components.append(img_without_staff[minr:maxr, minc:maxc])
        comp_with_staff.append(img_with_staff[minr:maxr, minc:maxc])
    return components, comp_with_staff, boundary


def split_symbol(imgs_with_staff, imgs_without_staff, segmenter, horizontal):
    """
    Split symbols from images with and without staff lines.

    Parameters:
    - imgs_with_staff: List of images with staff lines.
    - imgs_without_staff: List of images without staff lines.
    - segmenter: Segmenter object for processing.
    - horizontal: Boolean indicating if the image is horizontal.

    Returns:
    - saved_images: List of images containing split symbols.
    """
    imgs_spacing = []
    imgs_rows = []
    coord_imgs = []

    for staff_img in imgs_with_staff:
        spacing, rows, no_staff_img = coordinator(staff_img, horizontal)
        imgs_rows.append(rows)
        imgs_spacing.append(spacing)
        coord_imgs.append(no_staff_img)

    saved_images = []
    for i, no_staff_img in enumerate(coord_imgs):
        primitives, _, _ = get_connected_components(no_staff_img, imgs_with_staff[i])
        row_images = []
        for prim in primitives:
            prim = binary_opening(prim, square(segmenter.most_common - imgs_spacing[i]))
            saved_img = (255 * (1 - prim)).astype(np.uint8)
            row_images.append(saved_img)
        saved_images.append(row_images)

    return saved_images


# if __name__ == "__main__":
#     imgs = read_data("./output/region_with_staff_*.png")
#     for img in imgs:
#         img = rgb2gray(img)
#         horizontal = IsHorizontal(img)

#         imgs_with_staff, imgs_without_staff, segmenter = remove(img)
#         saved_imgs = split_symbol(
#             imgs_with_staff, imgs_without_staff, segmenter, horizontal
#         )
#         for i, row_images in enumerate(saved_imgs):
#             for j, saved_img in enumerate(row_images):
#                 cv2.imwrite(f"./output/processed_{i}_{j}.png", saved_img)
