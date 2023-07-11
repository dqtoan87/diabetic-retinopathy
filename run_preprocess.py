import numpy as np
import os
import cv2


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img          # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def normal_crop(img_path):
    image = cv2.imread(img_path)
    image = crop_image_from_gray(image)
    h, w, _ = image.shape
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), w / 10), -4, 128)
    image = cv2.resize(image,(1024,1024))
    return image


if __name__ == "__main__":
    img_list = os.listdir("Input")
    N = len(img_list)

    for img_name in img_list:
        print(img_name)
        img_path = os.path.join("Input", img_name)
        image = normal_crop(img_path)
        cv2.imwrite(os.path.join("Output", img_name), image)
