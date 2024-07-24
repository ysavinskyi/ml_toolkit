import numpy as np
from matplotlib import pyplot as plt
import cv2

from models.clusterization import Clusterization
from cv.img_preprocessing.img_preprocessing import *


def main(aerodrome_image, plane_image):
    """
    Demonstrates the clusterization of image using K-means and founds the needed cluster where sample object is
    :param aerodrome_image: image of aerodrome
    :param plane_image: image of the plane on the aerodrome
    """
    origin_image = cv2.imread(aerodrome_image)

    image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title('Original aerodrome image')
    plt.show()

    image = apply_gaussian_blur(image, (7, 7))
    plt.imshow(image)
    plt.title('Blurred aerodrome image')
    plt.show()

    pixel_array = image.reshape(-1, 3)

    n_clusters = 5
    model = Clusterization('k-means')
    img_kmeans = model.fit(pixel_array, k=n_clusters)

    cluster_mask = img_kmeans.labels_.reshape(image.shape[:2])

    contours_list = []
    for i in range(n_clusters):
        cluster = np.zeros(image.shape[:2])
        cluster[cluster_mask == i] = 255

        contours, _ = cv2.findContours(np.uint8(cluster), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_list.append(contours)

        plt.imshow(cluster)
        plt.title(f'Aerodrome image cluster index #{i}')
        plt.show()

    # Processing plane sample image to get its mean color
    plane_sample = cv2.imread(plane_image)
    plt.imshow(plane_sample)
    plt.title('Original plane sample image')
    plt.show()

    plane_sample = apply_median_blur(plane_sample, 5)
    plt.imshow(plane_sample)
    plt.title('Blured plane sample image')
    plt.show()

    plane_sample = cv2.cvtColor(plane_sample, cv2.COLOR_BGR2RGB)
    plane_sample_array = plane_sample.reshape(-1, 3)

    plane_kmeans = model.fit(plane_sample_array, k=2)
    plane_mask = plane_kmeans.labels_.reshape(plane_sample.shape[:2])
    plt.imshow(plane_mask)
    plt.title('Binary clusterized plane image')
    plt.show()

    # Defining the least numerous class on binary image to consider as the object
    unique_values, counts = np.unique(plane_mask, return_counts=True)
    plane_class = min(zip(unique_values, counts), key=lambda x: x[1])[0]
    plane_color = plane_kmeans.cluster_centers_[plane_class]
    # Finding the closest color cluster using Euclidean distance
    euclid_dist = np.linalg.norm(img_kmeans.cluster_centers_ - plane_color, axis=1)
    img_target_cluster = np.argmin(euclid_dist)

    contour_image = np.copy(image)
    cv2.drawContours(contour_image, contours_list[img_target_cluster], -1, (0, 255, 0), 2)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title(f'Contours of cluster #{img_target_cluster}')
    plt.imshow(contour_image)
    plt.show()


if __name__ == '__main__':
    input_n = None
    while input_n != 0:
        print('==== This script will show you how clusterization by color could help in finding object contours\n')
        input_n = input('Select a sample number (1,2,3) or 0 to exit:\n')
        if input_n != 0:
            main(f'../cv/images_source/aerodrome{input_n}.jpg',
                 f'../cv/images_source/plane_sample{input_n}.png')
        else:
            print('==== Thank you for testing!')
