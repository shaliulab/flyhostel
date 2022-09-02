import matplotlib.pyplot as plt
import numpy as np

def test_rotate_animal():
    contour = np.array([
        [0, 0], [1, 1], [2, 1], [1, 2], [2, 2], [2, 3], [3, 2], [3, 3 ], [4, 4]
    ]).reshape((-1,1,2))
    s=5
    crop = np.zeros((s, s), np.uint8)

    for i in range(contour.shape[0]):
        pt = contour[i,0,:]
        crop[s-1-pt[1], pt[0]] = 255
        # plt.scatter(*pt)
    rotated, (T, mask, contour_center) = rotate_animal(crop, contour)
    plt.imshow(rotated)
    plot_rotation(crop, mask, T, "test.png")
    plt.imshow(frame)
    print(centroid, contour.shape)
    