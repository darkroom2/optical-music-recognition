import cv2 as cv
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    cv.imshow('xd', np.ones((50, 50), np.uint8) * 111)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    print_hi('PyCharm')
