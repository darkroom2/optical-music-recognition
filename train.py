import subprocess
from pathlib import Path
from time import sleep, perf_counter

import cv2 as cv

# TODO: KARO przygotowac nuty do testow
# TODO: Z - zdjecie, S - screen
# TODO: wlazl_kotek_na_plotek           S,
#       wlazl_kotek_na_plotek_2         S,
#       soviet_anthem                   X,
#       toss_a_coin_to_your_witcher     X,
#       remember_me                     X,
#       remember_me                     X,
#       remember_me                     X,
#       remember_me                     X,
#       remember_me                     X,
# TODO: Zrobic screena jak i zdjecie i wrzucic do raw i dac dobrom nazwe
# TODO: Napisac skrypt testujacy i porownujacy wektory nut
# TODO: Sprawdzic jak dziala test.py od yolov5


def main():
    cmds = [
        # 'python train.py --img 96 --batch 273 --epochs 300 --data ../config/my_config.yaml --cfg ../config/yolov5x.yaml --weights yolov5x.pt --name yolov5x_notes_s96_b273_e300 --cache --device 0',
        # 'python train.py --img 96 --batch 256 --epochs 300 --data ../config/my_config.yaml --cfg ../config/yolov5x.yaml --weights yolov5x.pt --name yolov5x_notes_s96_b256_e300 --cache --device 0',
    ]

    for cmd in cmds:
        sleep(1)
        tic = perf_counter()
        subprocess.call(cmd, shell=True, cwd=f'./yolov5')
        toc = perf_counter()
        sleep(1)
        print(f'# DONE in {toc - tic} s.')
        sleep(5)


def draw_notes():
    for f in Path('./yolo_notes/labels/train').iterdir():
        img = cv.imread(str(f).replace('labels', 'images').replace('.txt', '.jpg'))
        img_h, img_w = img.shape[:2]

        c, x, y, w, h = f.read_text().split()

        """
            x, y, w, h = cv.boundingRect(threshed)
            x_cnt = x + w / 2
            y_cnt = y + h / 2
            norm_bounds = [x_cnt / img_w, y_cnt / img_h, w / img_w, h / img_h]
        """

        box_w = float(w) * img_w
        box_h = float(h) * img_h
        p1x = float(x) * img_w - box_w / 2
        p1y = float(y) * img_h - box_h / 2
        p2x = p1x + box_w
        p2y = p1y + box_h

        img = cv.rectangle(img, (int(p1x), int(p1y)), (int(p2x), int(p2y)), (0, 255, 0), 1)
        cv.imshow('xd', img)
        cv.waitKey()


if __name__ == '__main__':
    main()
    # draw_notes()
