import csv
from pathlib import Path

from matplotlib import pyplot as plt


def rename_items(path_in, path_out):
    photos_dir = Path(path_in)
    out_dir = Path(path_out)
    for persons_dir in photos_dir.iterdir():
        person_name = persons_dir.stem
        for arm_dir in persons_dir.iterdir():
            if arm_dir.is_dir():
                arm_name = arm_dir.stem
                for finger_dir in arm_dir.iterdir():
                    if finger_dir.is_dir():
                        finger_name = finger_dir.stem
                        for i, the_finger in enumerate(finger_dir.iterdir()):
                            # Rado_Lewa_Kciuk_1
                            out_path = out_dir / person_name
                            out_path.mkdir(parents=True, exist_ok=True)
                            out_file = out_path / (
                                    person_name + '_' + arm_name + '_' + finger_name + '_' + str(i + 1) + '.jpg')
                            the_finger.rename(out_file)


def rename_files(path_in, path_out):
    photos_dir = Path(path_in)
    out_dir = Path(path_out)

    for file in photos_dir.rglob('*Kamila_*'):
        file.replace(str(file).replace('Kamila', 'Kama'))
        print(file.parent, file.name)


def plot_roc(path_roc, path_out):
    thr = list()
    fmr = list()
    fnmr = list()
    tmr = list()
    with open(path_roc, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader, None)
        for row in reader:
            thr.append(float(row[0].replace(',', '.')))
            fmr.append(float(row[1].replace(',', '.')))
            fnmr.append(float(row[2].replace(',', '.')))
            tmr.append(1 - float(row[2].replace(',', '.')))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    axes[0].plot(fmr, fnmr)
    axes[0].set_xlabel('fmr')
    axes[0].set_ylabel('fnmr')
    axes[0].set_title('DET curve')
    axes[0].grid(True)

    axes[1].plot(fmr, tmr)
    axes[1].set_xlabel('fmr')
    axes[1].set_ylabel('tmr')
    axes[1].set_title('ROC curve')
    axes[1].grid(True)

    fig.savefig(path_out + r'\plot.png')

    # plt.show()
    # print('xd')


def main():
    path_in = r'C:\Users\Radek\Dysk Google\karo rzeczy\Inz\do_bazy\palce'
    path_out = r'C:\Users\Radek\Dysk Google\karo rzeczy\Inz\do_bazy\staty\10_osob_end'
    path_roc = r'C:\Users\Radek\Dysk Google\karo rzeczy\Inz\do_bazy\staty\10_osob_end\roc.csv'

    # rename_items(path_in, path_out)
    plot_roc(path_roc, path_out)
    # rename_files(path_in, path_out)


if __name__ == '__main__':
    main()
