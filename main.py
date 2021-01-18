import csv
from pathlib import Path

import cv2 as cv
import numpy as np
from more_itertools import consecutive_groups


def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def transform_and_crop_a4(path):
    """Original input"""
    src = cv.imread(path)

    """Convert to grayscale for further processing"""
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    """Blur to remove noise and preserve edges"""
    blurred = cv.bilateralFilter(gray, 9, 200, 200)

    """Threshold the image with adaptive thresholding"""
    threshed = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 115, 4)

    # # jadro operacji morfologicznej otwierania i zamykania
    # kernel = np.ones((9, 9), np.uint8)
    # # Otwarcie usuwa małe obiekty z pierwszego planu (zwykle brane jako jasne piksele) obrazu, umieszczając je w tle
    # threshed = cv.morphologyEx(threshed, cv.MORPH_OPEN, kernel)
    # # Zamknięcie usuwa małe otwory w pierwszym planie, zmieniając małe wysepki tła na pierwszy plan.
    # threshed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

    """Find edges on image"""
    edges = cv.Canny(threshed, 200, 250)

    # resized_xd = cv.resize(threshed, None, fx=0.2, fy=0.2, interpolation=cv.INTER_AREA)
    # cv.imshow('resized', resized_xd)
    # cv.waitKey()

    """Get the sheet contour"""
    # Max contour area, but lower than the input image area (to prevent taking input shape as page)
    max_countour_area = (edges.shape[0] - 10) * (edges.shape[1] - 10)
    # Min contour area to not process anything smaller than half of the screen
    half_of_screen_area = max_countour_area * 0.5

    # Contours of edge image
    contours = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    # Approximate contours to find rectangle-like shapes
    contours = [cv.approxPolyDP(x, 0.03 * cv.arcLength(x, True), True) for x in contours]
    # Valid contour's area lies between MAX_ and MIN_ and have 4 vertices and is convex.
    valid_cnts = list(filter(
        lambda cnt: len(cnt) == 4 and cv.isContourConvex(cnt) and half_of_screen_area < cv.contourArea(
            cnt) < max_countour_area, contours))

    if valid_cnts is None:
        raise Exception('valid_cnts should not be empty, bad input')

    # The page contour is max of the valid contours
    page_cnt = max(valid_cnts, key=cv.contourArea)

    # Convert contour to array of coordinates (vertices)
    page_vertices = np.asarray([x[0] for x in page_cnt.astype(dtype=np.float32)])

    # Find top_left (has the smallest sum) and bottom_right (has the biggest)
    top_left = min(page_vertices, key=lambda t: t[0] + t[1])
    bottom_right = max(page_vertices, key=lambda t: t[0] + t[1])
    top_right = max(page_vertices, key=lambda t: t[0] - t[1])
    bottom_left = min(page_vertices, key=lambda t: t[0] - t[1])

    max_width = int(max(distance(bottom_right, bottom_left), distance(top_right, top_left)))
    max_height = int(max(distance(top_right, bottom_right), distance(top_left, bottom_left)))

    arr = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    rectangle = np.asarray([top_left, top_right, bottom_right, bottom_left])
    perspective = cv.getPerspectiveTransform(rectangle, arr)
    result = cv.warpPerspective(src, perspective, (max_width, max_height))

    return result


def crop_all_sheets(path_in, path_out, ext):
    input_path = Path(path_in)
    if input_path.is_dir() is not True:
        raise Exception('input_path should exist, bad input')

    output_path = Path(path_out)
    if output_path.is_dir() is not True:
        output_path.mkdir(parents=True)

    for file in input_path.glob('*' + ext):
        sheet = transform_and_crop_a4(str(file))
        cv.imwrite(str(output_path / file.name), sheet)


def preprocess_sheets(target_height, target_margin, path_in, path_out, ext):
    input_path = Path(path_in)
    if input_path.is_dir() is not True:
        raise Exception('input_path should exist, bad input')

    output_path = Path(path_out)
    if output_path.is_dir() is not True:
        output_path.mkdir(parents=True)

    for file in input_path.glob('*' + ext):
        sheet = cv.imread(str(file), cv.IMREAD_GRAYSCALE)

        # przeskalowac do uniwersalnego rozmiaru (taki sam dla kazdego sheetu)
        resized = cv.resize(sheet, (round(target_height / 2 ** 0.5), target_height), interpolation=cv.INTER_AREA)

        # przyciecie zeby usunac ciemne pola przy krawedziach
        width, height = resized.shape[:2]
        cropped = resized[target_margin:width - target_margin, target_margin:height - target_margin]

        # threshold zeby policzyc pixele
        threshed = cv.adaptiveThreshold(cropped, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 101, 25)

        cv.imwrite(str(output_path / file.name), threshed)


def crop_staffs(target_margin, path_in, path_out, ext):
    input_path = Path(path_in)
    if input_path.is_dir() is not True:
        raise Exception('input_path should exist, bad input')

    output_path = Path(path_out)
    if output_path.is_dir() is not True:
        output_path.mkdir(parents=True)

    for file in input_path.glob('*' + ext):
        preprocessed_sheet = cv.imread(str(file), cv.IMREAD_GRAYSCALE)

        # liczymy pixele i znajdujemy przedzialy gdzie suma jest rozna od zera (miejsca w ktorych wystepuja pixele)
        sum_pixels = np.sum(preprocessed_sheet, axis=1)
        staff_indices = np.where(sum_pixels > 0)[0]

        # znajdujemy przedzialy z pieciolinia
        ranges = [list(x) for x in consecutive_groups(staff_indices)]
        ranges_lens = [len(x) for x in ranges]
        mean_len = sum(ranges_lens) / len(ranges)
        valid_ranges = []
        for x in ranges:
            the_range = list(x)
            if len(the_range) > mean_len * 0.75:
                valid_ranges.append(the_range)

        for i, staff_range in enumerate(valid_ranges):
            staff = preprocessed_sheet[staff_range[0] - target_margin:staff_range[-1] + target_margin, :]
            threshed = cv.threshold(staff, 127, 255, cv.THRESH_BINARY)[1]
            x, y, w, h = cv.boundingRect(threshed)
            key_margin = 60
            key_cropped = threshed[y:y + h, x + key_margin:x + w]
            x, y, w, h = cv.boundingRect(key_cropped)
            bound_cropped = key_cropped[y:y + h, x:x + w]
            cv.imwrite(str(output_path / (file.stem + '_' + str(i).zfill(2) + ext)), bound_cropped)


def crop_notes(divider_param, max_notes, train_notes_list, path_in, path_out, ext):
    input_path = Path(path_in)
    if input_path.is_dir() is not True:
        raise Exception('input_path should exist, bad input')

    output_path = Path(path_out)
    if output_path.is_dir() is not True:
        output_path.mkdir(parents=True)

    note_count = 0
    current_note = 0
    for file in input_path.glob('*' + ext):
        staff = cv.imread(str(file), cv.IMREAD_GRAYSCALE)
        # TODO: dopisac komenty ze threshold po to zeby artefakty kompresji jpg usunac (lub wgl zmienic na png)
        staff = cv.threshold(staff, 127, 255, cv.THRESH_BINARY)[1]

        # https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html
        # wykryj poziome linie zeby wiedziec odkad zaczac zliczac piksele
        width = staff.shape[0]
        horizontal_size = round(width / divider_param)
        kernel_horizontal = (horizontal_size + 5, 1)
        horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, kernel_horizontal)
        staff_horizontal = cv.erode(staff, horizontal_structure)
        staff_horizontal = cv.dilate(staff_horizontal, horizontal_structure)

        sum_horizontal = np.sum(staff_horizontal, axis=1).astype(np.float32)
        dilated_h = cv.dilate(sum_horizontal, np.ones((3, 3), np.uint8))
        line_indices = np.where(dilated_h > np.std(dilated_h) * 2)[0]

        # TODO: pobawic sie tymi nutkami (przycinanie pieciolinii perfekto juz jest)
        # wykryj pionowe linie za pomoca operacji morficznych uzaleznionych od wysokosci
        height = staff.shape[1]
        vertical_size = round(height / divider_param)
        kernel_vertical = (1, vertical_size)
        vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, kernel_vertical)
        staff_vertical = cv.erode(staff, vertical_structure)
        staff_vertical = cv.dilate(staff_vertical, vertical_structure)

        cropped = staff_vertical[line_indices[0] - 5:line_indices[-1] + 5]

        sum_vertical = np.sum(cropped, axis=0).astype(np.float32)
        dilated_v = cv.dilate(sum_vertical, np.ones((3, 3), np.uint8))

        dilated_v[dilated_v == 0] = np.nan

        tact_indices = np.where(dilated_v > (np.nanmean(dilated_v) - np.nanstd(dilated_v) * 3))[0]
        tact_margin = 4
        last_idx = 0
        for i in range(0, len(tact_indices)):
            if abs((last_idx + tact_margin) - (tact_indices[i] - tact_margin)) > 25:

                note = staff[:, last_idx + tact_margin:tact_indices[i] - tact_margin]
                last_idx = tact_indices[i]

                if note.shape[1] > 50:
                    note = note[:, 0:40]
                cv.imwrite(
                    str(output_path /
                        (file.stem[2] + '_' + train_notes_list[current_note][2:] + '_' + str(note_count).zfill(2) + ext)
                        ), note)
                if note_count + 1 == max_notes:
                    note_count = 0
                    current_note = (current_note + 1) % len(train_notes_list)
                else:
                    note_count += 1


# def get_matches_and_pairs(probe, candidates, threshold_matches, threshold_similarity):
#     # BFMatcher with default params
#     bf = cv.BFMatcher()
#
#     pairs_count = 0
#     matches_count = 0
#
#     for candidate in candidates:
#         pairs_count += 1
#
#         matches = bf.knnMatch(probe.template, candidate.template, k=2)
#
#         # Apply ratio test
#         good = []
#         for m, n in matches:
#             if m.distance < threshold_similarity * n.distance:
#                 good.append([m])
#
#         if len(good) > threshold_matches:
#             matches_count += 1
#
#     return matches_count, pairs_count


# def get_fmr_value(threshold_matches, threshold_similarity, all_notes, notes_list):
#     sum_pairs = 0
#     sum_matches = 0
#     for note_name in notes_list:
#         the_notes = list(filter(lambda note: note_name == note.name, all_notes))
#         other_notes = list(filter(lambda note: note_name != note.name, all_notes))
#         for the_note in the_notes:
#             matches, pairs = get_matches_and_pairs(the_note, other_notes, threshold_matches, threshold_similarity)
#             sum_matches += matches
#             sum_pairs += pairs
#     if sum_pairs == 0:
#         return 0
#     return sum_matches / sum_pairs


# def get_fnmr_value(threshold_matches, threshold_similarity, all_notes):
#     sum_pairs = 0
#     sum_matches = 0
#     for the_note in all_notes:
#         # otherUser.getName().contains(theUser.getName()) && otherUser.getId() != theUser.getId()
#         other_notes = list(
#             filter(lambda other_note: the_note.name == other_note.name and other_note.id != the_note.id, all_notes))
#         matches, pairs = get_matches_and_pairs(the_note, other_notes, threshold_matches, threshold_similarity)
#         sum_matches += matches
#         sum_pairs += pairs
#     if sum_pairs == 0:
#         return 0
#     return (sum_pairs - sum_matches) / sum_pairs


# def get_statistics_to_csv(thresholds_matches, thresholds_similarities, copies_count, train_notes_list, path_in):
#     input_path = Path(path_in)
#     if input_path.is_dir() is not True:
#         raise Exception('input_path should exist, bad input')
#
#     # TODO: unique names could be just notes list...
#
#     all_notes = list()
#
#     for file in input_path.iterdir():
#         file_name = file.stem
#         name, _id = file_name.rsplit('_', 1)
#         descriptor = np.load(str(file))
#         all_notes.append(NoteDetails(int(_id), name, descriptor))
#
#     all_notes_filtered = list(filter(lambda note: note.id < copies_count, all_notes))
#
#     for thr_similarity in thresholds_similarities:
#         with open(f'roc_sim_{thr_similarity:.2f}.csv', mode='w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(['Threshold', 'FMR', 'FNMR'])
#             for thr_match in thresholds_matches:
#                 _fmr = get_fmr_value(thr_match, thr_similarity, all_notes_filtered, train_notes_list)
#                 _fnmr = get_fnmr_value(thr_match, thr_similarity, all_notes_filtered)
#                 writer.writerow([thr_match, _fmr, _fnmr])


# class NoteDetails:
#     def __init__(self, _id, name, template):
#         self.id = _id
#         self.name = name
#         self.template = template
#
#     def __repr__(self):
#         return self.name + '_' + str(self.id)


# def stats(train_notes_list, notes_path, descriptors_path, ext_in):
#     """Compute descriptor for every note"""
#     ext_out = '.npy'
#     resize_up_factor = None
#     # init_descriptors('sift', resize_up_factor, notes_path, descriptors_path, ext_in, ext_out)
#
#     """Get statistics FMR and FNMR to plot ROC curve and find threshold that corresponds to EER"""
#     note_copies_count = 20  # ile kopii nutek wziac do obliczen
#     match_count_thresh = list(range(3, 5, 1))
#     similarity_score_thresh = np.arange(0.5, 0.7, 0.1)
#     get_statistics_to_csv(match_count_thresh, similarity_score_thresh, note_copies_count, train_notes_list,
#                           descriptors_path + r'\sift')
#
#     """Test if work"""
#     # user_input_path = r'C:\Users\Radek\PycharmProjects\omr2\user_raw'
#     # user_cropped_path = r'C:\Users\Radek\PycharmProjects\omr2\user_cropped'
#     # crop_all_sheets(user_input_path, user_cropped_path, file_ext)
#     # user_processed_path = r'C:\Users\Radek\PycharmProjects\omr2\user_processed'
#     # preprocess_sheets(page_height, page_margin, user_cropped_path, user_processed_path, file_ext)
#     # user_staffs_path = r'C:\Users\Radek\PycharmProjects\omr2\user_staffs'
#     # crop_staffs(staff_margin, user_processed_path, user_staffs_path, file_ext)


def user(user_raw_path, user_staffs_path, ext):
    """Crop all user sheets"""
    user_cropped_path = r'C:\Users\Radek\PycharmProjects\omr2\user_cropped'
    crop_all_sheets(user_raw_path, user_cropped_path, ext)

    """Preprocess sheets before extracting staffs from user data sheets"""
    user_preprocessed_path = r'C:\Users\Radek\PycharmProjects\omr2\user_preprocessed'
    page_height = 1600
    page_margin = 40
    preprocess_sheets(page_height, page_margin, user_cropped_path, user_preprocessed_path, ext)

    """Crop out staffs from preprocessed sheets"""
    staff_margin = 5
    crop_staffs(staff_margin, user_preprocessed_path, user_staffs_path, ext)


def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv.SIFT_create()
        norm = cv.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv.xfeatures2d.SURF_create(800)
        norm = cv.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv.ORB_create(400)
        norm = cv.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv.AKAZE_create()
        norm = cv.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv.BRISK_create()
        norm = cv.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv.NORM_L2:
            flann_params = dict(algorithm=1, trees=5)
        else:
            flann_params = dict(algorithm=6,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
        matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv.BFMatcher(norm)
    return detector, matcher


def prepare_for_cascades(negs_path, positives_path):
    input_path = Path(negs_path)
    neg = input_path.parent / r'neg.txt'
    if input_path.is_dir() is not True:
        raise Exception('input_path should exist, bad input')

    with open(neg, 'w') as bg:
        for file in input_path.iterdir():
            bg.write(str(file.relative_to(input_path.parent)) + '\n')

    input_path2 = Path(positives_path)
    pos = input_path2.parent / r'pos.txt'
    if input_path2.is_dir() is not True:
        raise Exception('input_path should exist, bad input')

    with open(pos, 'w') as pos_file:
        for file in input_path2.iterdir():
            note = cv.imread(str(file), cv.IMREAD_GRAYSCALE)
            threshed = cv.threshold(note, 127, 255, cv.THRESH_BINARY)[1]
            x, y, w, h = cv.boundingRect(threshed)
            # (x, y, width, height)
            bounds = ' '.join(str(x) for x in [x, y, w, h])
            pos_file.write(str(file.relative_to(input_path2.parent)) + ' 1 ' + bounds + '\n')


def prepare_for_yolo(positives_path, dict_of_notes):
    input_path = Path(positives_path)

    if input_path.is_dir() is not True:
        raise Exception('input_path should exist, bad input')

    # generate txt file with bboxes of the objects for each photo
    for file in input_path.glob('*.jpg'):
        # format per row: class x_center y_center width height
        name = file.name[:file.name.rindex('_')]
        note_class = dict_of_notes[name]

        note = cv.imread(str(file), cv.IMREAD_GRAYSCALE)
        img_h, img_w = note.shape[:2]
        threshed = cv.threshold(note, 127, 255, cv.THRESH_BINARY)[1]
        x, y, w, h = cv.boundingRect(threshed)
        x_cnt = x + w / 2
        y_cnt = y + h / 2
        norm_bounds = [x_cnt / img_w, y_cnt / img_h, w / img_w, h / img_h]
        # (x, y, width, height)
        bounds_str = ' '.join(f'{i:.6f}' for i in norm_bounds)

        note_txt = input_path / 'txt' / (str(file.stem) + '.txt')
        note_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(str(note_txt), 'w+') as pos_file:
            pos_file.write(f'{note_class} {bounds_str}')


def split_for_train_valid_test(positives_path, ratio_train, ratio_valid, ratio_test):
    if round(ratio_train + ratio_valid + ratio_test, 5) != 1.0:
        raise Exception('Split ratio exceeds 1.0')

    each_note_count = 20
    train_notes_count = int(each_note_count * ratio_train)
    validation_notes_count = int(each_note_count * ratio_valid)
    # test_notes_count = each_note_count * ratio_test

    input_path = Path(positives_path)

    if input_path.is_dir() is not True:
        raise Exception('input_path should exist, bad input')

    train_dir = input_path.parent / 'train'
    validate_dir = input_path.parent / 'validate'
    test_dir = input_path.parent / 'test'

    for _dir in [train_dir, validate_dir, test_dir]:
        _dir.mkdir(parents=True, exist_ok=True)

    for file in input_path.iterdir():
        _id = int(file.stem[file.stem.rindex('_') + 1:])
        if _id < train_notes_count:
            file.rename(train_dir / file.name)
        elif _id < train_notes_count + validation_notes_count:
            file.rename(validate_dir / file.name)
        else:
            file.rename(test_dir / file.name)

    #     all_notes_filtered = list(filter(lambda note: note.id < copies_count, all_notes))


def main():
    # List of notes included in train data
    train_notes_list = ['c_a', 'c_ais', 'c_b', 'c_h', 'c_c1', 'c_cis1', 'c_des1', 'c_d1', 'c_dis1', 'c_es1', 'c_e1',
                        'c_f1', 'c_fis1', 'c_ges1', 'c_g1', 'c_gis1', 'c_as1', 'c_a1', 'c_ais1', 'c_b1', 'c_h1', 'c_c2',
                        'c_cis2', 'c_des2', 'c_d2', 'c_dis2', 'c_es2', 'c_e2', 'c_f2', 'c_fis2', 'c_ges2', 'c_g2',
                        'c_gis2', 'c_as2', 'c_a2', 'c_ais2', 'c_b2', 'c_h2', 'c_c3', 'o_a', 'o_ais', 'o_b', 'o_h',
                        'o_c1', 'o_cis1', 'o_des1', 'o_d1', 'o_dis1', 'o_es1', 'o_e1', 'o_f1', 'o_fis1', 'o_ges1',
                        'o_g1', 'o_gis1', 'o_as1', 'o_a1', 'o_ais1', 'o_b1', 'o_h1', 'o_c2', 'o_cis2', 'o_des2', 'o_d2',
                        'o_dis2', 'o_es2', 'o_e2', 'o_f2', 'o_fis2', 'o_ges2', 'o_g2', 'o_gis2', 'o_as2', 'o_a2',
                        'o_ais2', 'o_b2', 'o_h2', 'o_c3', 'p_a', 'p_ais', 'p_b', 'p_h', 'p_c1', 'p_cis1', 'p_des1',
                        'p_d1', 'p_dis1', 'p_es1', 'p_e1', 'p_f1', 'p_fis1', 'p_ges1', 'p_g1', 'p_gis1', 'p_as1',
                        'p_a1', 'p_ais1', 'p_b1', 'p_h1', 'p_c2', 'p_cis2', 'p_des2', 'p_d2', 'p_dis2', 'p_es2', 'p_e2',
                        'p_f2', 'p_fis2', 'p_ges2', 'p_g2', 'p_gis2', 'p_as2', 'p_a2', 'p_ais2', 'p_b2', 'p_h2', 'p_c3',
                        'w_a', 'w_ais', 'w_b', 'w_h', 'w_c1', 'w_cis1', 'w_des1', 'w_d1', 'w_dis1', 'w_es1', 'w_e1',
                        'w_f1', 'w_fis1', 'w_ges1', 'w_g1', 'w_gis1', 'w_as1', 'w_a1', 'w_ais1', 'w_b1', 'w_h1', 'w_c2',
                        'w_cis2', 'w_des2', 'w_d2', 'w_dis2', 'w_es2', 'w_e2', 'w_f2', 'w_fis2', 'w_ges2', 'w_g2',
                        'w_gis2', 'w_as2', 'w_a2', 'w_ais2', 'w_b2', 'w_h2', 'w_c3']

    dict_of_notes = {train_notes_list[i]: i for i in range(0, len(train_notes_list))}

    # Raw photo extension
    ext = '.jpg'

    """Crop all training data sheets"""
    # Path with raw train sheets
    raw_sheets_path = r'/sheets/database/raw'
    # Path where to store cropped sheets
    cropped_sheets_path = r'/sheets/database/cropped'

    crop_all_sheets(raw_sheets_path, cropped_sheets_path, ext)

    """Preprocess sheets before extracting notes from trainig data sheets"""
    # Path where to store preprocessed sheets
    preprocessed_sheets_path = r'/sheets/database/preprocessed'
    # What height to resize the page to
    page_height = 1600
    # How much to crop the borders
    page_margin = 40

    preprocess_sheets(page_height, page_margin, cropped_sheets_path, preprocessed_sheets_path, ext)

    """Crop out staffs from preprocessed sheets"""
    # Path where to store the staffs
    staffs_path = r'/sheets/database/staffs'
    # How much to crop borders of staff image
    staff_margin = 5

    crop_staffs(staff_margin, preprocessed_sheets_path, staffs_path, ext)

    """Crop notes by vertical tact lines"""
    # Path where to store individual notes
    notes_path = r'/sheets/database/notes'
    # Parameter determining vertical and horizontal filter kernel dimensions
    height_divider = 33
    # How many notes of each type in training dataset
    each_note_copies = 20

    crop_notes(height_divider, each_note_copies, train_notes_list, staffs_path, notes_path, ext)

    # Path where to store notes descriptors
    # descriptors_path = r'C:\Users\Radek\PycharmProjects\omr2\descriptors'

    # Path with raw user sheets
    # user_raw_path = r'C:\Users\Radek\PycharmProjects\omr2\user_raw'

    # Path where to store individual user staffs
    # user_staffs_path = r'C:\Users\Radek\PycharmProjects\omr2\user_staffs'

    # stats(train_notes_list, notes_path, descriptors_path, ext)
    # user(user_raw_path, user_staffs_path, ext)
    # test(user_staffs_path, notes_path, ext)

    # Path with negative samples
    # negs_path = r'C:\Users\Radek\PycharmProjects\omr2\negs'

    # Path with positive samples
    # positives_path = r'C:\Users\Radek\PycharmProjects\omr2\notes'

    # prepare_for_cascades(negs_path, positives_path)
    # prepare_for_yolo(positives_path, dict_of_notes)
    # split_for_train_valid_test(positives_path, 0.7, 0.2, 0.1)

    # test(user_staffs_path, notes_path, ext)


if __name__ == '__main__':
    main()
