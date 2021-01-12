from pathlib import Path
from more_itertools import consecutive_groups

import cv2 as cv
import numpy as np

import csv

# TODO: handle it xd
train_notes_list = ["c_a", "c_ais", "c_b", "c_h", "c_c1", "c_cis1", "c_des1", "c_d1", "c_dis1", "c_es1", "c_e1", "c_f1",
                    "c_fis1", "c_ges1", "c_g1",
                    "c_gis1", "c_as1", "c_a1", "c_ais1", "c_b1", "c_h1", "c_c2", "c_cis2", "c_des2", "c_d2", "c_dis2",
                    "c_es2", "c_e2", "c_f2",
                    "c_fis2", "c_ges2", "c_g2", "c_gis2", "c_as2", "c_a2", "c_ais2", "c_b2", "c_h2", "c_c3"]


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
            cv.imwrite(str(output_path / (file.stem + '_' + str(i).zfill(2) + ext)), staff)


def crop_notes(divider_param, max_notes, path_in, path_out, ext):
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

        # wykryj pionowe linie za pomoca operacji morficznych uzaleznionych od wysokosci
        # https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html
        height = staff.shape[1]

        vertical_size = round(height / divider_param)

        kernel_size = (1, vertical_size)

        vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)

        eroded = cv.erode(staff, vertical_structure)
        dilated = cv.dilate(eroded, vertical_structure)

        thinned = cv.ximgproc.thinning(dilated, thinningType=cv.ximgproc.THINNING_GUOHALL)
        #
        # cv.imshow('thinned', thinned)
        # cv.waitKey()

        sum_pixels = np.sum(thinned, axis=0)
        tact_indices = np.where(sum_pixels > 4500)[0]
        tact_margin = 2
        last_idx = 0
        for i in range(0, len(tact_indices)):
            if abs((last_idx + tact_margin) - (tact_indices[i] - tact_margin)) > 5:
                note = staff[:, last_idx + tact_margin:tact_indices[i] - tact_margin]
                last_idx = tact_indices[i]
                # TODO: w nazwie juz podaj wartosc nutki i index, ez (w pliku script jest kolejnosc nut)
                # TODO: tutaj resize przed zapisem??
                cv.imwrite(
                    str(output_path /
                        (file.stem[2] + '_' + train_notes_list[current_note][2:] + '_' + str(note_count).zfill(2) + ext)
                        ), note)
                if note_count + 1 == max_notes:
                    note_count = 0
                    current_note = (current_note + 1) % len(train_notes_list)
                else:
                    note_count += 1


def init_descriptors(method, scale, path_in, path_out, ext_in, ext_out):
    input_path = Path(path_in)
    if input_path.is_dir() is not True:
        raise Exception('input_path should exist, bad input')

    output_path = Path(path_out)
    if output_path.is_dir() is not True:
        output_path.mkdir(parents=True)

    # ORB feature extractor

    if method == 'orb':
        det = cv.ORB_create()
    elif method == 'sift':
        det = cv.SIFT_create()
    elif method == 'kaze':
        det = cv.KAZE_create()
    else:
        det = cv.SIFT_create()

    for file in input_path.glob('*' + ext_in):
        note = cv.imread(str(file), cv.IMREAD_GRAYSCALE)
        threshed = cv.threshold(note, 127, 255, cv.THRESH_BINARY)[1]
        if scale is not None:
            threshed = cv.resize(threshed, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

        note_descriptor = det.detectAndCompute(threshed, None)[1]

        if note_descriptor is not None:
            final_out_dir = output_path / method
            if not final_out_dir.is_dir():
                final_out_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(final_out_dir / (file.stem + ext_out)), note_descriptor)


def get_matches_and_pairs(probe, candidates, threshold_matches, threshold_similarity):
    # BFMatcher with default params
    bf = cv.BFMatcher()

    pairs_count = 0
    matches_count = 0

    for candidate in candidates:
        pairs_count += 1

        matches = bf.knnMatch(probe.template, candidate.template, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < threshold_similarity * n.distance:
                good.append([m])

        if len(good) > threshold_matches:
            matches_count += 1

    return matches_count, pairs_count


def get_fmr_value(threshold_matches, threshold_similarity, all_notes, notes_list):
    sum_pairs = 0
    sum_matches = 0
    for note_name in notes_list:
        the_notes = list(filter(lambda note: note_name == note.name, all_notes))
        other_notes = list(filter(lambda note: note_name != note.name, all_notes))
        for the_note in the_notes:
            matches, pairs = get_matches_and_pairs(the_note, other_notes, threshold_matches, threshold_similarity)
            sum_matches += matches
            sum_pairs += pairs
    if sum_pairs == 0:
        return 0
    return sum_matches / sum_pairs


def get_fnmr_value(threshold_matches, threshold_similarity, all_notes):
    sum_pairs = 0
    sum_matches = 0
    for the_note in all_notes:
        # otherUser.getName().contains(theUser.getName()) && otherUser.getId() != theUser.getId()
        other_notes = list(
            filter(lambda other_note: the_note.name == other_note.name and other_note.id != the_note.id, all_notes))
        matches, pairs = get_matches_and_pairs(the_note, other_notes, threshold_matches, threshold_similarity)
        sum_matches += matches
        sum_pairs += pairs
    if sum_pairs == 0:
        return 0
    return (sum_pairs - sum_matches) / sum_pairs


def get_statistics_to_csv(thresholds_matches, thresholds_similarities, copies_count, path_in):
    input_path = Path(path_in)
    if input_path.is_dir() is not True:
        raise Exception('input_path should exist, bad input')

    # TODO: unique names could be just notes list...

    all_notes = list()

    for file in input_path.iterdir():
        file_name = file.stem
        name, _id = file_name.rsplit('_', 1)
        descriptor = np.load(str(file))
        all_notes.append(NoteDetails(int(_id), name, descriptor))

    all_notes_filtered = list(filter(lambda note: note.id < copies_count, all_notes))

    for thr_similarity in thresholds_similarities:
        with open('roc_sim' + f'{thr_similarity:.2f}' + '.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Threshold', 'FMR', 'FNMR'])
            for thr_match in thresholds_matches:
                # TODO: dodac do nazwy nutki przedrostek c_, p_ etc.
                _fmr = get_fmr_value(thr_match, thr_similarity, all_notes_filtered, train_notes_list)
                _fnmr = get_fnmr_value(thr_match, thr_similarity, all_notes_filtered)
                writer.writerow([thr_match, _fmr, _fnmr])


class NoteDetails:
    def __init__(self, _id, name, template):
        self.id = _id
        self.name = name
        self.template = template

    def __repr__(self):
        return self.name + '_' + str(self.id)


if __name__ == '__main__':
    # Photo extension
    file_ext = '.jpg'

    """Crops all training data sheets"""
    raw_sheets_path = r'C:\Users\Radek\PycharmProjects\omr\raw_sheets'
    cropped_sheets_path = r'C:\Users\Radek\PycharmProjects\omr\cropped_sheets'
    # crop_all_sheets(raw_sheets_path, cropped_sheets_path, file_ext)

    """Preprocess sheets before extracting notes from trainig data sheets"""
    preprocessed_sheets_path = r'C:\Users\Radek\PycharmProjects\omr\preprocessed_sheets'
    page_height = 1600
    page_margin = 40
    # preprocess_sheets(page_height, page_margin, cropped_sheets_path, preprocessed_sheets_path, file_ext)

    """Crop out staffs from preprocessed sheets"""
    staffs_path = r'C:\Users\Radek\PycharmProjects\omr\staffs'
    staff_margin = 5
    # crop_staffs(staff_margin, preprocessed_sheets_path, staffs_path, file_ext)

    """Crop notes by vertical tact lines"""
    notes_path = r'C:\Users\Radek\PycharmProjects\omr\notes'
    height_divider = 33  # used to make kernel that filters only vertical lines if height is 60 then
    each_note_copies = 20  # ile jest nutek danego rodzaju
    # crop_notes(height_divider, each_note_copies, staffs_path, notes_path, file_ext)

    """Compute descriptor for every note"""
    descriptors_path = r'C:\Users\Radek\PycharmProjects\omr\descriptors'
    file_ext_in = file_ext
    file_ext_out = '.npy'
    resize_up_factor = None
    # init_descriptors('sift', resize_up_factor, notes_path, descriptors_path, file_ext_in, file_ext_out)

    """Get statistics FMR and FNMR to plot ROC curve and find threshold that corresponds to EER"""
    note_copies_count = 20  # ile kopii nutek wziac do obliczen
    match_count_threshold = list(range(1, 26, 1))
    similarity_score_threshold = np.arange(0.3, 0.9, 0.05)
    get_statistics_to_csv(match_count_threshold, similarity_score_threshold, note_copies_count, descriptors_path + r'\sift')

    """Test if work"""
    # user_input_path = r'C:\Users\Radek\PycharmProjects\omr\user_input'
    # user_cropped_path = r'C:\Users\Radek\PycharmProjects\omr\user_cropped'
    # crop_all_sheets(user_input_path, user_cropped_path, file_ext)
    # user_processed_path = r'C:\Users\Radek\PycharmProjects\omr\user_processed'
    # preprocess_sheets(page_height, page_margin, user_cropped_path, user_processed_path, file_ext)
    # user_staffs_path = r'C:\Users\Radek\PycharmProjects\omr\user_staffs'
    # crop_staffs(staff_margin, user_processed_path, user_staffs_path, file_ext)
