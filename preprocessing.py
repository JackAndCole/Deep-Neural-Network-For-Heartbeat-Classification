import pickle
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np
import os
import scipy.signal as sg
import wfdb
from sklearn.utils import cpu_count
from tqdm import tqdm

base_dir = "dataset"

sampling_rate = 360
invalid_labels = ['|', '~', '!', '+', '[', ']', '"', 'x']  # non-beat labels

before = 90
after = 110

tol = 0.05


def pre_processing(record):
    filename = os.path.join(base_dir, record)
    signal = wfdb.rdrecord(filename, sampfrom=0, sampto=None, channels=[0]).p_signal[:, 0]

    baseline = sg.medfilt(sg.medfilt(signal, int(0.2 * sampling_rate) - 1), int(0.6 * sampling_rate) - 1)
    signal = signal - baseline

    annotation = wfdb.rdann(filename, extension="atr", sampfrom=0, sampto=None)
    r_peaks, labels = annotation.sample, np.array(annotation.symbol)

    # remove non-beat labels
    indices = [i for i, label in enumerate(labels) if label not in invalid_labels]
    r_peaks, labels = r_peaks[indices], labels[indices]

    # align r-peaks
    newR = []
    for r_peak in r_peaks:
        r_left = np.maximum(r_peak - int(tol * sampling_rate), 0)
        r_right = np.minimum(r_peak + int(tol * sampling_rate), len(signal))
        newR.append(r_left + np.argmax(signal[r_left:r_right]))
    r_peaks = np.array(newR, dtype="int")

    # r-peaks intervals
    rris = np.diff(r_peaks)

    avg_rri = np.mean(rris)
    x1, x2, y = [], [], []
    for index in tqdm(range(len(r_peaks)), desc=record, file=sys.stdout):
        if index == 0 or index == len(r_peaks) - 1:
            continue
        beat = signal[r_peaks[index] - before: r_peaks[index] + after]

        pre_rri = rris[index - 1]
        post_rri = rris[index]
        ratio_rri = pre_rri / post_rri
        local_rri = np.mean(rris[np.maximum(index - 10, 0):index])

        if labels[index] in ["N", "L", "R", "e", "j"]:
            label = 0  # N
        elif labels[index] in ["A", "a", "S", "J"]:
            label = 1  # SVEB
        elif labels[index] in ["V", "E"]:
            label = 2  # VEB
        elif labels[index] in ["F"]:
            label = 3  # F
        # elif labels[index] in ["/", "f", "Q"]:
        #     label = 4  # Q
        else:
            continue

        x1.append(beat)
        x2.append([pre_rri - avg_rri, post_rri - avg_rri, ratio_rri, local_rri - avg_rri])
        y.append(label)

    return x1, x2, y


if __name__ == "__main__":
    cpus = cpu_count() - 1 if cpu_count() <= 22 else 22  # for multiple processes

    print("train processing...")
    train_records = [
        '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
        '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
        '223', '230'
    ]
    tasks = []
    pool = Pool(cpus)
    for record in train_records:
        tasks.append(pool.apply_async(pre_processing, args=(record,)))
    pool.close()
    pool.join()

    x1_train, x2_train, y_train = [], [], []
    for task in tasks:
        x1, x2, y = task.get()
        x1_train.append(x1)
        x2_train.append(x2)
        y_train.append(y)
    x1_train = np.concatenate(x1_train, axis=0)
    x2_train = np.concatenate(x2_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    print("test processing...")
    test_records = [
        '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
        '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
        '233', '234'
    ]
    tasks = []
    pool = Pool(cpus)
    for record in test_records:
        tasks.append(pool.apply_async(pre_processing, args=(record,)))
    pool.close()
    pool.join()

    x1_test, x2_test, y_test = [], [], []
    for task in tasks:
        x1, x2, y = task.get()
        x1_test.append(x1)
        x2_test.append(x2)
        y_test.append(y)
    x1_test = np.concatenate(x1_test, axis=0)
    x2_test = np.concatenate(x2_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    with open(os.path.join(base_dir, "mitdb.pkl"), "wb") as f:
        pickle.dump((
            (x1_train, x2_train, y_train),
            (x1_test, x2_test, y_test)
        ), f, protocol=4)

    print("train labels:", Counter(y_train))
    print("test labels:", Counter(y_test))
