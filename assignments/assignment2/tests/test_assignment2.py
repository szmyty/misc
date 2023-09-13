from __future__ import annotations

from itertools import islice, zip_longest

import numpy as np

from assignment2.assignment2 import nparray_tail, sliding_window, window2


class TestAssignment2:
    def test_window(self: TestAssignment2, resources):
        sony_training_labels = np.load(
            str(resources.joinpath("sony_training_labels.npy")), allow_pickle=True
        )

        test_list = nparray_tail(sony_training_labels, 41).tolist()

        testagain = list(window2(test_list))
        print(testagain)

        test2 = [list(x) for x in sliding_window(test_list)]
        print(test2)

    def test_sublist_freq(self, resources):
        sony_training_labels = np.load(
            str(resources.joinpath("sony_training_labels.npy")), allow_pickle=True
        )

        print(sony_training_labels)
        # initializing list
        test_list = [4, 5, 3, 5, 7, 8, 3, 5, 7, 2, 7, 3, 2]
        test_list = sony_training_labels.tolist()

        # printing original list
        print("The original list is : " + str(test_list))

        # initializing Sublist
        sublist = ["-", "-", "-"]

        # slicing is used to extract chunks and compare
        res = len(
            [
                sublist
                for idx in range(len(test_list))
                if test_list[idx : idx + len(sublist)] == sublist
            ]
        )

        # printing result
        print("The sublist count : " + str(res))

    def test2(self, resources):
        sony_training_labels = np.load(
            str(resources.joinpath("sony_training_labels.npy")), allow_pickle=True
        )

        print(sony_training_labels)
        # initializing list
        test_list = [4, 5, 3, 5, 7, 8, 3, 5, 7, 2, 7, 3, 2]

        t = str(type(sony_training_labels))
        print(t)
        test_list = sony_training_labels.tolist()

        # printing original list
        print("The original list is : " + str(test_list))

        # initializing Sublist
        sublist = ["-", "-", "-"]

        # slicing is used to extract chunks and compare
        res = []
        idx = 0
        while True:
            try:
                # getting to the index
                idx = test_list.index(sublist[0], idx)
            except ValueError:
                break

            # using all() to check for all elements equivalence
            if all(
                x == y
                for (x, y) in zip_longest(
                    sublist, islice(test_list, idx, idx + len(sublist))
                )
            ):
                res.append(sublist)
                idx += len(sublist)
            idx += 1

        res = len(res)

        # printing result
        print("The sublist count : " + str(res))
