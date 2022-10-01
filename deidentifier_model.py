def deidentifier_model(file_seed, device, num_workers, batch_size, hospitals, vendors):
    import torch
    import transformers
    from transformers import pipeline
    from transformers.utils import logging
    import pandas as pd
    import collections
    import numpy as np

    logging.set_verbosity_error()

    reports = None

    with open("original_reports" + file_seed + ".npy", "rb") as f:
        reports = np.load(f, allow_pickle=True)

    reports = reports.tolist()

    classifier = pipeline(
        "token-classification",
        model="StanfordAIMI/stanford-deidentifier-base",
        device=device,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    classifier.model.to(device)

    # If not tokenizer fast, we can not use offsets mappings and the code will break
    assert classifier.tokenizer.is_fast

    # a little bit of clean up
    labels = [
        "PATIENT",
        "HCW",
        "HOSPITAL",
        "DATE",
        "ID",
        "GEO",
        "DATES",
        "PHONE",
        "FAX",
        "EMAIL",
        "SSN",
        "MRN",
        "PLAN",
        "ACCOUNT",
        "LICENSE",
        "VIN",
        "DEVICE",
        "WEB",
        # "IP",
        "BIOMETRIC",
        "PHOTO",
        "UNIQUE",
        "VENDOR",
    ]

    for i in range(len(reports)):
        report = reports[i]
        report = report.replace("\\", "")
        report = report.replace("[[", "")
        report = report.replace("]]", "")
        report = report.strip()

        # for label in labels:
        #    report = report.replace(label, "")

        reports[i] = report

    import time

    break_tokens = [":", ";", "."]
    break_tokens_ids = []

    reports_save = reports[:]

    for break_token in break_tokens:
        break_tokens_ids.append(
            int(classifier.preprocess(break_token)["input_ids"][0][1])
        )

    reports_tokenized = classifier.tokenizer(
        reports,
        truncation=False,
        return_special_tokens_mask=True,
        return_offsets_mapping=classifier.tokenizer.is_fast,
    )

    input_ids = reports_tokenized["input_ids"]
    offset_mapping = reports_tokenized["offset_mapping"]

    for i in range(len(offset_mapping)):
        offset_mapping[i][-1] = (offset_mapping[i][-2][1], offset_mapping[i][-2][1])

    # We are going to modify reports in place
    # and if we pick a random token because no break tokens, make sure it is not a subtoken

    assert len(input_ids) == len(reports)

    report_idx_to_leftover_chunks_idx = collections.defaultdict(list)

    for i in range(len(reports)):
        if len(input_ids[i]) <= 512:
            # no need to break anything
            continue
        else:
            # need to break
            report = reports[i]
            char_start_index = 0
            input_id_start_index = 0
            j_to_split_upon = None
            j = 0

            while j < len(input_ids[i]):
                # we split every 508 tokens (slightly conservative for the first and last chunks)
                input_id = input_ids[i][j]

                if input_id in break_tokens_ids:
                    j_to_split_upon = j

                if j and (
                    (j - input_id_start_index) == 508 or j == len(input_ids[i]) - 1
                ):
                    # we split
                    if j == len(input_ids[i]) - 1 and input_id_start_index == j:
                        break

                    if j_to_split_upon is None:
                        j_to_split_upon = j
                        # print(i, j_to_split_upon)

                        if j < len(input_ids[i]) - 1:
                            while (
                                classifier.tokenizer.convert_ids_to_tokens(
                                    input_ids[i][j_to_split_upon + 1]
                                )[:2]
                                == "##"
                                and j - j_to_split_upon <= 100
                                and j_to_split_upon > input_id_start_index + 1
                            ):
                                # it is a subtoken
                                j_to_split_upon -= 1

                        if j - j_to_split_upon >= 100:
                            print(
                                "particularly long token that might need investigation in report at index :",
                                i,
                            )

                    assert j_to_split_upon is not None

                    char_end_index = offset_mapping[i][j_to_split_upon][1]

                    if char_start_index == 0:
                        # the first split
                        reports[i] = report[:char_end_index]
                    else:
                        report_idx_to_leftover_chunks_idx[i].append(len(reports))
                        reports.append(report[char_start_index:char_end_index])

                    char_start_index = char_end_index
                    # we go back to where we cut
                    assert j_to_split_upon >= input_id_start_index
                    j = j_to_split_upon + 1
                    input_id_start_index = j
                    j_to_split_upon = None

                else:
                    j += 1

            assert report == reports[i] + "".join(
                [reports[j] for j in report_idx_to_leftover_chunks_idx[i]]
            )

    # and use the offset thing and we should be good !

    assert not any(
        [
            (x > 512)
            for x in list(
                map(
                    len,
                    classifier.tokenizer(
                        reports,
                        truncation=False,
                        return_special_tokens_mask=True,
                        return_offsets_mapping=classifier.tokenizer.is_fast,
                    )["input_ids"],
                )
            )
        ]
    )

    for i in range(len(reports_save)):
        assert reports_save[i] == reports[i] + "".join(
            [reports[j] for j in report_idx_to_leftover_chunks_idx[i]]
        )

    # there is this propagation thing to explore

    import time

    start = time.time()
    predictions = classifier(reports)
    print("inference_time", time.time() - start)
    print("processed", len(reports), "reports")

    # We can pluging rule based model with the template
    # def rule_based_detector(report)
    # return [{'entity_group': 'ID',
    #'score': 0.6958507,
    #  'word': 'fkjhg567 -',
    #  'start': 19}]
    # And then we merge these things

    # we can add a rule based that filters out some words ! with words that are too similar

    ages = set(
        [
            "90",
            "91",
            "92",
            "93",
            "94",
            "95",
            "96",
            "97",
            "98",
            "99",
            "100",
            "101",
            "102",
            "103",
            "104",
            "105",
            "106",
            "107",
            "108",
            "109",
            "110",
            "111",
            "112",
            "113",
            "114",
            "115",
            "116",
            "117",
            "118",
            "119",
            "120",
            "121",
            "122",
            "123",
            "124",
            "125",
            "126",
            "127",
            "128",
            "129",
            "130",
            "ninety",
            "ninety-one",
            "ninety-two",
            "ninety-three",
            "ninety-four",
            "ninety-five",
            "ninety-six",
            "ninety-seven",
            "ninety-eight",
            "ninety-nine",
            "ninety",
            "ninety one",
            "ninety two",
            "ninety three",
            "ninety four",
            "ninety five",
            "ninety six",
            "ninety seven",
            "ninety eight",
            "ninety nine",
            "ninety",
            "ninetyone",
            "ninetytwo",
            "ninetythree",
            "ninetyfour",
            "ninetyfive",
            "ninetysix",
            "ninetyseven",
            "ninetyeight",
            "ninetynine",
        ]
    )

    suffix_to_exclude = set(
        map(
            lambda x: x.lower(),
            [
                " ml",
                "  ml",
                "/",
                " mg",
                "mg",
                "  mg",
                "ml",
                "mg",
                " /",
                "-ML",
                " mL",
                "mL",
                "  mL",
                " ml",
                "  ml",
                "ml",
                ".0",
                ".1",
                ".2",
                ".3",
                ".4",
                ".5",
                ".6",
                ".7",
                ".8",
                ".9",
                " cc",
                "  mL",
                "th",
                " beats",
                "m",
                " cm",
                "  cm",
                "cm",
                " mm",
                "  mm",
                "mm",
                " mcg",
                "  mcg",
                "mcg",
                "  minutes",
                " minutes",
                "minutes",
                " g/m",
                "g/m",
                "  g/m",
                " bpm",
                "bpm",
                "  bpm",
                "  cc",
                "cc",
                " cc",
                "%",
                "grams",
                " grams",
                "  grams",
            ],
        )
    )

    prefix_to_exclude = set(
        map(
            lambda x: x.lower(),
            ["0.", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "image "],
        )
    )

    suffix_to_exclude_lengths = set(map(len, suffix_to_exclude))
    prefix_to_exclude_lengths = set(map(len, prefix_to_exclude))

    ages_lengths = set(map(len, ages))

    # the one hundred numbers
    # fautes d'orthographe
    def rule_based_age_detector(report):
        i = 0
        predictions = []

        while i < len(report):
            for length in ages_lengths:
                if i + length <= len(report) and report[i : i + length].lower() in ages:

                    if report[i : i + length].isnumeric() and i + length < len(report):
                        if report[i + length].isnumeric():
                            continue

                        need_to_continue = False
                        for length_2 in suffix_to_exclude_lengths:
                            if (
                                i + length + length_2 <= len(report)
                                and report[i + length : i + length + length_2].lower()
                                in suffix_to_exclude
                            ):
                                need_to_continue = True
                                break
                        if need_to_continue:
                            continue

                    if report[i : i + length].isnumeric() and i > 0:
                        if report[i - 1].isnumeric():
                            continue

                        need_to_continue = False
                        for length_2 in prefix_to_exclude_lengths:
                            if (
                                i - length_2 >= 0
                                and report[i - length_2 : i].lower()
                                in prefix_to_exclude
                            ):
                                need_to_continue = True
                                break
                        if need_to_continue:
                            continue

                    predictions.append(
                        {
                            "entity": "AGE",
                            "score": 0.9,
                            "index": None,
                            "word": report[i : i + length],
                            "start": i,
                            "end": i + length,
                        }
                    )
                    i += length
                    break
            else:
                i += 1

        return predictions

    hospitals = [x.lower() for x in hospitals]
    hospitals_lengths = set(map(len, hospitals))

    def rule_based_hospital_model(report):
        i = 0
        predictions = []

        while i < len(report):
            for length in hospitals_lengths:
                if (
                    i + length <= len(report)
                    and report[i : i + length].lower() in hospitals
                ):

                    predictions.append(
                        {
                            "entity": "HOSPITAL",
                            "score": 0.9,
                            "index": None,
                            "word": report[i : i + length],
                            "start": i,
                            "end": i + length,
                        }
                    )
                    i += length
                    break
            else:
                i += 1

        return predictions

    vendors = [x.lower() for x in vendors]
    if "@" not in vendors:
        vendors.append("@")

    vendors_lengths = set(map(len, vendors))

    def rule_based_vendor_model(report):
        i = 0
        predictions = []

        while i < len(report):
            for length in vendors_lengths:
                if (
                    i + length <= len(report)
                    and report[i : i + length].lower() in vendors
                ):

                    predictions.append(
                        {
                            "entity": "VENDOR",
                            "score": 0.9,
                            "index": None,
                            "word": report[i : i + length],
                            "start": i,
                            "end": i + length,
                        }
                    )
                    i += length
                    break
            else:
                i += 1

        return predictions

    # test function
    import numpy as np

    def generate_bit_map_prediction(prediction_list, report):
        bit_map = np.zeros(len(report), dtype=int)

        for prediction in prediction_list:
            bit_map[prediction["start"] : prediction["end"]] = 1

        return bit_map

    import copy

    def merge_predictions(prediction_list, prediction_rule_based_list, report):
        prediction_list_save = copy.deepcopy(prediction_list)
        bit_map = generate_bit_map_prediction(prediction_list, report)

        # per report
        for i in range(len(prediction_list) - 1):
            assert prediction_list[i]["start"] < prediction_list[i]["end"]
            assert prediction_list[i]["end"] <= prediction_list[i + 1]["start"]

        if len(prediction_list):
            assert prediction_list[-1]["start"] < prediction_list[-1]["end"]

        for i in range(len(prediction_rule_based_list) - 1):
            assert (
                prediction_rule_based_list[i]["start"]
                < prediction_rule_based_list[i]["end"]
            )
            assert (
                prediction_rule_based_list[i]["end"]
                <= prediction_rule_based_list[i + 1]["start"]
            )

        if len(prediction_rule_based_list):
            assert (
                prediction_rule_based_list[-1]["start"]
                < prediction_rule_based_list[-1]["end"]
            )

        prediction_list_index = 0
        prediction_rule_based_list_index = 0

        while prediction_rule_based_list_index < len(
            prediction_rule_based_list
        ) and prediction_list_index < len(prediction_list):

            if (
                prediction_list[prediction_list_index]["end"]
                <= prediction_rule_based_list[prediction_rule_based_list_index]["start"]
            ):
                # normal_end <= rule_based_start
                prediction_list_index += 1

            else:
                # normal_end > rule_based_start

                if (
                    prediction_list[prediction_list_index]["start"]
                    >= prediction_rule_based_list[prediction_rule_based_list_index][
                        "end"
                    ]
                ):
                    # normal_start >= rule_based_end
                    prediction_list.insert(
                        prediction_list_index,
                        prediction_rule_based_list[prediction_rule_based_list_index],
                    )
                    prediction_list_index += 1
                    prediction_rule_based_list_index += 1

                elif (
                    prediction_list[prediction_list_index]["start"]
                    <= prediction_rule_based_list[prediction_rule_based_list_index][
                        "start"
                    ]
                    and prediction_list[prediction_list_index]["end"]
                    >= prediction_rule_based_list[prediction_rule_based_list_index][
                        "end"
                    ]
                ):
                    # normal_start <= rule_based_start and normal_end >= rule_based_end
                    prediction_rule_based_list_index += 1

                # elif prediction_list[prediction_list_index]['start'] >= \
                # prediction_rule_based_list[prediction_rule_based_list_index]['start'] and\
                # prediction_list[prediction_list_index]['end'] <= \
                # prediction_rule_based_list[prediction_rule_based_list_index]['end']:
                #    #normal_start >= rule_based_start and normal_end <= rule_based_end
                else:
                    if (
                        prediction_list[prediction_list_index]["start"]
                        > prediction_rule_based_list[prediction_rule_based_list_index][
                            "start"
                        ]
                    ):
                        temp = copy.deepcopy(
                            prediction_rule_based_list[prediction_rule_based_list_index]
                        )
                        temp["end"] = prediction_list[prediction_list_index]["start"]
                        temp["word"] = report[temp["start"] : temp["end"]]
                        prediction_list.insert(prediction_list_index, temp)
                        prediction_list_index += 1

                    if (
                        prediction_list[prediction_list_index]["end"]
                        < prediction_rule_based_list[prediction_rule_based_list_index][
                            "end"
                        ]
                    ):
                        temp = copy.deepcopy(
                            prediction_rule_based_list[prediction_rule_based_list_index]
                        )
                        temp["start"] = prediction_list[prediction_list_index]["end"]
                        temp["word"] = report[temp["start"] : temp["end"]]
                        prediction_rule_based_list[
                            prediction_rule_based_list_index
                        ] = temp
                    else:
                        prediction_rule_based_list_index += 1

        while prediction_rule_based_list_index < len(prediction_rule_based_list):
            prediction_list.append(
                prediction_rule_based_list[prediction_rule_based_list_index]
            )
            prediction_rule_based_list_index += 1

        assert len(set(map(lambda x: tuple(x.items()), prediction_list_save))) == len(
            prediction_list_save
        )
        assert len(prediction_list_save) == len(
            set(map(lambda x: tuple(x.items()), prediction_list_save)).intersection(
                set(map(lambda x: tuple(x.items()), prediction_list))
            )
        )

        assert not (
            bit_map
            & np.logical_not(generate_bit_map_prediction(prediction_list, report))
        ).sum()

        generate_bit_map_prediction(prediction_list, report)
        return prediction_list

    assert len(predictions) == len(reports)

    # You can put whatever rule based model that takes a report and output the prediction format
    # And then we create the function that merges the predictions !
    rule_based_models = [
        rule_based_age_detector,
        rule_based_hospital_model,
        rule_based_vendor_model,
    ]

    for i in range(len(reports)):
        for rule_based_model in rule_based_models:
            prediction_rule_based = rule_based_model(reports[i])
            predictions[i] = merge_predictions(
                predictions[i], prediction_rule_based, reports[i]
            )

    # Check that the predictions are correctly ordered, and no overlapping predictions
    for prediction in predictions:
        for i in range(len(prediction) - 1):
            assert prediction[i]["start"] < prediction[i]["end"]
            assert prediction[i]["end"] <= prediction[i + 1]["start"]
        if len(prediction):
            assert prediction[-1]["start"] < prediction[-1]["end"]

    def propagate_predictions_to_letters_around(prediction_list, report):
        prediction_index = 0
        bit_map = generate_bit_map_prediction(prediction_list, report)

        while prediction_index < len(prediction_list):
            # We must propagate on the left and on the right

            # the left limit is included
            left_limit_report_index = (
                prediction_list[prediction_index - 1]["end"]
                if prediction_index > 0
                else 0
            )
            # the right limit is not included
            right_limit_report_index = (
                prediction_list[prediction_index + 1]["start"]
                if prediction_index < len(prediction_list) - 1
                else len(report)
            )

            start_index = prediction_list[prediction_index]["start"]
            end_index = prediction_list[prediction_index]["end"]

            while (
                start_index > left_limit_report_index
                and report[start_index - 1].isalnum()
            ):
                start_index -= 1

            while end_index < right_limit_report_index and report[end_index].isalnum():
                end_index += 1

            prediction_list[prediction_index]["start"] = start_index
            prediction_list[prediction_index]["end"] = end_index
            prediction_list[prediction_index]["word"] = report[start_index:end_index]

            prediction_index += 1

        assert not (
            bit_map
            & np.logical_not(generate_bit_map_prediction(prediction_list, report))
        ).sum()

        return prediction_list

    for i in range(len(reports)):
        predictions[i] = propagate_predictions_to_letters_around(
            predictions[i], reports[i]
        )

    # Check that the predictions are correctly ordered, and no overlapping predictions
    for prediction in predictions:
        for i in range(len(prediction) - 1):
            assert prediction[i]["start"] < prediction[i]["end"]
            assert prediction[i]["end"] <= prediction[i + 1]["start"]
        if len(prediction):
            assert prediction[-1]["start"] < prediction[-1]["end"]

    def fuse_continuous_predictions(prediction_list, report):
        # if a continuous span of characters has different labels, we fuse into one span with one label
        index = 0
        bit_map = generate_bit_map_prediction(prediction_list, report)

        while (
            index < len(prediction_list) - 1
        ):  # not a problem in python if prediction_list is mutated during the execution
            # of the inner loop

            if prediction_list[index]["end"] < prediction_list[index + 1]["start"]:
                index += 1

            elif prediction_list[index]["end"] > prediction_list[index + 1]["start"]:
                raise Exception("overlapping")

            else:
                assert (
                    prediction_list[index]["end"] == prediction_list[index + 1]["start"]
                )
                start_index = index
                label_to_prob = collections.defaultdict(int)
                label_to_prob[prediction_list[index]["entity"]] += prediction_list[
                    index
                ]["score"] * (
                    prediction_list[index]["end"] - prediction_list[index]["start"]
                )

                while (
                    index + 1 < len(prediction_list)
                    and prediction_list[index]["end"]
                    == prediction_list[index + 1]["start"]
                ):
                    index += 1
                    label_to_prob[prediction_list[index]["entity"]] += prediction_list[
                        index
                    ]["score"] * (
                        prediction_list[index]["end"] - prediction_list[index]["start"]
                    )

                end_index = index

                label_to_assign = max(label_to_prob, key=lambda x: label_to_prob[x])
                prob_to_assign = max(label_to_prob.values())

                prediction_list[start_index]["entity"] = label_to_assign
                prediction_list[start_index]["score"] = prob_to_assign
                prediction_list[start_index]["end"] = prediction_list[end_index]["end"]
                prediction_list[start_index]["index"] = None
                prediction_list[start_index]["word"] = report[
                    prediction_list[start_index]["start"] : prediction_list[
                        start_index
                    ]["end"]
                ]

                for _ in range(start_index + 1, end_index + 1):
                    prediction_list.pop(start_index + 1)

                index = start_index + 1

        assert not (
            bit_map != generate_bit_map_prediction(prediction_list, report)
        ).sum()

        return prediction_list

    for i in range(len(reports)):
        predictions[i] = fuse_continuous_predictions(predictions[i], reports[i])

    # Check that the predictions are correctly ordered, and no overlapping predictions
    for prediction in predictions:
        for i in range(len(prediction) - 1):
            assert prediction[i]["start"] < prediction[i]["end"]
            assert prediction[i]["end"] <= prediction[i + 1]["start"]
        if len(prediction):
            assert prediction[-1]["start"] < prediction[-1]["end"]

    characters_that_can_be_skipped = set(
        [
            "\t",
            "\n",
            "\r",
            "\x0b",
            "\x0c",
            " ",
        ]
    )

    def fuse_neighbor_predictions_from_the_same_class(prediction_list, report):
        index = 0

        bit_map = generate_bit_map_prediction(prediction_list, report)

        while index < len(prediction_list) - 1:
            if prediction_list[index]["end"] >= prediction_list[index + 1]["start"]:
                raise Exception("overlapping")

            assert prediction_list[index]["end"] < prediction_list[index + 1]["start"]

            if prediction_list[index]["entity"] != prediction_list[index + 1]["entity"]:
                index += 1

            elif all(
                [
                    char in characters_that_can_be_skipped
                    for char in report[
                        prediction_list[index]["end"] : prediction_list[index + 1][
                            "start"
                        ]
                    ]
                ]
            ):
                assert (
                    prediction_list[index]["entity"]
                    == prediction_list[index + 1]["entity"]
                )

                prediction_list[index]["score"] = (
                    prediction_list[index]["score"]
                    * (prediction_list[index]["end"] - prediction_list[index]["start"])
                    + prediction_list[index + 1]["score"]
                    * (
                        prediction_list[index + 1]["end"]
                        - prediction_list[index + 1]["start"]
                    )
                ) / (
                    (prediction_list[index]["end"] - prediction_list[index]["start"])
                    + (
                        prediction_list[index + 1]["end"]
                        - prediction_list[index + 1]["start"]
                    )
                )

                prediction_list[index]["end"] = prediction_list[index + 1]["end"]
                prediction_list[index]["index"] = None
                prediction_list[index]["word"] = report[
                    prediction_list[index]["start"] : prediction_list[index]["end"]
                ]

                prediction_list.pop(index + 1)

            else:
                index += 1

        assert not (
            bit_map
            & np.logical_not(generate_bit_map_prediction(prediction_list, report))
        ).sum()

        return prediction_list

    for i in range(len(reports)):
        predictions[i] = fuse_neighbor_predictions_from_the_same_class(
            predictions[i], reports[i]
        )

    # Check that the predictions are correctly ordered, and no overlapping predictions
    for prediction in predictions:
        for i in range(len(prediction) - 1):
            assert prediction[i]["start"] < prediction[i]["end"]
            assert prediction[i]["end"] <= prediction[i + 1]["start"]
        if len(prediction):
            assert prediction[-1]["start"] < prediction[-1]["end"]

    model_labels_to_hips_labels = {
        "VENDOR": "VENDOR",
        "DATE": "DATES",
        "HCW": "HCW",
        "HOSPITAL": "HOSPITAL",
        "ID": "UNIQUE",
        "PATIENT": "PATIENT",
        "PHONE": "PHONE",
        "AGE": "AGE",
    }

    labeled_reports = []
    assert len(reports) == len(predictions)

    for i in range(len(reports)):
        labeled_report = reports[i]
        offset = 0

        for prediction in predictions[i]:
            assert (
                labeled_report[
                    prediction["start"] + offset : prediction["end"] + offset
                ]
                == prediction["word"]
            )
            labeled_report = (
                labeled_report[: prediction["start"] + offset]
                + "\\"
                + model_labels_to_hips_labels[prediction["entity"]]
                + "[["
                + labeled_report[
                    prediction["start"] + offset : prediction["end"] + offset
                ]
                + "]]"
                + labeled_report[prediction["end"] + offset :]
            )

            offset += 1 + len(model_labels_to_hips_labels[prediction["entity"]]) + 2 + 2

        labeled_reports.append(labeled_report)

    labeled_reports_reconstituated = []

    for i in range(len(reports_save)):
        labeled_reports_reconstituated.append(
            labeled_reports[i]
            + "".join(
                [labeled_reports[j] for j in report_idx_to_leftover_chunks_idx[i]]
            )
        )

    assert len(labeled_reports_reconstituated) == len(reports_save)

    with open("labeled_reports" + file_seed + ".npy", "wb") as f:
        np.save(
            f,
            np.array(labeled_reports_reconstituated).astype("object"),
            allow_pickle=True,
        )

    with open("original_reports" + file_seed + ".npy", "wb") as f:
        np.save(f, np.array(reports_save).astype("object"), allow_pickle=True)

    len(reports_save) == len(labeled_reports_reconstituated)

    return
