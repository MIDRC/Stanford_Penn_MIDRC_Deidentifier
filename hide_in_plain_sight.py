def hide_in_plain_sight(file_seed):

    import pandas as pd
    import numpy as np
    import json
    import re
    import random
    import xlrd
    import openpyxl

    labeled_reports = None
    with open("labeled_reports" + file_seed + ".npy", "rb") as f:
        labeled_reports = np.load(f, allow_pickle=True)

    original_reports = None
    with open("original_reports" + file_seed + ".npy", "rb") as f:
        original_reports = np.load(f, allow_pickle=True)

    label_list = [
        "PATIENT",
        "HCW",
        "HOSPITAL",
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

    doc_dict = {"empty": 0, "not empty": 0}
    phi_dict = {}
    phi_values_dict = {
        "VENDOR": [],
        "DATES": [],
        "HCW": [],
        "HOSPITAL": [],
        "UNIQUE": [],
        "PATIENT": [],
        "PHONE": [],
        "AGE": [],
    }
    data_per_report = {
        "VENDOR": [],
        "DATES": [],
        "HCW": [],
        "HOSPITAL": [],
        "UNIQUE": [],
        "PATIENT": [],
        "PHONE": [],
        "AGE": [],
    }

    def find_labels(x):
        phi_list = re.findall(r"\\.+?\[\[.+?\]\]", x, flags=re.DOTALL)
        phi_type_list = []
        for phi in phi_list:
            phi_content = re.sub(r"\\.+?\[\[", "", phi, flags=re.DOTALL)
            phi_content = re.sub(r"\]\]", "", phi_content, flags=re.DOTALL)

            phi_label = re.sub(r"\\", "", phi, flags=re.DOTALL)
            phi_label = re.sub(r"\[\[.+?\]\]", "", phi_label, flags=re.DOTALL)

            if (
                phi_label not in phi_values_dict.keys()
                or phi_label not in data_per_report.keys()
            ):
                print(phi_label)
                raise Exception("not found")

            if phi_label in phi_dict.keys():
                phi_dict[phi_label] += 1
                phi_values_dict[phi_label].append(phi_content)
            else:
                phi_dict[phi_label] = 1
                phi_values_dict[phi_label] = [phi_content]
            phi_type_list.append(phi_label)

        for y in set(phi_type_list):
            data_per_report[y].append(0)
        for y in phi_type_list:
            data_per_report[y][-1] += 1
            if data_per_report[y][-1] >= 21:
                pass
                # print(x)

        if len(phi_list) == 0:
            doc_dict["empty"] += 1
        else:
            doc_dict["not empty"] += 1

    for index_, labeled_report in enumerate(labeled_reports):
        find_labels(labeled_report)
        # if index_ == 36:
        #    print(phi_values_dict["DATES"][-10:])

    label_dict = {
        "none": 0,
        "VENDOR": 1,
        "DATES": 2,
        "HCW": 3,
        "HOSPITAL": 4,
        "UNIQUE": 5,
        "PATIENT": 6,
        "PHONE": 7,
        "AGE": 8,
    }

    # ## Check if labelling is correct

    # Checks whether all types of PHI are detected, and otherwise outputs the index to correct
    report_with_problem_index = []

    def find_reports_with_problem(
        index, report, labeled_report, report_with_problem_index, label_dict
    ):
        report_index = 0
        labeled_report_index = 0
        report_with_problem_index.append(index)

        while report_index < len(report) and labeled_report_index < len(labeled_report):
            if labeled_report[labeled_report_index] == "\\":
                found_label = None

                for label in label_dict.keys():
                    if label == "none":
                        continue

                    if (
                        labeled_report[
                            labeled_report_index
                            + 1 : labeled_report_index
                            + 1
                            + len(label)
                        ]
                        == label
                    ):
                        found_label = label
                        break

                assert found_label is not None
                assert (
                    labeled_report[
                        labeled_report_index
                        + 1 : labeled_report_index
                        + 1
                        + len(found_label)
                    ]
                    == found_label
                )
                assert (
                    labeled_report[
                        labeled_report_index
                        + 1
                        + len(found_label) : labeled_report_index
                        + 1
                        + len(found_label)
                        + 2
                    ]
                    == "[["
                )

                labeled_report_index += 1 + len(found_label) + 2
                assert (
                    labeled_report[labeled_report_index - 2 : labeled_report_index]
                    == "[["
                )
                assert (
                    labeled_report[labeled_report_index : labeled_report_index + 2]
                    != "]]"
                )

                while (
                    labeled_report[labeled_report_index : labeled_report_index + 2]
                    != "]]"
                ):
                    assert labeled_report[labeled_report_index] == report[report_index]
                    report_index += 1
                    labeled_report_index += 1

                labeled_report_index += 2

            else:
                assert labeled_report[labeled_report_index] == report[report_index]
                report_index += 1
                labeled_report_index += 1

        assert len(report) == report_index
        assert len(labeled_report) == labeled_report_index
        report_with_problem_index.pop()

        return

    def find_reports_with_problem_old(
        index, report, labeled_report, report_with_problem_index, label_dict
    ):
        split_labeled_sentence = re.split(r"(\W)", labeled_report, flags=re.DOTALL)
        split_labeled_sentence = [
            x for x in split_labeled_sentence if x != " " and x != ""
        ]

        split_sentence = re.split(r"(\W)", report, flags=re.DOTALL)
        split_sentence = [x for x in split_sentence if x != " " and x != ""]

        labeled_sentence_token_number = 0
        labeled_sentence_retokenized = []
        i = 0
        while i < len(split_labeled_sentence) - 5:
            if (
                split_labeled_sentence[i] == "\\"
                and split_labeled_sentence[i + 2] == "["
                and split_labeled_sentence[i + 3] == "["
            ):
                j = i + 4
                try:
                    while not (
                        split_labeled_sentence[j] == "]"
                        and split_labeled_sentence[j + 1] == "]"
                    ):

                        labeled_sentence_retokenized.append(split_labeled_sentence[j])
                        j += 1
                except:
                    print("problem with report {}".format(index))
                    try:
                        print(split_labeled_sentence[i : i + 5])
                    except:
                        pass
                    print(labeled_report)

                    report_with_problem_index.append(index)
                    return
                if split_labeled_sentence[i + 1] not in label_dict.keys():
                    print("problem with report {}".format(index))
                    print(split_labeled_sentence[i + 1])
                    print(labeled_report)
                    report_with_problem_index.append(index)
                    return
                labeled_sentence_token_number += j - i - 4
                i = j + 2
            else:
                labeled_sentence_retokenized.append(split_labeled_sentence[i])
                i += 1
                labeled_sentence_token_number += 1

        while i < len(split_labeled_sentence):
            labeled_sentence_retokenized.append(split_labeled_sentence[i])
            i += 1
            labeled_sentence_token_number += 1

        if labeled_sentence_token_number != len(split_sentence):
            print(index)
            report_with_problem_index.append(index)

    for index, (report, labeled_report) in enumerate(
        zip(original_reports, labeled_reports)
    ):
        find_reports_with_problem(
            index, report, labeled_report, report_with_problem_index, label_dict
        )

    if report_with_problem_index != []:
        raise Exception(
            "Problem with the labelling: this will probably cause PHI leakage !"
        )

    # ## Generate dates

    def parse_date_format(date):
        """Parse the format of a date which is a string
        Returns its format as a string
        """
        if bool(re.match(r".*([a-z]|[A-Z]).*", date, flags=re.DOTALL)):
            # It has some letters in it
            if sum(c.isdigit() for c in date) < 4:
                return "digits_letters_no_year"  #'March 1'
            elif sum(c.isdigit() for c in date) == 4:
                return "digits_letters_no_day"  #'March 2018'
            elif sum(c.isdigit() for c in date) > 4:
                if "," in date:  # and date[date.index(',') - 1].isdigit():
                    return "digits_letters_with_comma"  #'March 1, 2018'
                else:
                    return "digits_letters_no_comma"  #'1 March 2018'
        else:
            # no letters
            if (
                bool(re.match(r"(.*0.{1}\/.*)", date, flags=re.DOTALL))
                and date.count("/") == 2
            ):
                return "full_digits_with_trailing_zeros"  #'01/02/2020'
            elif (
                bool(re.match(r"(.+\/.{4})", date, flags=re.DOTALL))
                or bool(re.match(r"(.{4}\/.+)", date, flags=re.DOTALL))
            ) and date.count("/") == 1:
                return "full_digits_no_day"  #'2/2020'
            elif date.count("/") == 1:
                return "full_digits_no_year"  #'1/2'
            else:
                return "full_digits_no_trailing_zeros"  #'1/1/2020'

    def ensure_frequency_threshold(
        frequencies_dict, threshhold=0.025, f_filter=None, t_filter=None
    ):
        frequencies_dict_equi = frequencies_dict.copy()
        total = 0
        keys_list = []
        for i in frequencies_dict.keys():
            total += frequencies_dict[i]
            keys_list += [i] * frequencies_dict[i]

        modifications_have_just_been_done = True
        hard_limit = 0
        while modifications_have_just_been_done:
            modifications_have_just_been_done = False
            hard_limit += 1
            if hard_limit > 10000:
                print("HARD LIMIT REACHED")
                print(frequencies_dict_equi)
                return

            for i in frequencies_dict_equi.keys():
                if frequencies_dict_equi[i] / total < threshhold and f_filter is None:
                    modifications_have_just_been_done = True
                    frequencies_dict_equi[i] += 1
                    # frequencies_dict_equi[random.choice(keys_list)] += 1
                    total += 1
                elif (
                    f_filter is not None
                    and t_filter is not None
                    and frequencies_dict_equi[i] / total < threshhold
                    and not f_filter(i)
                ):
                    modifications_have_just_been_done = True
                    frequencies_dict_equi[i] += 1
                    # frequencies_dict_equi[random.choice(keys_list)] += 1
                    total += 1

                elif (
                    f_filter is not None
                    and t_filter is not None
                    and frequencies_dict_equi[i] / total < t_filter
                    and f_filter(i)
                ):
                    modifications_have_just_been_done = True
                    frequencies_dict_equi[i] += 1
                    # frequencies_dict_equi[random.choice(keys_list)] += 1
                    total += 1

        return frequencies_dict_equi

    # ensure_frequency_threshold(frequencies_dict_date, 0.025)

    import re

    def count_frequencies(date_list):
        """Takes the list of all dates of the reports
        Returns the number of dates per date format
        """
        else_type = []
        frequencies_dict = {
            "full_digits_no_trailing_zeros": 0,
            "full_digits_with_trailing_zeros": 0,
            "full_digits_no_day": 0,
            "full_digits_no_year": 0,
            "digits_letters_no_year": 0,
            "digits_letters_no_day": 0,
            "digits_letters_with_comma": 0,
            "digits_letters_no_comma": 0,
        }

        for date in date_list:
            date_format = parse_date_format(date)
            if date_format in frequencies_dict.keys():
                frequencies_dict[date_format] += 1
        return ensure_frequency_threshold(
            frequencies_dict,
            0.025,
        )  # lambda key: key == 'full_digits_no_day', 0.05)

    frequencies_dict_date = count_frequencies(phi_values_dict["DATES"])

    def generate_distribution_probabilities(frequencies_dict):
        """Generates a dict of intervals per data format, which are a partititon of [0, 1]"""
        total = 0
        for x in frequencies_dict.values():
            total += x
        # print(total)
        if total == 0:
            total = 1

        cum_prob = 0
        distribution_probabilities = {}
        for i in frequencies_dict.keys():
            distribution_probabilities[i] = (
                cum_prob,
                cum_prob + frequencies_dict[i] / total,
            )
            cum_prob += frequencies_dict[i] / total

        return distribution_probabilities

    distribution_probabilities_date = generate_distribution_probabilities(
        frequencies_dict_date
    )

    # The following functions convert a datetime to a string date
    import datetime

    from full_digits_no_trailing_zeros import (
        convert_date_to_string_full_digits_no_trailing_zeros,
        convert_string_to_date_full_digits_no_trailing_zeros,
    )
    from full_digits_with_trailing_zeros import (
        convert_date_to_string_full_digits_with_trailing_zeros,
        convert_string_to_date_full_digits_with_trailing_zeros,
    )
    from full_digits_no_day import (
        convert_date_to_string_full_digits_no_day,
        convert_string_to_date_full_digits_no_day,
    )
    from full_digits_no_year import (
        convert_date_to_string_full_digits_no_year,
        convert_string_to_date_full_digits_no_year,
    )

    from digits_letters_no_year import (
        convert_date_to_string_digits_letters_no_year,
        convert_string_to_date_digits_letters_no_year,
    )
    from digits_letters_no_day import (
        convert_date_to_string_digits_letters_no_day,
        convert_string_to_date_digits_letters_no_day,
    )
    from digits_letters_with_comma import (
        convert_date_to_string_digits_letters_with_comma,
        convert_string_to_date_digits_letters_with_comma,
    )
    from digits_letters_no_comma import (
        convert_date_to_string_digits_letters_no_comma,
        convert_string_to_date_digits_letters_no_comma,
    )

    convert_date_to_string = {
        "full_digits_no_trailing_zeros": convert_date_to_string_full_digits_no_trailing_zeros,
        "full_digits_with_trailing_zeros": convert_date_to_string_full_digits_with_trailing_zeros,
        "full_digits_no_day": convert_date_to_string_full_digits_no_day,
        "full_digits_no_year": convert_date_to_string_full_digits_no_year,
        "digits_letters_no_year": convert_date_to_string_digits_letters_no_year,
        "digits_letters_no_day": convert_date_to_string_digits_letters_no_day,
        "digits_letters_with_comma": convert_date_to_string_digits_letters_with_comma,
        "digits_letters_no_comma": convert_date_to_string_digits_letters_no_comma,
    }

    convert_string_to_date = {
        "full_digits_no_trailing_zeros": convert_string_to_date_full_digits_no_trailing_zeros,
        "full_digits_with_trailing_zeros": convert_string_to_date_full_digits_with_trailing_zeros,
        "full_digits_no_day": convert_string_to_date_full_digits_no_day,
        "full_digits_no_year": convert_string_to_date_full_digits_no_year,
        "digits_letters_no_year": convert_string_to_date_digits_letters_no_year,
        "digits_letters_no_day": convert_string_to_date_digits_letters_no_day,
        "digits_letters_with_comma": convert_string_to_date_digits_letters_with_comma,
        "digits_letters_no_comma": convert_string_to_date_digits_letters_no_comma,
    }

    from random import randrange
    import random
    from datetime import timedelta

    # from datetime import datetime

    def generate_random_date(start, end):
        """
        This function will return a random datetime between two datetime
        objects.
        Start and end must be round dates
        """
        assert (end - start).days >= 0
        if start == end or (end - start).days < 365:
            return start + datetime.timedelta(days=random.randint(-365, 365))

        delta = end - start
        assert delta.days >= 0
        int_delta = ((delta.days - 1) * 24 * 60 * 60) + delta.seconds
        random_second = None
        if int_delta == 0:
            raise Exception("not supposed to happen")
            random_second = 24 * 60 * 60
        else:
            random_second = 24 * 60 * 60 + randrange(int_delta)
        return (start + timedelta(seconds=random_second)).replace(
            minute=0, second=0, hour=0
        )

    def generate_date(
        date,
        distribution_probabilities,
        convert_date_to_string,
        convert_string_to_date,
        min_date,
        max_date,
        store_dates=None,
        post_distribution_probabilities=None,
        constraint=None,
        index=None,
    ):
        parsed_date = None

        try:
            date_format = parse_date_format(date)
            if post_distribution_probabilities is not None:
                post_distribution_probabilities[date_format] += 1
            parsed_date = convert_string_to_date[date_format](date)
        except:
            try:
                parsed_date = pd.to_datetime(date).to_pydatetime().replace(tzinfo=None)
            except:
                # print('problem with parsing')
                # print(date)
                if index is not None:
                    print(index)
                parsed_date = datetime.datetime.strptime(
                    "1/1/2010", "%m/%d/%Y"
                ) + datetime.timedelta(days=random.randint(-3650, 3650))
                # print(parsed_date)

        if parsed_date.year < 1995:
            parsed_date = parsed_date.replace(year=parsed_date.year + 100)

        if store_dates is not None:
            store_dates.append(parsed_date)

        lower_boundary_date, upper_boundary_date = min_date, max_date

        # if constraint is not None:
        # lower_boundary_date, upper_boundary_date = constraint.get_constraint(min_date, max_date)

        item = None
        random_date = None
        if constraint is not None:
            item = constraint.get_constraint()

        if constraint is not None and item is not None:
            random_date = item

        if random_date is None or random_date == parsed_date:
            if random_date == parsed_date:
                print("####Not supposed to happen ?")

            random_date = generate_random_date(lower_boundary_date, upper_boundary_date)

            n_iterations = 0
            while parsed_date == random_date:
                if n_iterations > 10:
                    print("Stopping non-solvable case, with forced solution")
                    if constraint is not None:
                        random_date = constraint.get_previous_date()
                    else:
                        random_date = datetime.datetime(
                            2010, 1, 1, 0, 0
                        ) + datetime.timedelta(days=random.randint(-365, 365))
                    break
                random_date = generate_random_date(
                    lower_boundary_date, upper_boundary_date
                )
                n_iterations += 1

        if random_date is None:
            raise Exception("not supposed to happen")

        if random_date < max(
            lower_boundary_date - datetime.timedelta(days=30 * 365),
            datetime.datetime(1950, 1, 1),
        ):
            # print("enforcing date")
            # print(date)
            # print(parsed_date)
            # print(random_date)
            random_date = lower_boundary_date + datetime.timedelta(
                days=random.randint(-365, 365)
            )
        elif random_date > min(
            upper_boundary_date + datetime.timedelta(days=30 * 365),
            datetime.datetime(2060, 1, 1),
        ):
            # print("enforcing date")
            # print(date)
            # print(parsed_date)
            # print(random_date)
            random_date = upper_boundary_date + datetime.timedelta(
                days=random.randint(-365, 365)
            )

        seed = random.random()
        for date_format in distribution_probabilities.keys():
            bounds = distribution_probabilities[date_format]
            if seed >= bounds[0] and seed < bounds[1]:
                generated_date = convert_date_to_string[date_format](random_date)
                if constraint is not None:
                    constraint.add_date(random_date)
                if random.random() > 0.6:
                    generated_date = generated_date.replace("/", "-")

                if date == generated_date:
                    print("## special case")
                    print(date, "##", generated_date)
                    generated_date = convert_date_to_string[date_format](
                        random_date + datetime.timedelta(days=random.randint(365, 720))
                    )
                    if constraint is not None:
                        constraint.remove_last_added_date()
                        constraint.add_date(random_date)
                    if random.random() > 0.6:
                        generated_date = generated_date.replace("/", "-")
                    print("##", generated_date)
                return generated_date

    class DateConstraint:
        def __init__(self):
            self.index = 0
            self.constraint_list = (
                []
            )  # list of tuples of min, max which can be equal if we want to enforce equality
            self.already_generated_dates = []

        def add_date(self, date):
            self.already_generated_dates.append(date)
            self.index += 1

        def remove_last_added_date(self):
            self.already_generated_dates.pop()
            self.index -= 1

        def get_constraint(self):
            constraint = self.constraint_list[self.index]
            date = None
            if constraint is not None and len(self.already_generated_dates) > 0:
                date = self.already_generated_dates[self.index - 1] + constraint
            return date

        def get_previous_date(self):
            if self.index == 0:
                return datetime.datetime(2010, 1, 1, 0, 0) + datetime.timedelta(
                    days=random.randint(-3650, 3650)
                )
            else:
                return self.already_generated_dates[self.index - 1]

    def get_date_year(past_date_list, date):
        # Find most similar date in past_date_list, if it exists, based on day and month
        for x in past_date_list:
            if x.day == date.day and x.month == date.month:
                return x

        for i in range(len(past_date_list) - 1, -1, -1):
            if past_date_list[i].year != 1900:
                return datetime.datetime(past_date_list[i].year, date.month, date.day)
        return None

    def generate_date_constraint(report, convert_string_to_date):
        date_list = re.findall(r"\\DATES\[\[.+?\]\]", report, flags=re.DOTALL)
        date_list = [
            re.sub(
                r"\]\]",
                "",
                re.sub(r"\\.+?\[\[", "", date, flags=re.DOTALL),
                flags=re.DOTALL,
            )
            for date in date_list
        ]
        parsed_date_list = []

        for date in date_list:
            parsed_date = None

            try:
                date_format = parse_date_format(date)
                parsed_date = convert_string_to_date[date_format](date)
            except:
                try:
                    parsed_date = (
                        pd.to_datetime(date).to_pydatetime().replace(tzinfo=None)
                    )
                except:
                    parsed_date = (
                        parsed_date_list[-1]
                        if len(parsed_date_list)
                        else datetime.datetime.strptime("1/1/2010", "%m/%d/%Y")
                        + datetime.timedelta(days=random.randint(-3650, 3650))
                    )

            if parsed_date.year < 1995:
                parsed_date = parsed_date.replace(year=parsed_date.year + 100)

            parsed_date_list.append(parsed_date)

        constraint_list = []

        for i, date in enumerate(parsed_date_list):
            if len(constraint_list) == 0:
                constraint_list.append(None)
            else:
                past_date = parsed_date_list[i - 1]
                current_date = date

                if past_date.year == 1900:
                    past_date_candidate = get_date_year(
                        parsed_date_list[: i - 1], past_date
                    )

                    if past_date_candidate == None:
                        if current_date.year != 1900:
                            past_date = datetime.datetime(
                                current_date.year, past_date.month, past_date.day
                            )
                    else:
                        past_date = past_date_candidate

                if current_date.year == 1900 and past_date.year != 1900:
                    current_date = get_date_year(parsed_date_list[:i], current_date)

                assert current_date is not None
                assert past_date is not None

                if abs(current_date.year - past_date.year) > 15:
                    current_date = current_date.replace(day=min(current_date.day, 28))
                    current_date = current_date.replace(
                        year=past_date.year + random.randint(1, 15)
                    )

                constraint_list.append(current_date - past_date)

        constraint = DateConstraint()
        constraint.constraint_list = constraint_list
        return constraint

    # For the min date and max date, we check the range of dates in the reports
    def get_min_max_dates(date_list, convert_string_to_date):
        min_date = None
        max_date = None
        index = 0
        min_index = 0
        for date in date_list:
            parsed_date = None

            try:
                date_format = parse_date_format(date)
                parsed_date = convert_string_to_date[date_format](date)
            except:
                try:
                    parsed_date = (
                        pd.to_datetime(date).to_pydatetime().replace(tzinfo=None)
                    )
                except:
                    parsed_date = datetime.datetime.strptime(
                        "1/1/2010", "%m/%d/%Y"
                    ) + datetime.timedelta(days=random.randint(-365 * 2, 365 * 2))
            if parsed_date.year < 1995:
                parsed_date = parsed_date.replace(year=parsed_date.year + 100)
            if parsed_date.year == 2030:
                continue
            try:
                if (
                    min_date is None or parsed_date < min_date
                ) and parsed_date.year != 1900:
                    min_date = parsed_date
                    min_index = index
                if (
                    max_date is None or parsed_date > max_date
                ) and parsed_date.year != 1900:
                    max_date = parsed_date
            except:
                print(min_date)
                print(max_date)
                print(parsed_date)
                raise Exception("problem with datetime")

            index += 1

        return max(min_date, datetime.datetime.strptime("1/1/2000", "%m/%d/%Y")), min(
            max_date, datetime.datetime.strptime("1/1/2022", "%m/%d/%Y")
        )

    min_date, max_date = get_min_max_dates(
        phi_values_dict["DATES"], convert_string_to_date
    )

    def find_problem_parsing(report):
        for x in re.findall(
            r"\\DATES\[\[.+?\]\]", report.LabeledReport, flags=re.DOTALL
        ):
            generate_date(
                re.sub(
                    r"\]\]",
                    "",
                    re.sub(r"\\.+?\[\[", "", x, flags=re.DOTALL),
                    flags=re.DOTALL,
                ),
                distribution_probabilities_date,
                convert_date_to_string,
                convert_string_to_date,
                min_date,
                max_date,
                store_dates,
                index=report.name,
            )

    # ## Generate healthcare workers

    # Database surnames
    data_surnames = (
        pd.DataFrame(pd.read_csv("./Common_Surnames_Census_2000.csv").iloc[:, 0])
        .astype("str")
        .drop_duplicates()
        .dropna()
    )
    # data_surnames = data_surnames[data_surnames.apply(lambda x: isinstance(x, str), axis = 1)]
    # data_surnames = pd.concat([ data_surnames[:1000], data_surnames.sample(200)])

    # Database firstnames
    data_firstnames = (
        pd.DataFrame(pd.read_excel("./SSA_Names_DB.xlsx").iloc[:, 0])
        .astype("str")
        .drop_duplicates()
        .dropna()
    )
    # data_firstnames = data_firstnames[data_firstnames.apply(lambda x: isinstance(x, str), axis = 1)]
    # data_firstnames = pd.concat([data_firstnames[:1000], data_firstnames.sample(200)])

    data_surnames = [x.capitalize() for x in data_surnames.values.squeeze().tolist()]
    data_firstnames = [
        x.capitalize() for x in data_firstnames.values.squeeze().tolist()
    ]

    assert not len(list(filter(lambda x: (not isinstance(x, str)), data_surnames)))
    assert not len(list(filter(lambda x: (not isinstance(x, str)), data_firstnames)))

    def parse_name_format(name):
        """Parse the format of a name which is a string
        Returns its format as a string
        """
        if name.count(",") == 2:
            return "NAME, NAME, CRE"
        elif name.count(",") == 1:
            return "NAME, FIRSTNAME/NAME, CRE"
        else:
            return "NAME NAME/CRE NAME/NAME CRE"

    def count_name_format_frequencies(name_list):
        """Takes the list of all names of the reports
        Returns the number of names per name format
        """
        frequencies_dict = {
            "NAME, NAME, CRE": 0,
            "NAME, FIRSTNAME/NAME, CRE": 0,
            "NAME NAME/CRE NAME/NAME CRE": 0,
        }

        for name in name_list:
            frequencies_dict[parse_name_format(name)] += 1
        return frequencies_dict

    frequencies_dict_name = count_name_format_frequencies(phi_values_dict["HCW"])

    distribution_probabilities_name = generate_distribution_probabilities(
        frequencies_dict_name
    )

    def convert_name_to_format_1(firstname, middlename, lastname, credential):
        random_number = random.random()
        if random_number > 0.9:
            firstname += " " + middlename[0]
            if random.random() > 0.8:
                firstname += "."
        elif random_number > 0.8:
            firstname += " " + middlename

        if credential[0:2].lower() == "dr" and random.random() > 0.1:
            return "{} {}, {}".format(credential, lastname, firstname)

        if random.random() > 0.5:
            return "{}, {}, {}".format(firstname, lastname, credential)
        else:
            return "{}, {}, {}".format(lastname, firstname, credential)

    def convert_name_to_format_2(firstname, middlename, lastname, credential):
        random_number = random.random()
        if random_number > 0.9:
            firstname += " " + middlename[0]
            if random.random() > 0.6:
                firstname += "."
        elif random_number > 0.8:
            firstname += " " + middlename

        prob_list = [0.875, 0.75, 0.5, 0.25, 0.125]
        if credential[0:2].lower() == "dr":
            prob_list = [0.6, 0.2, 0.15, 0.1, 0.05]

        random_number = random.random()
        if random_number > prob_list[0]:
            return "{}, {}".format(firstname, lastname)
        elif random_number > prob_list[1]:
            return "{}, {}".format(lastname, firstname)
        elif random_number > prob_list[2]:
            return "{} {}, {}".format(firstname, lastname, credential)
        elif random_number > prob_list[3]:
            return "{} {}, {}".format(lastname, firstname, credential)
        elif random_number > prob_list[4]:
            return "{}, {}".format(firstname, credential)
        else:
            return "{}, {}".format(lastname, credential)

    def convert_name_to_format_3(firstname, middlename, lastname, credential):
        random_number = random.random()
        if random_number > 0.9:
            firstname += " " + middlename[0]
            if random.random() > 0.8:
                firstname += "."
        elif random_number > 0.8:
            firstname += " " + middlename

        random_number = random.random()
        prob_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        if credential[0:2].lower() == "dr":
            prob_list = [0.93, 0.86, 0.84, 0.82, 0.58, 0.34, 0.32, 0.3, 0.14]
        if random_number > prob_list[0]:
            return "{} {}".format(firstname, lastname)
        elif random_number > prob_list[1]:
            return "{} {}".format(lastname, firstname)
        elif random_number > prob_list[2]:
            return "{} {} {}".format(firstname, lastname, credential)
        elif random_number > prob_list[3]:
            return "{} {} {}".format(lastname, firstname, credential)
        elif random_number > prob_list[4]:
            return "{} {} {}".format(credential, lastname, firstname)
        elif random_number > prob_list[5]:
            return "{} {} {}".format(credential, firstname, lastname)
        elif random_number > prob_list[6]:
            return "{} {}".format(firstname, credential)
        elif random_number > prob_list[7]:
            return "{} {}".format(lastname, credential)
        elif random_number > prob_list[8]:
            return "{} {}".format(credential, lastname)
        else:
            return "{} {}".format(credential, firstname)

    convert_name_to_string = {
        "NAME, NAME, CRE": convert_name_to_format_1,
        "NAME, FIRSTNAME/NAME, CRE": convert_name_to_format_2,
        "NAME NAME/CRE NAME/NAME CRE": convert_name_to_format_3,
    }

    def insert_dots(credential):
        random_number = random.random()
        if credential == "Dr.":
            if random_number > 0.2:
                return credential
            else:
                return "Dr"
        if random_number > 0.05:
            return credential
        else:
            if credential == "Dr.":
                return "Dr"
            elif credential == "PA-C":
                return "P.A.-C."
            else:
                return "".join([x + "." for x in credential])

    already_used_names = []

    def generate_random_name(
        data_firstnames,
        data_surnames,
        already_used_names,
        already_used_names_more_frequent,
    ):
        """
        This function will return a random name.
        """

        if len(already_used_names) > 0 and random.random() > 0.9:
            if random.random() > 0.4 and len(already_used_names_more_frequent):
                # some names are repeated a lot of times
                firstname, middlename, lastname, credential = random.sample(
                    already_used_names_more_frequent, 1
                )[0]
                return (firstname, middlename, lastname, credential)
            else:
                firstname, middlename, lastname, credential = random.sample(
                    already_used_names, 1
                )[0]
                if random.random() > 1 - 1 / (
                    len(already_used_names_more_frequent) + 1
                ) and (
                    (firstname, middlename, lastname, credential)
                    not in already_used_names_more_frequent
                ):
                    already_used_names_more_frequent.append(
                        (firstname, middlename, lastname, credential)
                    )
                return (firstname, middlename, lastname, credential)

        # We will use predefined list of credentials. This is to avoid any leakage.
        # We can insert except for doctors some points between the letters
        frequent_credentials = ["MD", "Dr."]
        rare_credentials = ["PA", "PA-C", "CRNP"]
        very_rare_credentials = [
            "RN",
            "APNP",
            "APRN",
            "CNM",
            "CNP",
            "CRNA",
            "DNP",
            "LPN",
            "DO",
            "MBBS",
            "MB",
        ]
        random_number = random.random()
        credential = None

        if random_number > 0.15:
            credential = random.sample(frequent_credentials, 1)[0]
        elif random_number > 0.05:
            credential = random.sample(rare_credentials, 1)[0]
        else:
            credential = random.sample(very_rare_credentials, 1)[0]

        firstname = None
        middlename = None
        lastname = None

        if random.random() > 0.1:
            firstname = random.sample(data_firstnames[:1000], 1)[0]
        else:
            firstname = random.sample(data_firstnames, 1)[0]

        if random.random() > 0.1:
            middlename = random.sample(data_firstnames[:1000], 1)[0]
        else:
            middlename = random.sample(data_firstnames, 1)[0]

        if random.random() > 0.1:
            lastname = random.sample(data_surnames[:1000], 1)[0]
        else:
            lastname = random.sample(data_surnames, 1)[0]

        # we must also sometimes replicate a previous name (probability proportional to the size)

        already_used_names.append((firstname, middlename, lastname, credential))
        return firstname, middlename, lastname, credential

    class EqualityConstraint:
        def __init__(self):
            self.index = 0
            self.constraint_list = (
                []
            )  # list of indexes, -1 if new, and the index of a previous occurence if equal to it
            self.already_generated_items = []
            self.default_item = ""

        def add_item(self, item):
            self.already_generated_items.append(item)
            self.index += 1

        def get_constraint(self):
            constraint = self.constraint_list[self.index]
            if constraint == -1:
                return None
            else:
                return self.already_generated_items[constraint]

        def get_previous_item(self):
            if self.index == 0:
                return self.default_item
            else:
                return self.already_generated_items[self.index - 1]

    def generate_equality_constraint(report, are_items_equal, phi_label):
        """are_items_equal is an equality function between two string items, which has to be symmetric"""
        item_list = re.findall(
            r"\\" + phi_label + "\[\[.+?\]\]", report, flags=re.DOTALL
        )
        item_list = [
            re.sub(
                r"\]\]",
                "",
                re.sub(r"\\.+?\[\[", "", item, flags=re.DOTALL),
                flags=re.DOTALL,
            )
            for item in item_list
        ]

        constraint_list = []
        for i, item in enumerate(item_list):
            index = -1
            for j, past_item in enumerate(item_list[:i]):
                if are_items_equal(past_item, item):
                    index = j
                    break
            constraint_list.append(index)

        constraint = EqualityConstraint()
        constraint.constraint_list = constraint_list
        return constraint

    def are_names_equal(old_name, new_name):
        credentials = set(
            map(
                lambda x: x.lower(),
                [
                    "MD",
                    "Dr.",
                    "Dr",
                    "PA",
                    "PA-C",
                    "CRNP",
                    "RN",
                    "APNP",
                    "APRN",
                    "CNM",
                    "CNP",
                    "CRNA",
                    "DNP",
                    "LPN",
                    "DO",
                    "MBBS",
                    "MB",
                ],
            )
        )

        old_name_set = (
            set(re.sub(r"[^\w]", " ", old_name, flags=re.DOTALL).lower().split())
            - credentials
        )
        new_name_set = (
            set(re.sub(r"[^\w]", " ", new_name, flags=re.DOTALL).lower().split())
            - credentials
        )

        return len(old_name_set.intersection(new_name_set)) > 0

    already_used_names = []
    already_used_names_more_frequent = []

    def generate_name(
        name,
        distribution_probabilities,
        convert_name_to_string,
        data_firstnames,
        data_surnames,
        already_used_names,
        already_used_names_more_frequent,
        post_distribution_probabilities=None,
        constraint=None,
        store_names=None,
    ):

        parsed_name = name.lower().split()
        # print(constraint.constraint_list)

        item = None
        firstname, middlename, lastname, credential = None, None, None, None
        if constraint is not None:
            item = constraint.get_constraint()
        if constraint is not None and item is not None:
            firstname, middlename, lastname, credential = item
        else:

            firstname, middlename, lastname, credential = generate_random_name(
                data_firstnames,
                data_surnames,
                already_used_names,
                already_used_names_more_frequent,
            )

            while (
                firstname.lower() in parsed_name
                or middlename.lower() in parsed_name
                or lastname.lower() in parsed_name
            ):
                firstname, middlename, lastname, credential = generate_random_name(
                    data_firstnames,
                    data_surnames,
                    already_used_names,
                    already_used_names_more_frequent,
                )

        if store_names is not None:
            store_names.append((firstname, middlename, lastname, credential))

        seed = random.random()
        for name_format in distribution_probabilities.keys():
            bounds = distribution_probabilities[name_format]
            if seed >= bounds[0] and seed < bounds[1]:
                if post_distribution_probabilities is not None:
                    post_distribution_probabilities[name_format] += 1
                if constraint is not None:
                    constraint.add_item((firstname, middlename, lastname, credential))
                credential = insert_dots(credential)
                return convert_name_to_string[name_format](
                    firstname, middlename, lastname, credential
                )

    # ## Generate patients

    def parse_patient_format(patient):
        """Parse the format of a patient which is a string
        Returns its format as a string
        """
        return "STANDARD"

    def count_patient_format_frequencies(patient_list):
        """Takes the list of all patients of the reports
        Returns the number of patients per patients format
        """
        frequencies_dict = {"STANDARD": 0}

        for patient in patient_list:
            frequencies_dict[parse_patient_format(patient)] += 1
        return frequencies_dict

    frequencies_dict_patient = count_patient_format_frequencies(
        phi_values_dict["PATIENT"]
    )

    distribution_probabilities_patient = generate_distribution_probabilities(
        frequencies_dict_patient
    )

    def convert_patient_to_string(firstname, middlename, lastname, honorific):
        random_number = random.random()

        if random_number > 0.95:
            firstname += " " + middlename[0]
            if random.random() > 0.6:
                firstname += "."
        elif random_number > 0.9:
            firstname += " " + middlename

        random_number = random.random()

        if random_number > 0.45:
            return "{} {}".format(firstname, lastname)
        elif random_number > 0.35:
            return "{} {}".format(lastname, firstname)
        elif random_number > 0.25:
            return "{} {} {}".format(honorific, lastname, firstname)
        elif random_number > 0.15:
            return "{} {} {}".format(honorific, firstname, lastname)
        elif random_number > 0.05:
            return "{} {}".format(honorific, lastname)
        else:
            return "{} {}".format(honorific, firstname)

    convert_patient_to_string = {"STANDARD": convert_patient_to_string}

    def insert_dots_patient(honorific):
        random_number = random.random()
        if random_number > 0.3:
            return honorific
        else:
            return honorific + "."

    already_used_patients = []
    already_used_patients_more_frequent = []

    def generate_random_patient(
        data_firstnames,
        data_surnames,
        already_used_patients,
        already_used_patients_more_frequent,
    ):
        """
        This function will return a random patient.
        """

        if len(already_used_patients) > 0 and random.random() > 0.9:
            if random.random() > 0.8 and len(already_used_patients_more_frequent):
                # some names are repeated a lot of times
                firstname, middlename, lastname, honorific = random.sample(
                    already_used_patients_more_frequent, 1
                )[0]
                return (firstname, middlename, lastname, honorific)
            else:
                firstname, middlename, lastname, honorific = random.sample(
                    already_used_patients, 1
                )[0]
                if random.random() > 1 - 1 / (
                    len(already_used_patients_more_frequent) + 1
                ) and (
                    (firstname, middlename, lastname, honorific)
                    not in already_used_patients_more_frequent
                ):
                    already_used_patients_more_frequent.append(
                        (firstname, middlename, lastname, honorific)
                    )
                return (firstname, middlename, lastname, honorific)

        # We will use predefined list of credentials. This is to avoid any leakage.
        # We can insert except for doctors some points between the letters
        frequent_honorifics = ["Mr", "Mrs", "Miss", "Ms"]
        rare_honorifics = ["Mx", "Sir", "Dr"]
        very_rare_honorifics = ["Lady", "Lord"]
        random_number = random.random()
        honorific = None

        if random_number > 0.01:
            honorific = random.sample(frequent_honorifics, 1)[0]
        elif random_number > 0.005:
            honorific = random.sample(rare_honorifics, 1)[0]
        else:
            honorific = random.sample(very_rare_honorifics, 1)[0]

        firstname = None
        middlename = None
        lastname = None

        if random.random() > 0.1:
            firstname = random.sample(data_firstnames[:1000], 1)[0]
        else:
            firstname = random.sample(data_firstnames, 1)[0]

        if random.random() > 0.1:
            middlename = random.sample(data_firstnames[:1000], 1)[0]
        else:
            middlename = random.sample(data_firstnames, 1)[0]

        if random.random() > 0.1:
            lastname = random.sample(data_surnames[:1000], 1)[0]
        else:
            lastname = random.sample(data_surnames, 1)[0]

        # we must also sometimes replicate a previous name (probability proportional to the size)

        already_used_patients.append((firstname, middlename, lastname, honorific))
        return firstname, middlename, lastname, honorific

    def are_patients_equal(old_patient, new_patient):
        honorifics = set(
            map(
                lambda x: x.lower(),
                ["Mr", "Mrs", "Miss", "Ms", "Mx", "Sir", "Dr", "Lady", "Lord"],
            )
        )

        old_patient_set = (
            set(re.sub(r"[^\w]", " ", old_patient, flags=re.DOTALL).lower().split())
            - honorifics
        )
        new_patient_set = (
            set(re.sub(r"[^\w]", " ", new_patient, flags=re.DOTALL).lower().split())
            - honorifics
        )

        return len(old_patient_set.intersection(new_patient_set)) > 0

    already_used_patients = []
    already_used_patients_more_frequent = []

    def generate_patient(
        patient,
        distribution_probabilities,
        convert_patient_to_string,
        data_firstnames,
        data_surnames,
        already_used_patients,
        already_used_patients_more_frequent,
        post_distribution_probabilities=None,
        constraint=None,
    ):

        parsed_patient = patient.lower().split()

        item = None
        firstname, middlename, lastname, honorific = None, None, None, None
        if constraint is not None:
            item = constraint.get_constraint()
        if constraint is not None and item is not None:
            firstname, middlename, lastname, honorific = item
        else:
            firstname, middlename, lastname, honorific = generate_random_patient(
                data_firstnames,
                data_surnames,
                already_used_patients,
                already_used_patients_more_frequent,
            )

            while (
                firstname.lower() in parsed_patient
                or middlename.lower() in parsed_patient
                or lastname.lower() in parsed_patient
            ):
                firstname, middlename, lastname, honorific = generate_random_patient(
                    data_firstnames,
                    data_surnames,
                    already_used_patients,
                    already_used_patients_more_frequent,
                )

        seed = random.random()
        for patient_format in distribution_probabilities.keys():
            bounds = distribution_probabilities[patient_format]
            if seed >= bounds[0] and seed < bounds[1]:
                if post_distribution_probabilities is not None:
                    post_distribution_probabilities[patient_format] += 1
                if constraint is not None:
                    constraint.add_item((firstname, middlename, lastname, honorific))
                honorific = insert_dots_patient(honorific)
                return convert_patient_to_string[patient_format](
                    firstname, middlename, lastname, honorific
                )

    # ## Generate hospitals

    try:
        data_hospitals = (
            pd.read_csv("./hospitals.txt", delimiter="\t", header=None)
            .rename(columns={0: "hospital"})
            .astype("str")
            .drop_duplicates()
            .dropna()
        )

    except UnicodeDecodeError:
        data_hospitals = (
            pd.read_csv(
                "./hospitals.txt",
                delimiter="\t",
                header=None,
                encoding="unicode_escape",
            )
            .rename(columns={0: "hospital"})
            .astype("str")
            .drop_duplicates()
            .dropna()
        )

    data_universities = (
        pd.read_csv("./universities.txt", delimiter="\t", header=None)
        .rename(columns={0: "university"})
        .astype("str")
        .drop_duplicates()
        .dropna()
    )

    data_universities.head()

    data_departments = pd.DataFrame(
        [
            "Anesthesiology",
            "Cardiology",
            "Clinical pathology",
            "Gastroenterology",
            "General surgery",
            "Geriatric",
            "Gynaecology",
            "Haematology",
            "Neurology",
            "Oncology",
            "Opthalmology",
            "Orthopaedic",
            "Pediatrics",
            "Pharmacy",
            "Psychiatry",
            "Radiology",
            "Urology",
        ]
    ).rename(columns={0: "department"})

    from collections import Counter

    # Counter(" ".join(data_hospitals["hospital"]).split()).most_common(100)
    data_frequent_tokens = (
        pd.read_csv("./frequent_hospitals_tokens.txt", delimiter="\t", header=None)
        .rename(columns={0: "token"})
        .astype("str")
        .drop_duplicates()
        .dropna()
    )

    data_hospitals = [
        x.lower().strip() for x in data_hospitals.values.squeeze().tolist()
    ]
    data_universities = [
        x.lower().strip() for x in data_universities.values.squeeze().tolist()
    ]
    data_departments = [
        x.lower().strip() for x in data_departments.values.squeeze().tolist()
    ]

    data_frequent_tokens = [
        x.lower().strip() for x in data_frequent_tokens.values.squeeze().tolist()
    ]

    data_frequent_tokens.append("state")
    data_frequent_tokens.append("college")

    assert not (len(list(filter(lambda x: (not isinstance(x, str)), data_hospitals))))
    assert not (
        len(list(filter(lambda x: (not isinstance(x, str)), data_frequent_tokens)))
    )
    assert not (
        len(list(filter(lambda x: (not isinstance(x, str)), data_universities)))
    )
    assert not (len(list(filter(lambda x: (not isinstance(x, str)), data_departments))))

    def parse_hospital_format(name):
        """Parse the format of a hospital which is a string
        Returns its format as a string
        """
        if name.strip().count(" ") > 0:
            return "long_form"
        else:
            return "short_form"

    def count_hospital_format_frequencies(hospital_list):
        """Takes the list of all hospitals of the reports
        Returns the number of hospitals per hospital format
        """
        frequencies_dict = {"long_form": 0, "short_form": 0}

        for hospital in hospital_list:
            frequencies_dict[parse_hospital_format(hospital)] += 1

        # manual enforcement
        frequencies_dict = {
            "long_form": 3
            if frequencies_dict["long_form"] < frequencies_dict["short_form"]
            else 4,
            "short_form": 4
            if frequencies_dict["long_form"] < frequencies_dict["short_form"]
            else 3,
        }
        return frequencies_dict

    frequencies_dict_hospital = count_hospital_format_frequencies(
        phi_values_dict["HOSPITAL"]
    )
    frequencies_dict_hospital

    distribution_probabilities_hospital = generate_distribution_probabilities(
        frequencies_dict_hospital
    )

    def convert_hospital_to_long(hospital):
        return hospital.title()

    def convert_hospital_to_short(hospital):
        hospital_tokens = None
        if random.random() > 0.2:
            hospital_tokens = re.sub(r"[^\w]", " ", hospital, flags=re.DOTALL).split()
        else:
            hospital_tokens = hospital.split()

        if (
            len(re.sub(r"[^\w]", " ", hospital, flags=re.DOTALL).split()) == 1
            or len(hospital.split()) == 1
        ):
            return hospital.title()

        if len("".join(e[0] for e in hospital_tokens).upper()) == 1:
            return hospital.title()

        return "".join(e[0] for e in hospital_tokens).upper()

    convert_hospital_to_string = {
        "long_form": convert_hospital_to_long,
        "short_form": convert_hospital_to_short,
    }

    def remove_hospital_tokens(hospital, data_frequent_tokens):
        random_number = random.random()
        if random_number > 0.3:
            return hospital.title()

        hospital_tokens = hospital.split()
        if set(hospital_tokens).intersection(set(data_frequent_tokens)) == set(
            hospital_tokens
        ):
            return hospital.title()
        elif random_number > 0.15:
            for token in data_frequent_tokens:
                if token in hospital_tokens:
                    hospital_tokens.remove(token)
            return " ".join(hospital_tokens).title()
        else:
            tokens_in_common = list(
                set(hospital_tokens).intersection(set(data_frequent_tokens))
            )
            if len(tokens_in_common) == 0:
                return hospital.title()
            number_tokens_selected = random.randint(1, len(tokens_in_common))
            random.shuffle(tokens_in_common)
            tokens_in_common = tokens_in_common[:number_tokens_selected]
            for token in tokens_in_common:
                hospital_tokens.remove(token)
            return " ".join(hospital_tokens).title()

    already_used_hospitals = []
    already_used_hospitals_more_frequent = []

    def generate_random_hospital(
        data_hospitals,
        data_frequent_tokens,
        already_used_hospitals,
        already_used_hospitals_more_frequent,
    ):
        """
        This function will return a random hospital.
        """

        if random.random() > (1 / (len(already_used_hospitals) + 1)):
            if random.random() > 0.2 and len(already_used_hospitals_more_frequent):
                # some names are repeated a lot of times
                hospital = random.sample(already_used_hospitals_more_frequent, 1)[0]
                return hospital
            else:
                hospital = random.sample(already_used_hospitals, 1)[0]
                if (
                    random.random()
                    > 1 - 1 / (10 ** len(already_used_hospitals_more_frequent))
                ) and (hospital not in already_used_hospitals_more_frequent):
                    already_used_hospitals_more_frequent.append(hospital)
                return hospital

        if random.random() > 0.4:
            hospital = random.sample(data_hospitals, 1)[0]
        else:
            hospital = random.sample(data_universities, 1)[0]

        # we must also sometimes replicate a previous name (probability proportional to the size)

        already_used_hospitals.append(hospital)
        return hospital

    from difflib import SequenceMatcher

    def add_department(hospital, data_departments, parsed_hospital):
        # hospital is the fake hospital
        # parsed_hospital is the true parsed hospital

        if random.random() > 0.2:
            return hospital.title()

        department = None
        number_iterations = 0

        while department is None or any(
            [
                SequenceMatcher(
                    None, parsed_hospital_token.lower(), department.lower()
                ).ratio()
                >= 0.7
                for parsed_hospital_token in parsed_hospital
            ]
        ):
            department = random.choice(data_departments)
            number_iterations += 1
            if number_iterations > 15:
                raise Exception("problem")

        if random.random() > 0.5:
            return (
                department
                + random.choice([" of", " at", "", ",", " -"])
                + " "
                + hospital
            ).title()

        return (
            hospital + random.choice([" of", "", ",", " -"]) + " " + department
        ).title()

    def reverse_frequent_tokens(hospital, data_frequent_tokens):
        if random.random() > 0.5:
            return hospital.title()

        hospital_split = hospital.split()
        start_index = 0
        current_tokens_are_frequent = False

        for i, token in enumerate(hospital_split):
            if token.lower() in data_frequent_tokens:
                if current_tokens_are_frequent:
                    continue
                else:
                    current_tokens_are_frequent = True
                    start_index = i

            else:
                if current_tokens_are_frequent:
                    current_tokens_are_frequent = False

        if not current_tokens_are_frequent:
            return hospital.title()

        hospital_split = (
            hospital_split[start_index:]
            + [random.choice(["of", "of", "at", ""])]
            + hospital_split[:start_index]
        )

        return " ".join(hospital_split).strip().replace("  ", " ").title()

    from difflib import SequenceMatcher

    already_used_hospitals = []
    already_used_hospitals_more_frequent = []

    def are_hospitals_equal(old_hospital, new_hospital):
        # If old_hospital is a plain word
        old_clean = set(
            re.sub(r"[^\w]", " ", old_hospital.lower(), flags=re.DOTALL).split()
        )
        new_clean = set(
            re.sub(r"[^\w]", " ", new_hospital.lower(), flags=re.DOTALL).split()
        )

        intersection_len = len(old_clean.intersection(new_clean))
        if (
            len(old_clean) > 0
            and len(new_clean) > 0
            and (
                intersection_len / len(old_clean) >= 0.66
                or intersection_len / len(new_clean) >= 0.66
            )
        ):
            return True

        # If old_hospital is an abreviation
        old_clean = re.sub(r"[^\w]", "", old_hospital.lower(), flags=re.DOTALL)
        new_clean = re.sub(
            r"[^\w]",
            "",
            convert_hospital_to_short(new_hospital).lower(),
            flags=re.DOTALL,
        )

        if SequenceMatcher(None, old_clean, new_clean).ratio() >= 0.66:
            return True

        old_clean = re.sub(
            r"[^\w]",
            "",
            convert_hospital_to_short(old_hospital).lower(),
            flags=re.DOTALL,
        )
        new_clean = re.sub(r"[^\w]", "", new_hospital.lower(), flags=re.DOTALL)

        if SequenceMatcher(None, old_clean, new_clean).ratio() >= 0.66:
            return True

        return False

    def generate_hospitals(
        old_hospital,
        distribution_probabilities,
        convert_hospital_to_string,
        data_hospitals,
        data_frequent_tokens,
        already_used_hospitals,
        already_used_hospitals_more_frequent,
        post_distribution_probabilities=None,
        constraint=None,
    ):

        parsed_hospital = old_hospital.lower().split()

        item = None
        hospital = None
        if constraint is not None:
            item = constraint.get_constraint()
        if constraint is not None and item is not None:
            hospital = item
        else:

            hospital = generate_random_hospital(
                data_hospitals,
                data_frequent_tokens,
                already_used_hospitals,
                already_used_hospitals_more_frequent,
            )

            count = 0
            while are_hospitals_equal(old_hospital, hospital):
                count += 1
                if count > 50:
                    raise Exception("problem hospital")
                hospital = generate_random_hospital(
                    data_hospitals,
                    data_frequent_tokens,
                    already_used_hospitals,
                    already_used_hospitals_more_frequent,
                )

        seed = random.random()
        for hospital_format in distribution_probabilities.keys():
            bounds = distribution_probabilities[hospital_format]
            if seed >= bounds[0] and seed < bounds[1]:
                if post_distribution_probabilities is not None:
                    post_distribution_probabilities[hospital_format] += 1
                if constraint is not None:
                    constraint.add_item(hospital)
                hospital = remove_hospital_tokens(hospital, data_frequent_tokens)
                hospital = reverse_frequent_tokens(hospital, data_frequent_tokens)

                return add_department(
                    convert_hospital_to_string[hospital_format](hospital),
                    data_departments,
                    parsed_hospital,
                )

    # ## Generate vendors

    import json

    with open("./companies.txt", "r") as fp:
        data_vendors = json.load(fp)

    assert not len(list(filter(lambda x: (not isinstance(x, str)), data_vendors)))

    def convert_vendor_to_short_without_numbers(vendor):
        return convert_vendor_to_short(re.sub(r"[\d]", " ", vendor, flags=re.DOTALL))

    def convert_vendor_to_short_without_symbols(vendor):
        return convert_vendor_to_short(re.sub(r"[^\w]", " ", vendor, flags=re.DOTALL))

    def convert_vendor_to_short_without_both(vendor):
        return convert_vendor_to_short(
            re.sub(r"[^\w]|[\d]", " ", vendor, flags=re.DOTALL)
        )

    def convert_vendor_to_short(vendor):
        vendor_tokens = None
        vendor_tokens = vendor.split()

        if len(vendor_tokens) == 1:
            return vendor.title()
        return "".join(e[0] for e in vendor_tokens).upper()

    already_used_vendors = []
    already_used_vendors_more_frequent = []

    def generate_random_vendor(
        data_vendors, already_used_vendors, already_used_vendors_more_frequent
    ):
        """
        This function will return a random vendor.
        """

        if random.random() > (1 / (len(already_used_vendors) + 1)):
            if random.random() > 0.2 and len(already_used_vendors_more_frequent):
                # some names are repeated a lot of times
                vendor = random.sample(already_used_vendors_more_frequent, 1)[0]
                return vendor
            else:
                vendor = random.sample(already_used_vendors, 1)[0]
                if (
                    random.random()
                    > 1 - 1 / (10 ** len(already_used_vendors_more_frequent))
                ) and (vendor not in already_used_vendors_more_frequent):
                    already_used_vendors_more_frequent.append(vendor)
                return vendor

        vendor = random.sample(data_vendors, 1)[0]

        # we must also sometimes replicate a previous name (probability proportional to the size)

        already_used_vendors.append(vendor)
        return vendor

    from difflib import SequenceMatcher

    already_used_vendors = []
    already_used_vendors_more_frequent = []

    def are_vendors_equal(old_vendor, new_vendor):
        # If old_hospital is a plain word
        old_clean = set(
            re.sub(r"[^\w]", " ", old_vendor.lower(), flags=re.DOTALL).split()
        )
        new_clean = set(
            re.sub(r"[^\w]", " ", new_vendor.lower(), flags=re.DOTALL).split()
        )

        intersection_len = len(old_clean.intersection(new_clean))
        if (
            len(old_clean) > 0
            and len(new_clean) > 0
            and (
                intersection_len / len(old_clean) >= 0.66
                or intersection_len / len(new_clean) >= 0.66
            )
        ):
            return True

        # If old_hospital is an abreviation
        old_clean = old_vendor.lower()
        new_clean1 = convert_vendor_to_short(new_vendor).lower()
        new_clean2 = convert_vendor_to_short_without_numbers(new_vendor).lower()
        new_clean3 = convert_vendor_to_short_without_symbols(new_vendor).lower()
        new_clean4 = convert_vendor_to_short_without_both(new_vendor).lower()

        if SequenceMatcher(None, old_clean, new_clean1).ratio() >= 0.66:
            return True
        if SequenceMatcher(None, old_clean, new_clean2).ratio() >= 0.66:
            return True
        if SequenceMatcher(None, old_clean, new_clean3).ratio() >= 0.66:
            return True
        if SequenceMatcher(None, old_clean, new_clean4).ratio() >= 0.66:
            return True

        new_clean = new_vendor.lower()
        old_clean1 = convert_vendor_to_short(old_vendor).lower()
        old_clean2 = convert_vendor_to_short_without_numbers(old_vendor).lower()
        old_clean3 = convert_vendor_to_short_without_symbols(old_vendor).lower()
        old_clean4 = convert_vendor_to_short_without_both(old_vendor).lower()

        if SequenceMatcher(None, new_clean, old_clean1).ratio() >= 0.66:
            return True
        if SequenceMatcher(None, new_clean, old_clean2).ratio() >= 0.66:
            return True
        if SequenceMatcher(None, new_clean, old_clean3).ratio() >= 0.66:
            return True
        if SequenceMatcher(None, new_clean, old_clean4).ratio() >= 0.66:
            return True

        return False

    def generate_vendor(
        old_vendor,
        data_vendors,
        already_used_vendors,
        already_used_vendors_more_frequent,
        post_distribution_probabilities=None,
        constraint=None,
    ):

        item = None
        vendor = None
        if constraint is not None:
            item = constraint.get_constraint()
        if constraint is not None and item is not None:
            vendor = item
        else:

            vendor = generate_random_vendor(
                data_vendors, already_used_vendors, already_used_vendors_more_frequent
            )

            count = 0
            while are_vendors_equal(old_vendor, vendor):
                count += 1
                if count > 50:
                    raise Exception("problem vendor")
                vendor = generate_random_vendor(
                    data_vendors,
                    already_used_vendors,
                    already_used_vendors_more_frequent,
                )
        if constraint is not None:
            constraint.add_item(vendor)
        return vendor

    # ## Unique

    import string

    already_used_unique = []
    already_used_unique_more_frequent = []

    def format_unique(unique):
        formatted_unique = unique
        random_seed = random.random()
        correct_prob = 1
        correct_prob_2 = 1

        if random_seed > 0.5:
            formatted_unique = formatted_unique.upper()
        elif random_seed > 0.35:
            formatted_unique = formatted_unique.lower()
            correct_prob_2 = 0.7
        else:
            correct_prob = 1.5

        if random.random() > 0.8:
            interval = abs(int(np.random.normal(3, 1)))
            if interval <= 0:
                interval = 1
            character = None
            seed_character = random.random()
            if seed_character > 0.3 * correct_prob:
                character = "-"
            elif seed_character > 0.2 * correct_prob:
                character = " "
            elif seed_character > 0.1 * correct_prob:
                character = "."
            else:
                character = "_"
            formatted_unique = character.join(
                re.findall(
                    r".{1," + str(interval) + "}", formatted_unique, flags=re.DOTALL
                )
            )

        if random.random() > 0.9 * correct_prob_2:
            formatted_unique = "#" + formatted_unique

        return formatted_unique

    def generate_random_unique(already_used_unique, already_used_unique_more_frequent):
        """
        This function will return a random unique.
        """

        if random.random() > 0.7 and len(already_used_unique):
            if random.random() > 0.5 and len(already_used_unique_more_frequent):
                # some names are repeated a lot of times
                unique = random.sample(already_used_unique_more_frequent, 1)[0]
                return unique
            else:
                unique = random.sample(already_used_unique, 1)[0]
                if (
                    random.random()
                    > 1 - 1 / (10 ** len(already_used_unique_more_frequent))
                ) and (unique not in already_used_unique_more_frequent):
                    already_used_unique_more_frequent.append(unique)
                return unique

        unique_length = abs(int(np.random.normal(9, 2.5)))
        if unique_length <= 0:
            unique_length = 1

        number_letters = 0
        random_seed = random.random()
        if random_seed > 0.85:
            number_letters = random.randint(1, unique_length)
        elif random_seed > 0.7:
            number_letters = unique_length

        number_numbers = unique_length - number_letters

        letters = [random.choice(string.ascii_letters) for i in range(number_letters)]
        numbers = [random.randint(0, 9) for i in range(number_numbers)]

        unique = letters + numbers
        random.shuffle(unique)

        unique = "".join(map(str, unique))

        # we must also sometimes replicate a previous name (probability proportional to the size)
        already_used_unique.append(unique)

        return unique

    from difflib import SequenceMatcher

    already_used_unique = []
    already_used_unique_more_frequent = []

    def are_uniques_equal(old_unique, new_unique):

        # RETIRER LES CHARACTERES EVENTUELLEMENT
        # If old_hospital is a plain word
        old_clean = re.sub(r"[^\w]", "", old_unique.lower(), flags=re.DOTALL)
        new_clean = re.sub(r"[^\w]", "", new_unique.lower(), flags=re.DOTALL)

        if SequenceMatcher(None, old_clean, new_clean).ratio() >= 0.66:
            return True

        return False

    def generate_unique(
        old_unique,
        already_used_unique,
        already_used_unique_more_frequent,
        post_distribution_probabilities=None,
        constraint=None,
    ):

        item = None
        unique = None
        if constraint is not None:
            item = constraint.get_constraint()
        if constraint is not None and item is not None:
            unique = item
        else:

            unique = generate_random_unique(
                already_used_unique, already_used_unique_more_frequent
            )

            count = 0
            while are_uniques_equal(old_unique, unique):
                count += 1
                if count > 50:
                    print(" problem unique")
                unique = generate_random_unique(
                    already_used_unique, already_used_unique_more_frequent
                )
        if constraint is not None:
            constraint.add_item(unique)
        unique = format_unique(unique)
        return unique

    # ## Phone

    import string

    already_used_phone = []
    already_used_phone_more_frequent = []

    def format_phone(phone):
        formatted_phone = phone

        if random.random() > 0.05:
            formatted_phone = "(" + formatted_phone[:3] + ")" + formatted_phone[3:]

        if random.random() > 0.05:
            if formatted_phone[0] == "(":
                if random.random() > 0.05:
                    formatted_phone = formatted_phone[:5] + " " + formatted_phone[5:]
                else:
                    formatted_phone = formatted_phone[:5] + "-" + formatted_phone[5:]
            else:
                if random.random() > 0.05:
                    formatted_phone = formatted_phone[:3] + " " + formatted_phone[3:]
                else:
                    formatted_phone = formatted_phone[:3] + "-" + formatted_phone[3:]

        if random.random() > 0.05:
            formatted_phone = formatted_phone[:-4] + "-" + formatted_phone[-4:]

        if random.random() > 0.99:
            formatted_phone = "1" + formatted_phone

            if random.random() > 0.5:
                formatted_phone = formatted_phone[0] + " " + formatted_phone[1:]

            if random.random() > 0.5:
                formatted_phone = "+" + formatted_phone

        return formatted_phone

    def generate_random_phone(already_used_phone, already_used_phone_more_frequent):
        """
        This function will return a random phone.
        """

        if random.random() > 0.7 and len(already_used_phone):
            if random.random() > 0.5 and len(already_used_phone_more_frequent):
                # some names are repeated a lot of times
                phone = random.sample(already_used_phone_more_frequent, 1)[0]
                return phone
            else:
                phone = random.sample(already_used_phone, 1)[0]
                if (
                    random.random()
                    > 1 - 1 / (10 ** len(already_used_phone_more_frequent))
                ) and (phone not in already_used_phone_more_frequent):
                    already_used_phone_more_frequent.append(phone)
                return phone

        phone_length = 10

        numbers = [random.randint(0, 9) for i in range(phone_length)]
        numbers[0] = random.randint(1, 9)

        phone = "".join(map(str, numbers))

        # we must also sometimes replicate a previous name (probability proportional to the size)
        already_used_phone.append(phone)

        return phone

    from difflib import SequenceMatcher

    already_used_phone = []
    already_used_phone_more_frequent = []

    def are_phones_equal(old_phone, new_phone):

        # RETIRER LES CHARACTERES EVENTUELLEMENT
        # If old_hospital is a plain word
        old_clean = re.sub(r"[^\w]", "", old_phone.lower(), flags=re.DOTALL)
        new_clean = re.sub(r"[^\w]", "", new_phone.lower(), flags=re.DOTALL)

        if SequenceMatcher(None, old_clean, new_clean).ratio() >= 0.66:
            return True

        return False

    def generate_phone(
        old_phone,
        already_used_phone,
        already_used_phone_more_frequent,
        post_distribution_probabilities=None,
        constraint=None,
    ):
        item = None
        phone = None
        if constraint is not None:
            item = constraint.get_constraint()
        if constraint is not None and item is not None:
            phone = item
        else:

            phone = generate_random_phone(
                already_used_phone, already_used_phone_more_frequent
            )

            count = 0
            while are_phones_equal(old_phone, phone):
                count += 1
                if count > 50:
                    raise Exception(" problem phone")
                phone = generate_random_phone(
                    already_used_phone, already_used_phone_more_frequent
                )
        if constraint is not None:
            constraint.add_item(phone)
        phone = format_phone(phone)
        return phone

    # ## Age

    import string

    already_used_age = []
    already_used_age_more_frequent = []

    def format_age(age):
        formatted_age = age

        return str(formatted_age)

    def generate_random_age(already_used_age, already_used_age_more_frequent):
        """
        This function will return a random age.
        """
        if random.random() > 0.05:
            age = random.randint(90, 100)
        else:
            age = random.randint(101, 120)
        already_used_age.append(age)

        return age

    from difflib import SequenceMatcher

    already_used_age = []
    already_used_age_more_frequent = []

    def are_ages_equal(old_age, new_age):

        old_age_without_letters = "".join([x for x in list(old_age) if x.isnumeric()])

        if int(old_age_without_letters if old_age_without_letters else 90) == int(
            new_age
        ):
            return True

        return False

    def generate_age(
        old_age,
        already_used_age,
        already_used_age_more_frequent,
        post_distribution_probabilities=None,
        constraint=None,
    ):
        item = None
        age = None
        if constraint is not None:
            item = constraint.get_constraint()
        if constraint is not None and item is not None:
            age = item
        else:

            age = generate_random_age(already_used_age, already_used_age_more_frequent)

            count = 0
            while are_ages_equal(old_age, age):
                count += 1
                if count > 50:
                    raise Exception(" problem phone")
                age = generate_random_age(
                    already_used_age, already_used_age_more_frequent
                )
        if constraint is not None:
            constraint.add_item(age)
        age = format_age(age)
        return age

    # ## Compiling everything: Hide in plain sight

    symbol_list = (
        string.printable[62:88] + string.printable[90:94] + string.printable[95:]
    )
    letter_and_number_list = string.printable[:62]

    def generate_typing_errors(input_str, error_prob=1 / 500, small_error_prob=1 / 100):
        # Removing a character, in particular spaces or symbols
        # Remove spaces
        # Lower case everything, upper case everything, title, or lower/uppercase random letters
        # Insert randomly letters or spaces or symbols (very rare for letters or symbols)
        # To do not perturb the algorithms, theseerrors (except the case) are very rare. 1/300 ? yes that's good

        output_str = input_str

        # Randomly change case
        if random.random() > 1 - small_error_prob:
            case_type_seed = random.random()

            if case_type_seed > 0.5:
                output_str = output_str.title()

            elif case_type_seed > 0.3:
                output_str = output_str.lower()

            elif case_type_seed > 0.1:
                output_str = output_str.upper()

            else:
                # invert the case of random letters
                output_str_len = len(output_str)
                output_str_alpha_len = len(
                    [
                        letter_index
                        for letter_index in range(output_str_len)
                        if output_str[letter_index].isalpha()
                    ]
                )
                number_errors = min(
                    abs(int(np.random.normal(0, 1.7))), output_str_alpha_len
                )

                if number_errors == 0:
                    number_errors = min(1, output_str_alpha_len)
                already_corrected = []

                for i in range(number_errors):
                    letter_index = random.randint(0, output_str_len - 1)
                    count = 0
                    while letter_index in already_corrected or (
                        not output_str[letter_index].isalpha()
                    ):
                        count += 1
                        if count > 50:
                            raise Exception(" problem typing")
                            print(number_errors)
                            print(output_str)
                        letter_index = random.randint(0, output_str_len - 1)

                    already_corrected.append(letter_index)
                    if output_str[letter_index].isupper():
                        output_str = (
                            output_str[:letter_index]
                            + output_str[letter_index].lower()
                            + output_str[letter_index + 1 :]
                        )
                    else:
                        output_str = (
                            output_str[:letter_index]
                            + output_str[letter_index].upper()
                            + output_str[letter_index + 1 :]
                        )

        # Randomly insert spaces
        if random.random() > 1 - small_error_prob:
            output_str_len = len(output_str)
            number_spaces = abs(int(np.random.normal(0, 1.7)))
            if number_spaces == 0:
                number_spaces = min(1, output_str_len)
            index_spaces_in_the_middle = [
                i for i, x in enumerate(output_str) if x == " "
            ]

            for i in range(number_spaces):

                if random.random() > 0.5 and len(index_spaces_in_the_middle) > 0:
                    # in the middle
                    index_spaces_in_the_middle = [
                        i for i, x in enumerate(output_str) if x == " "
                    ]
                    letter_index = random.choice(index_spaces_in_the_middle)
                    output_str = (
                        output_str[:letter_index] + " " + output_str[letter_index:]
                    )

                else:
                    if random.random() > 0.5:
                        # At the beginning
                        output_str = " " + output_str
                    else:
                        output_str = output_str + " "

        if random.random() > 1 - error_prob:
            number_errors = abs(int(np.random.normal(0, 1.7)))
            if number_errors == 0:
                number_errors = 1

            # print('number errors', number_errors)
            for i in range(number_errors):
                error_type_seed = random.random()
                index_spaces_or_symbols = [
                    i
                    for i, x in enumerate(output_str)
                    if not x.isalpha() and not x.isnumeric()
                ]
                output_str_len = len(output_str)

                # Remove space or symbol
                if error_type_seed > 0.5 and len(index_spaces_or_symbols) > 0:
                    # print('remove space or symbol')
                    letter_index = random.choice(index_spaces_or_symbols)
                    output_str = (
                        output_str[:letter_index] + output_str[letter_index + 1 :]
                    )

                # Randomly insert symbol
                elif error_type_seed > 0.3:
                    # print('insert symbol')
                    letter_index = random.randint(0, output_str_len)  # not - 1
                    symbol = random.choice(symbol_list)
                    output_str = (
                        output_str[:letter_index] + symbol + output_str[letter_index:]
                    )

                # Remove other type of character
                elif (
                    error_type_seed > 0.15
                    and len(index_spaces_or_symbols) < output_str_len
                ):
                    # print('remove letter or number')
                    letter_index = random.randint(0, output_str_len - 1)
                    count = 0
                    while letter_index in index_spaces_or_symbols:
                        count += 1
                        if count > 50:
                            print(" problem typing 2")
                        letter_index = random.randint(
                            0, output_str_len - 1
                        )  # average case is faster this way
                    output_str = (
                        output_str[:letter_index] + output_str[letter_index + 1 :]
                    )

                # Randomly insert other type of character
                else:
                    # print('insert letter or number')
                    letter_index = random.randint(0, output_str_len)  # not - 1
                    letter_or_number = random.choice(letter_and_number_list)
                    output_str = (
                        output_str[:letter_index]
                        + letter_or_number
                        + output_str[letter_index:]
                    )

        # Handle the case
        if random.random() < 0.4:
            if random.random() > 0.5:
                output_str = output_str.upper()
            else:
                output_str = output_str.lower()

        return output_str

    def memorize_phi_lengths(fake_phi, length_memory, phi_memory):
        split_fake_phi = re.split(r"(\W)", fake_phi, flags=re.DOTALL)
        split_fake_phi = [x for x in split_fake_phi if x != " " and x != ""]
        length_memory.append(len(fake_phi))
        phi_memory.append(fake_phi)
        return fake_phi

    # while loop with distinct that the input to spot the errors

    already_used_vendors = []
    already_used_vendors_more_frequent = []
    already_used_names = []
    already_used_names_more_frequent = []
    already_used_patients = []
    already_used_patients_more_frequent = []
    already_used_hospitals = []
    already_used_hospitals_more_frequent = []
    already_used_unique = []
    already_used_unique_more_frequent = []
    already_used_phone = []
    already_used_phone_more_frequent = []
    already_used_age = []
    already_used_age_more_frequent = []

    dict_generate_phi = {
        "VENDOR": (
            lambda vendor, constraint: generate_vendor(
                vendor,
                data_vendors,
                already_used_vendors,
                already_used_vendors_more_frequent,
                post_distribution_probabilities=None,
                constraint=constraint,
            )
        ),
        "DATES": (
            lambda date, constraint: generate_date(
                date,
                distribution_probabilities_date,
                convert_date_to_string,
                convert_string_to_date,
                min_date,
                max_date,
                store_dates=None,
                constraint=constraint,
            )
        ),
        "HCW": (
            lambda hcw, constraint: generate_name(
                hcw,
                distribution_probabilities_name,
                convert_name_to_string,
                data_firstnames,
                data_surnames,
                already_used_names,
                already_used_names_more_frequent,
                constraint=constraint,
            )
        ),
        "PATIENT": (
            lambda patient, constraint: generate_patient(
                patient,
                distribution_probabilities_patient,
                convert_patient_to_string,
                data_firstnames,
                data_surnames,
                already_used_patients,
                already_used_patients_more_frequent,
                constraint=constraint,
            )
        ),
        "HOSPITAL": (
            lambda hospital, constraint: generate_hospitals(
                hospital,
                distribution_probabilities_hospital,
                convert_hospital_to_string,
                data_hospitals,
                data_frequent_tokens,
                already_used_hospitals,
                already_used_hospitals_more_frequent,
                constraint=constraint,
            )
        ),
        "UNIQUE": (
            lambda unique, constraint: generate_unique(
                unique,
                already_used_unique,
                already_used_unique_more_frequent,
                post_distribution_probabilities=None,
                constraint=constraint,
            )
        ),
        "PHONE": (
            lambda phone, constraint: generate_phone(
                phone,
                already_used_phone,
                already_used_phone_more_frequent,
                post_distribution_probabilities=None,
                constraint=constraint,
            )
        ),
        "AGE": (
            lambda age, constraint: generate_age(
                age,
                already_used_age,
                already_used_age_more_frequent,
                post_distribution_probabilities=None,
                constraint=constraint,
            )
        ),
    }

    dict_generate_constraint = {
        "VENDOR": (
            lambda report: generate_equality_constraint(
                report, are_vendors_equal, "VENDOR"
            )
        ),
        "DATES": (
            lambda report: generate_date_constraint(report, convert_string_to_date)
        ),
        "HCW": (
            lambda report: generate_equality_constraint(report, are_names_equal, "HCW")
        ),
        "PATIENT": (
            lambda report: generate_equality_constraint(
                report, are_patients_equal, "PATIENT"
            )
        ),
        "HOSPITAL": (
            lambda report: generate_equality_constraint(
                report, are_hospitals_equal, "HOSPITAL"
            )
        ),
        "UNIQUE": (
            lambda report: generate_equality_constraint(
                report, are_uniques_equal, "UNIQUE"
            )
        ),
        "PHONE": (
            lambda report: generate_equality_constraint(
                report, are_phones_equal, "PHONE"
            )
        ),
        "AGE": (lambda report: None),
    }

    def generate_dict_contraints(report, dict_generate_constraint):
        dict_constraints = {}
        for phi in dict_generate_constraint.keys():
            dict_constraints[phi] = dict_generate_constraint[phi](report)
        return dict_constraints

    def get_phi_order(report):
        phi_report_list = re.findall(r"\\.+?\[\[.+?\]\]", report, flags=re.DOTALL)
        phi_report_list = [
            re.sub(
                r"\\",
                "",
                re.sub(r"\[\[.+?\]\]", "", x, flags=re.DOTALL),
                flags=re.DOTALL,
            )
            for x in phi_report_list
        ]
        return phi_report_list

    # intialize all the already used !!!
    # count = 0
    phi_lengths = []

    def generate_deidentified_report(
        report,
        dict_generate_phi,
        dict_generate_constraint,
        error_prob=1 / 500,
        small_error_prob=1 / 100,
    ):
        # global count
        # report is a labeled report with PHI labels
        deidentified_report = report
        phi_list = list(dict_generate_phi.keys())
        dict_constraints = generate_dict_contraints(report, dict_generate_constraint)
        # print(count)

        # count += 1
        phi_order_list = get_phi_order(report)

        length_memory_dict = {}
        length_memory_dict_mem = {}

        for phi in phi_list:
            length_memory = []
            phi_memory = []

            deidentified_report = re.sub(
                r"\\" + phi + "\[\[.+?\]\]",
                lambda match: memorize_phi_lengths(
                    generate_typing_errors(
                        dict_generate_phi[phi](
                            re.sub(
                                r"\]\]",
                                "",
                                re.sub(
                                    r"\\.+?\[\[", "", match.group(0), flags=re.DOTALL
                                ),
                                flags=re.DOTALL,
                            ),
                            dict_constraints[phi],
                        ),
                        error_prob,
                        small_error_prob,
                    ),
                    length_memory,
                    phi_memory,
                ),
                deidentified_report,
                flags=re.DOTALL,
            )

            # if phi == 'UNIQUE' and phi_memory != []:
            #    print('######')
            #    print(length_memory)
            #    print(phi_memory)
            #    print('----')

            for i, x in enumerate(
                re.findall(r"\\" + phi + "\[\[.+?\]\].", report, flags=re.DOTALL)
            ):
                phi_memory[i] += re.sub(r"\\.+?\[\[.+?\]\]", "", x, flags=re.DOTALL)

            # CORRECT ERROR WITH DR.
            # if random.random() > 0.05 and phi == "HCW":
            #    if (
            #        re.findall(r"(?i)dr\.\.", deidentified_report, flags=re.DOTALL)
            #        != []
            #    ):
            #        pass
            #        # print('dr.. SPOTTED')
            #    deidentified_report = re.sub(
            #        r"(?i)dr\.\.",
            #        lambda match: match.group(0)[:-1],
            #        deidentified_report,
            #        flags=re.DOTALL,
            #    )
            #    for i, x in enumerate(phi_memory):
            #        if "dr.." in x.lower():
            #            # print('dr.. REPLACED')
            #            # print(count - 1)
            #            # print(deidentified_report)
            #            length_memory[i] -= 1

            length_memory_dict[phi] = length_memory
            length_memory_dict_mem[phi] = length_memory[:]

        length_memory = []
        for x in phi_order_list:
            length_memory.append(length_memory_dict[x][0])
            length_memory_dict[x].pop(0)

        phi_lengths.append(length_memory)

        # if length_memory_dict_mem['UNIQUE'] != []:
        #    print(phi_lengths[-1])
        #    print(phi_order_list)
        #    print(already_used_unique[-phi_order_list.count('UNIQUE'):])
        #    print(deidentified_report)

        return deidentified_report

    deidentified_reports = []
    for index_, labeled_report in enumerate(labeled_reports):
        # print(index_)
        deidentified_reports.append(
            generate_deidentified_report(
                labeled_report, dict_generate_phi, dict_generate_constraint
            )
        )

    assert len(deidentified_reports) == len(labeled_reports)

    # for deidentified_report in deidentified_reports:
    #    assert "\\" not in deidentified_report
    #    assert "[[" not in deidentified_report
    #    assert "]]" not in deidentified_report

    for report_index in range(len(labeled_reports)):
        labeled_report = labeled_reports[report_index]
        deidentified_report = deidentified_reports[report_index]
        phi_length = phi_lengths[report_index]

        labeled_report_index = 0
        deidentified_report_index = 0
        phi_length_index = 0

        while labeled_report_index < len(labeled_report):
            if labeled_report[labeled_report_index] != "\\":
                try:
                    assert (
                        labeled_report[labeled_report_index]
                        == deidentified_report[deidentified_report_index]
                    )
                except:
                    print(labeled_report)
                    print(deidentified_report)
                    print(phi_length)
                    raise Exception("problem here")
                labeled_report_index += 1
                deidentified_report_index += 1

            else:
                while (
                    labeled_report[labeled_report_index : labeled_report_index + 2]
                    != "[["
                ):
                    labeled_report_index += 1
                labeled_report_index += 1
                labeled_report_index += 1
                labeled_phi = ""
                while (
                    labeled_report[labeled_report_index : labeled_report_index + 2]
                    != "]]"
                ):
                    labeled_phi += labeled_report[labeled_report_index]
                    labeled_report_index += 1
                labeled_report_index += 1
                labeled_report_index += 1

                deidentified_phi = deidentified_report[
                    deidentified_report_index : deidentified_report_index
                    + phi_length[phi_length_index]
                ]
                try:
                    assert labeled_phi != deidentified_phi
                except:
                    print(labeled_phi, "####", deidentified_phi)
                    print(
                        "##### PROBLEM HERE ? ########## PROBLEM HERE ? ########## PROBLEM HERE ? ########## PROBLEM HERE ? ########## PROBLEM HERE ? ########## PROBLEM HERE ? ########## PROBLEM HERE ? #####"
                    )
                    # raise Exception("Problem here")

                deidentified_report_index += phi_length[phi_length_index]
                phi_length_index += 1

        assert deidentified_report_index == len(deidentified_report)
        assert phi_length_index == len(phi_length)

    with open("deidentified_reports" + file_seed + ".npy", "wb") as f:
        np.save(f, np.array(deidentified_reports).astype("object"), allow_pickle=True)

    with open("phi_lengths" + file_seed + ".npy", "wb") as f:
        np.save(f, np.array(phi_lengths, dtype=object), allow_pickle=True)
