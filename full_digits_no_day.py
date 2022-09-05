import random
import datetime
from utils import suffix


def convert_date_to_string_full_digits_no_day(date, allow_weird_date_format=True):
    if random.random() > 0.40:
        # Year at the end
        temp_date = None
        random_seed = random.random()

        if random_seed > 0.51:
            # With month, year in 4 digits
            temp_date = date.strftime("%m/%Y")

        elif random_seed > 0.34:
            # With month, year in 2 digits
            temp_date = date.strftime("%m/%y")

        elif random_seed > 0.17:
            # With day, year in 4 digits
            temp_date = date.strftime("%d/%Y")

        else:
            # With day, year in 2 digits
            temp_date = date.strftime("%d/%y")

        if temp_date[0] == "0" and random.random() > 0.5:
            # Without trailing zero
            return temp_date[1:]

        else:
            # With trailing zero
            return temp_date

    else:
        # Year at the beginning
        temp_date = None
        random_seed = random.random()

        if random_seed > 0.75:
            # With month, year in 4 digits
            temp_date = date.strftime("%Y/%m")

        elif random_seed > 0.5:
            # With month, year in 2 digits
            temp_date = date.strftime("%y/%m")

        elif random_seed > 0.25:
            # With day, year in 4 digits
            temp_date = date.strftime("%Y/%d")

        else:
            # With day, year in 2 digits
            temp_date = date.strftime("%y/%d")

        if temp_date[-2] == "0" and random.random() > 0.5:
            # Without trailing zero
            return temp_date[:-2] + temp_date[-1]

        else:
            # With trailing zero
            return temp_date


def convert_string_to_date_full_digits_no_day(date):
    formats_to_try = ["%m/%Y", "%m/%y", "%d/%Y", "%d/%y"]

    for format_to_try in formats_to_try:
        try:
            return datetime.datetime.strptime(date, format_to_try)

        except:
            continue

    raise Exception("failed to parse")
