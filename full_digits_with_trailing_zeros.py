import random
import datetime
from utils import suffix


def convert_date_to_string_full_digits_with_trailing_zeros(
    date, allow_weird_date_format=True
):
    converted_date = None
    date_format_seed = random.random()

    if date_format_seed > 0.66:
        # Year at the end, 4 digits

        if random.random() > 0.125:
            # Month before day (US)
            converted_date = date.strftime("%m/%d/%Y")

        else:
            # Day before month (EU)
            converted_date = date.strftime("%d/%m/%Y")

    elif date_format_seed > 0.5:
        # Year at the beginning, 4 digits

        if random.random() > 0.25:
            # Month before day (US)
            converted_date = date.strftime("%Y/%m/%d")

        else:
            # Day before month (EU)
            converted_date = date.strftime("%Y/%d/%m")

    elif date_format_seed > 0.34:
        # Year at the beginning, 2 digits

        if random.random() > 0.25:
            # Month before day (US)
            converted_date = date.strftime("%y/%m/%d")

        else:
            # Day before month (EU)
            converted_date = date.strftime("%y/%d/%m")

    else:
        # Year at the end, 2 digits

        if random.random() > 0.125:
            # Month before day (US)
            converted_date = date.strftime("%m/%d/%y")

        else:
            # Day before month (EU)
            converted_date = date.strftime("%d/%m/%y")

    return converted_date


def convert_string_to_date_full_digits_with_trailing_zeros(date):
    formats_to_try = ["%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%d/%m/%y"]

    for format_to_try in formats_to_try:
        try:
            return datetime.datetime.strptime(date, format_to_try)

        except:
            continue

    raise Exception("failed to parse")
