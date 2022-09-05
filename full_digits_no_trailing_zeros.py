import random
import datetime
from utils import suffix


def convert_date_to_string_full_digits_no_trailing_zeros(
    date, allow_weird_date_format=True
):
    temp_date = None
    date_format_seed = random.random()

    if date_format_seed > 0.22:
        # Year at the end, with 4 digits

        if random.random() > 0.002:
            # Month before day (US)
            temp_date = date.strftime("%m/%d/%Y")

        else:
            # Day before month (EU)
            temp_date = date.strftime("%d/%m/%Y")

        converted_date = ""
        if temp_date[0] == "0":
            converted_date += temp_date[1:3]

        else:
            converted_date += temp_date[0:3]

        if temp_date[3] == "0":
            converted_date += temp_date[4:]

        else:
            converted_date += temp_date[3:]

    elif date_format_seed > 0.21:
        # Year the beginning, with 4 digits

        if random.random() > 0.15:
            # Month before day (US)
            temp_date = date.strftime("%Y/%m/%d")

        else:
            # Day before month (EU)
            temp_date = date.strftime("%Y/%d/%m")

        converted_date = temp_date[:5]
        if temp_date[5] == "0":
            converted_date += temp_date[6:8]

        else:
            converted_date += temp_date[5:8]

        if temp_date[8] == "0":
            converted_date += temp_date[9:]

        else:
            converted_date += temp_date[8:]

    elif date_format_seed > 0.2:
        # Year at the beginning, with 2 digits

        if random.random() > 0.15:
            # Month before day (US)
            temp_date = date.strftime("%y/%m/%d")

        else:
            # Day before month (EU)
            temp_date = date.strftime("%y/%d/%m")

        converted_date = temp_date[:3]
        if temp_date[3] == "0":
            converted_date += temp_date[4:6]

        else:
            converted_date += temp_date[3:6]

        if temp_date[6] == "0":
            converted_date += temp_date[7:]

        else:
            converted_date += temp_date[6:]

    else:
        # Year at the end, with two digits

        if random.random() > 0.007:
            # Month before day (US)
            temp_date = date.strftime("%m/%d/%y")

        else:
            # Day before month (EU)
            temp_date = date.strftime("%d/%m/%y")

        converted_date = ""
        if temp_date[0] == "0":
            converted_date += temp_date[1:3]

        else:
            converted_date += temp_date[0:3]

        if temp_date[3] == "0":
            converted_date += temp_date[4:]

        else:
            converted_date += temp_date[3:]

    return converted_date


def convert_string_to_date_full_digits_no_trailing_zeros(date):
    formats_to_try = ["%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%d/%m/%y"]

    for format_to_try in formats_to_try:
        try:
            return datetime.datetime.strptime(date, format_to_try)

        except:
            continue

    raise Exception("failed to parse")
