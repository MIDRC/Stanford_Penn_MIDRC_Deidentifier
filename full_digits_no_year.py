import random
import datetime
from utils import suffix


def convert_date_to_string_full_digits_no_year(date):
    temp_date = None

    if random.random() > 0.1:
        # Month before day (US)
        temp_date = date.strftime("%m/%d")

    else:
        # Day before month (EU)
        temp_date = date.strftime("%d/%m")

    converted_date = ""
    random_seed = random.random()

    if temp_date[0] == "0" and random_seed > 0.5:
        # Without trailing zero
        converted_date += temp_date[1:3]

    else:
        # With trailing zero
        converted_date += temp_date[0:3]

    if temp_date[3] == "0" and random_seed > 0.5:
        # Without trailing zero
        converted_date += temp_date[4:]

    else:
        # With trailing zero
        converted_date += temp_date[3:]

    return converted_date


def convert_string_to_date_full_digits_no_year(date):
    formats_to_try = ["%m/%d", "%d/%m", "%m/%y", "%y/%m", "%d/%y", "%y/%d"]

    for format_to_try in formats_to_try:
        try:
            return datetime.datetime.strptime(date, format_to_try)

        except:
            continue

    raise Exception("failed to parse")
