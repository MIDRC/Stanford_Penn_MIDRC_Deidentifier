import random
import datetime
from utils import suffix
import re


def convert_date_to_string_digits_letters_no_day(date):
    random_seed = random.random()
    converted_date = None

    if random_seed > 0.15:
        # Year at the end, 4 digits
        converted_date = date.strftime("%B %Y")

    elif random_seed > 0.1:
        # Year at the end, 2 digits
        converted_date = date.strftime("%B %y")

    elif random_seed > 0.05:
        # Year at the beginning, 4 digits
        converted_date = date.strftime("%Y %B")

    else:
        # Year at the beginning, 2 digits
        converted_date = date.strftime("%y %B")

    return converted_date


def convert_string_to_date_digits_letters_no_day(date):
    clean_date = date.lower().replace("th", "").replace("nd", "").replace("rd", "")
    clean_date = re.sub(r"\dst", lambda match: match.group(0)[0], clean_date, flags=re.DOTALL)
    formats_to_try = ["%B %Y", "%Y %B", "%B, %Y", "%Y, %B"]

    for format_to_try in formats_to_try:

        try:
            return datetime.datetime.strptime(clean_date, format_to_try)

        except:
            continue

    raise Exception("failed to parse")
