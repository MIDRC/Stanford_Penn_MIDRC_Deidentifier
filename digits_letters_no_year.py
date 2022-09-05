import random
import datetime
from utils import suffix
import re


def convert_date_to_string_digits_letters_no_year(date):
    temp_date = None

    if random.random() > 0.2:
        # Month before day (US)
        temp_date = date.strftime("%B %d")
        converted_date = ""

        if temp_date[-2] == "0" and random.random() > 0.5:
            # Without trailing zero

            if random.random() > 0.3:
                # Without ordinal
                converted_date = temp_date[0:-2] + temp_date[-1]

            else:
                # With ordinal
                converted_date = temp_date[0:-2] + suffix(int(temp_date[-1]))

        else:
            # With trailing zero

            if random.random() > 0.2:
                # Without ordinal
                converted_date += temp_date

            else:
                # With ordinal

                if temp_date[-2] == "0":
                    converted_date += temp_date[0:-1] + suffix(int(temp_date[-1]))

                else:
                    converted_date += temp_date[0:-2] + suffix(int(temp_date[-2:]))

    else:
        # Day before month (EU)
        temp_date = date.strftime("%d %B")
        converted_date = ""

        if temp_date[0] == "0" and random.random() > 0.5:
            # Without trailing zero

            if random.random() > 0.5:
                # Without ordinal
                converted_date = temp_date[1:]

            else:
                # With ordinal
                converted_date = suffix(int(temp_date[1])) + temp_date[2:]

        else:
            # With trailing zero

            if random.random() > 0.5:
                # Without ordinal
                converted_date += temp_date

            else:
                # With ordinal

                if temp_date[0] == "0":
                    converted_date += (
                        temp_date[0] + suffix(int(temp_date[1])) + temp_date[2:]
                    )

                else:
                    converted_date += suffix(int(temp_date[0:2])) + temp_date[2:]

    return converted_date


def convert_string_to_date_digits_letters_no_year(date):
    clean_date = date.lower().replace("th", "").replace("nd", "").replace("rd", "")
    clean_date = re.sub(r"\dst", lambda match: match.group(0)[0], clean_date, flags=re.DOTALL)
    formats_to_try = ["%B %d", "%d %B", "%B, %d", "%d, %B"]

    for format_to_try in formats_to_try:

        try:
            return datetime.datetime.strptime(clean_date, format_to_try)

        except:
            continue

    raise Exception("failed to parse")
