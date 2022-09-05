import random
import datetime
from utils import suffix
import re


def convert_date_to_string_digits_letters_with_comma(date):
    temp_date = None

    if random.random() > 0.2:
        # Year at the end

        if random.random() > 0.1:
            # Year in 4 digits
            temp_date = date.strftime("%B %d, %Y")
            converted_date = ""

            if temp_date[-8] == "0":

                if random.random() > 0.5:
                    # Without ordinal
                    converted_date = temp_date[0:-8] + temp_date[-7:]

                else:
                    # With ordinal
                    converted_date = (
                        temp_date[0:-8] + suffix(int(temp_date[-7])) + temp_date[-6:]
                    )

            else:

                if random.random() > 0.5:
                    # Without ordinal
                    converted_date += temp_date

                else:
                    # With ordinal
                    converted_date = (
                        temp_date[0:-8] + suffix(int(temp_date[-8:-6])) + temp_date[-6:]
                    )

        else:
            # Year in 2 digits
            temp_date = date.strftime("%B %d, %y")
            converted_date = ""

            if temp_date[-6] == "0":

                if random.random() > 0.5:
                    # Without ordinal
                    converted_date = temp_date[0:-6] + temp_date[-5:]

                else:
                    # With ordinal
                    converted_date = (
                        temp_date[0:-6] + suffix(int(temp_date[-5])) + temp_date[-4:]
                    )

            else:

                if random.random() > 0.5:
                    # Without ordinal
                    converted_date += temp_date

                else:
                    # With ordinal
                    converted_date = (
                        temp_date[0:-6] + suffix(int(temp_date[-6:-4])) + temp_date[-4:]
                    )

    else:
        # Year at the beginning

        if random.random() > 0.5:
            # Year in 4 digits
            temp_date = date.strftime("%Y, %B %d")
            converted_date = ""

            if temp_date[-2] == "0":

                if random.random() > 0.5:
                    # Without ordinal
                    converted_date = temp_date[0:-2] + temp_date[-1]

                else:
                    # With ordinal
                    converted_date = temp_date[0:-2] + suffix(int(temp_date[-1]))

            else:

                if random.random() > 0.5:
                    # Without ordinal
                    converted_date += temp_date

                else:
                    # With ordinal
                    converted_date = temp_date[0:-2] + suffix(int(temp_date[-2:]))

        else:
            # Year in 2 digits
            temp_date = date.strftime("%y, %B %d")
            converted_date = ""

            if temp_date[-2] == "0":

                if random.random() > 0.5:
                    # Without ordinal
                    converted_date = temp_date[0:-2] + temp_date[-1]

                else:
                    # With ordinal
                    converted_date = temp_date[0:-2] + suffix(int(temp_date[-1]))

            else:

                if random.random() > 0.5:
                    # Without ordinal
                    converted_date += temp_date

                else:
                    # With ordinal
                    converted_date = temp_date[0:-2] + suffix(int(temp_date[-2:]))

    return converted_date


def convert_string_to_date_digits_letters_with_comma(date):
    clean_date = date.lower().replace("th", "").replace("nd", "").replace("rd", "")
    clean_date = re.sub(r"\dst", lambda match: match.group(0)[0], clean_date, flags=re.DOTALL)
    formats_to_try = ["%B %d, %Y", "%d %B, %Y", "%B %Y, %d"]

    for format_to_try in formats_to_try:

        try:
            return datetime.datetime.strptime(clean_date, format_to_try)

        except:
            continue

    raise Exception("failed to parse")
