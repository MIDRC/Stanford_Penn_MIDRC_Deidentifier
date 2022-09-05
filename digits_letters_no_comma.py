import random
import datetime
from utils import suffix
import re


def convert_date_to_string_digits_letters_no_comma(date):
    temp_date = None

    if random.random() > 0.5:
        # Year at the end

        if random.random() > 0.5:
            # Year in 4 digits

            if random.random() > 0.32:
                # Month before date (US format)
                temp_date = date.strftime("%B %d %Y")
                converted_date = ""

                if temp_date[-7] == "0":

                    if random.random() > 0.5:
                        # No ordinal
                        converted_date = temp_date[:-7] + temp_date[-6:]

                    else:
                        # With ordinal
                        converted_date = (
                            temp_date[:-7] + suffix(int(temp_date[-6])) + temp_date[-5:]
                        )

                else:

                    if random.random() > 0.5:
                        # No ordinal
                        converted_date += temp_date

                    else:
                        # With ordinal
                        converted_date = (
                            temp_date[:-7]
                            + suffix(int(temp_date[-7:-5]))
                            + temp_date[-5:]
                        )

            else:
                # Day before month (EU format)
                temp_date = date.strftime("%d %B %Y")
                converted_date = ""

                if temp_date[0] == "0":

                    if random.random() > 0.5:
                        # No ordinal
                        converted_date = temp_date[1:]

                    else:
                        # With ordinal
                        converted_date = suffix(int(temp_date[1])) + temp_date[2:]

                else:

                    if random.random() > 0.5:
                        # No ordinal
                        converted_date += temp_date

                    else:
                        # With ordinal
                        converted_date = suffix(int(temp_date[0:2])) + temp_date[2:]

        else:
            # Year in 2 digits

            if random.random() > 0.32:
                # Month before date (US format)
                temp_date = date.strftime("%B %d %y")
                converted_date = ""

                if temp_date[-5] == "0":

                    if random.random() > 0.5:
                        # No ordinal
                        converted_date = temp_date[:-5] + temp_date[-4:]

                    else:
                        # With ordinal
                        converted_date = (
                            temp_date[:-5] + suffix(int(temp_date[-4])) + temp_date[-3:]
                        )

                else:

                    if random.random() > 0.5:
                        # No ordinal
                        converted_date += temp_date

                    else:
                        # With ordinal
                        converted_date = (
                            temp_date[:-5]
                            + suffix(int(temp_date[-5:-3]))
                            + temp_date[-3:]
                        )

            else:
                # Day before month (EU format)
                temp_date = date.strftime("%d %B %y")
                converted_date = ""

                if temp_date[0] == "0":

                    if random.random() > 0.5:
                        converted_date = temp_date[1:]

                    else:
                        converted_date = suffix(int(temp_date[1])) + temp_date[2:]

                else:

                    if random.random() > 0.5:
                        converted_date += temp_date

                    else:
                        converted_date = suffix(int(temp_date[0:2])) + temp_date[2:]

    else:
        # Year at the beginning

        if random.random() > 0.5:
            # Year in 4 digits

            if random.random() > 0.32:
                # Month before day (US)
                temp_date = date.strftime("%Y %B %d")
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
                # Day before month (EU)
                temp_date = date.strftime("%Y %d %B")
                converted_date = ""

                if temp_date[5] == "0":

                    if random.random() > 0.5:
                        # Without ordinal
                        converted_date = temp_date[0:5] + temp_date[6:]

                    else:
                        # With ordinal
                        converted_date = (
                            temp_date[0:5] + suffix(int(temp_date[6])) + temp_date[7:]
                        )

                else:

                    if random.random() > 0.5:
                        # Without ordinal
                        converted_date += temp_date

                    else:
                        # With ordinal
                        converted_date = (
                            temp_date[0:5] + suffix(int(temp_date[5:7])) + temp_date[7:]
                        )

        else:
            # Year in 2 digits

            if random.random() > 0.32:
                # Month before day (US)
                temp_date = date.strftime("%y %B %d")
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
                # Day before month (EU)
                temp_date = date.strftime("%y %d %B")
                converted_date = ""

                if temp_date[3] == "0":

                    if random.random() > 0.5:
                        # Without ordinal
                        converted_date = temp_date[0:3] + temp_date[4:]

                    else:
                        # With ordinal
                        converted_date = (
                            temp_date[0:3] + suffix(int(temp_date[4])) + temp_date[5:]
                        )

                else:

                    if random.random() > 0.5:
                        # Without ordinal
                        converted_date += temp_date

                    else:
                        # With ordinal
                        converted_date = (
                            temp_date[0:3] + suffix(int(temp_date[3:5])) + temp_date[5:]
                        )

    return converted_date


def convert_string_to_date_digits_letters_no_comma(date):
    clean_date = date.lower().replace("th", "").replace("nd", "").replace("rd", "")
    clean_date = re.sub(r"\dst", lambda match: match.group(0)[0], clean_date, flags=re.DOTALL)
    formats_to_try = ["%d %B %Y", "%B %d %Y", "%Y %B %d", "%B %Y %d"]

    for format_to_try in formats_to_try:

        try:
            return datetime.datetime.strptime(clean_date, format_to_try)

        except:
            continue

    raise Exception("failed to parse")
