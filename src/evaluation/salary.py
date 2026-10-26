"""Check whether a predicted salary appears in a posting's text."""

import re

from job_extractor.schema import JobPosting

SALARY_FIELDS = ("salary_min_eur", "salary_max_eur")

# Amounts as written in postings: "90,000", "100.000", "100 000", "85000", "100k", "85.5k".
# Thousands groups need exactly three digits, so "8.50" is read as 8 and 50, not 850.
_NUMBER = re.compile(
    r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s?[kK]\b"
    r"|(?<!\d)(\d{1,3}(?:[.,\s  ]\d{3})+|\d+)(?!\d)"
)


def amounts_in(text: str) -> set[int]:
    amounts = set()
    for k_value, plain in _NUMBER.findall(text):
        if k_value:
            amounts.add(round(float(k_value.replace(",", ".")) * 1000))
        else:
            amounts.add(int(re.sub(r"\D", "", plain)))
    return amounts


def ungrounded_salary(text: str, posting: JobPosting) -> list[str]:
    """Names of the salary fields whose value can't be found in the posting text.

    A value counts as found if it appears as written, or as a monthly figure (value / 12),
    since the prompt asks the model to annualise monthly salaries.
    """
    amounts = amounts_in(text)
    missing = []
    for field in SALARY_FIELDS:
        value = getattr(posting, field)
        if value is None:
            continue
        if value in amounts or (value % 12 == 0 and value // 12 in amounts):
            continue
        missing.append(field)
    return missing
