from datetime import date, datetime

AGE_MIN = 20
AGE_MAX = 60


def parse_age(value):
    try:
        age = int(value)
    except (TypeError, ValueError):
        return None
    if AGE_MIN <= age <= AGE_MAX:
        return age
    return None


def parse_birthdate(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def calculate_age_from_birthdate(birthdate: date, today: date | None = None) -> int:
    now = today or date.today()
    age = now.year - birthdate.year
    if (now.month, now.day) < (birthdate.month, birthdate.day):
        age -= 1
    return age


def normalize_age_preference(value):
    if not isinstance(value, dict):
        return None
    try:
        min_age = int(value.get("min"))
        max_age = int(value.get("max"))
    except (TypeError, ValueError):
        return None

    lower = min(min_age, max_age)
    upper = max(min_age, max_age)
    if lower < AGE_MIN or upper > AGE_MAX:
        return None

    return {"min": lower, "max": upper}
