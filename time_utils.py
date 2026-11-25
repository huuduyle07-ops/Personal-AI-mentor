"""All application days and schedules use modern Korea Standard Time."""
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    KST = ZoneInfo("Asia/Seoul")
except ZoneInfoNotFoundError:
    KST = timezone(timedelta(hours=9), "Asia/Seoul")


def now_kst():
    return datetime.now(KST)


def following_day(day):
    return (date.fromisoformat(day) + timedelta(days=1)).isoformat()


def valid_day(day):
    if not isinstance(day, str) or date.fromisoformat(day).isoformat() != day:
        raise ValueError("Use an ISO date: YYYY-MM-DD")
    return day


def valid_minutes(value, low=0, high=1440):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"Minutes must be an integer from {low} to {high}")
    return value


def valid_time(value):
    if not isinstance(value, str) or len(value) != 5 or value[2] != ":":
        raise ValueError("Use a time in HH:MM format")
    try:
        hour, minute = map(int, value.split(":"))
    except ValueError:
        raise ValueError("Use a time in HH:MM format") from None
    if not 0 <= hour < 24 or not 0 <= minute < 60 or value != f"{hour:02}:{minute:02}":
        raise ValueError("Use a valid 24-hour HH:MM time")
    return value
