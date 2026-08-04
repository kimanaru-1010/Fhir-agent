from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

from app.skin_images.schemas import ResolvedSkinImageSearchFilters, SkinImageSearchFilters


LOCAL_TZ = timezone(timedelta(hours=7))
TIME_RANGES = {
    "morning": (time(5, 0), time(11, 30)),
    "noon": (time(11, 30), time(14, 0)),
    "afternoon": (time(14, 0), time(18, 0)),
    "evening": (time(18, 0), time(23, 0)),
    "night": (time(23, 0), time(5, 0)),
}


def resolve_skin_image_filters(
    filters: SkinImageSearchFilters,
    *,
    now: datetime | None = None,
) -> ResolvedSkinImageSearchFilters:
    patient_id = (filters.patient_id or "").strip()
    if not patient_id:
        raise ValueError("Patient ID is required to search skin images")

    local_now = (now or datetime.now(LOCAL_TZ)).astimezone(LOCAL_TZ)
    local_from, local_to = _resolve_date_window(filters, local_now)
    local_from, local_to = _apply_time_window(filters, local_from, local_to, local_now)

    modality = (filters.modality or "").strip().upper() or None
    if modality in {"SKIN", "DERMATOLOGY", "DERMATOLOGY IMAGE"}:
        modality = "XC"

    return ResolvedSkinImageSearchFilters(
        patient_id=patient_id,
        modality=modality,
        from_datetime=_to_utc(local_from),
        to_datetime=_to_utc(local_to),
        sort=filters.sort if filters.sort in {"asc", "desc"} else "desc",
        count=max(filters.count, 1) if filters.count is not None else None,
    )


def _resolve_date_window(
    filters: SkinImageSearchFilters,
    local_now: datetime,
) -> tuple[datetime | None, datetime | None]:
    if filters.last_N_minutes:
        return local_now - timedelta(minutes=filters.last_N_minutes), local_now

    if filters.specific_date:
        day = date.fromisoformat(_normalize_date(filters.specific_date))
        return _start_of_day(day), _end_of_day(day)

    if filters.specific_year:
        year = int(filters.specific_year)
        return (
            datetime(year, 1, 1, tzinfo=LOCAL_TZ),
            datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=LOCAL_TZ),
        )

    today = local_now.date()
    match (filters.date_range or "").lower():
        case "today":
            return _start_of_day(today), _end_of_day(today)
        case "yesterday":
            day = today - timedelta(days=1)
            return _start_of_day(day), _end_of_day(day)
        case "this_week":
            start = today - timedelta(days=today.weekday())
            return _start_of_day(start), _end_of_day(start + timedelta(days=6))
        case "last_week":
            this_start = today - timedelta(days=today.weekday())
            start = this_start - timedelta(days=7)
            return _start_of_day(start), _end_of_day(start + timedelta(days=6))
        case "this_month":
            start = today.replace(day=1)
            next_month = _add_month(start)
            return _start_of_day(start), _end_of_day(next_month - timedelta(days=1))
        case "last_month":
            this_month = today.replace(day=1)
            start = _add_month(this_month, -1)
            return _start_of_day(start), _end_of_day(this_month - timedelta(days=1))
        case "this_year":
            return datetime(today.year, 1, 1, tzinfo=LOCAL_TZ), datetime(today.year, 12, 31, 23, 59, 59, 999999, tzinfo=LOCAL_TZ)
        case "last_year":
            year = today.year - 1
            return datetime(year, 1, 1, tzinfo=LOCAL_TZ), datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=LOCAL_TZ)
        case "recent" | "recent_1day":
            return local_now - timedelta(days=1), local_now
        case _:
            return None, None


def _apply_time_window(
    filters: SkinImageSearchFilters,
    local_from: datetime | None,
    local_to: datetime | None,
    local_now: datetime,
) -> tuple[datetime | None, datetime | None]:
    if filters.time:
        parsed = _parse_time(filters.time)
        base_day = (local_from or local_now).date()
        point = datetime.combine(base_day, parsed, LOCAL_TZ)
        return point.replace(second=0, microsecond=0), point.replace(second=59, microsecond=999999)

    if filters.time_range:
        range_name = filters.time_range.lower()
        if range_name in TIME_RANGES:
            start_time, end_time = TIME_RANGES[range_name]
            base_day = (local_from or local_now).date()
            start = datetime.combine(base_day, start_time, LOCAL_TZ)
            end = datetime.combine(base_day, end_time, LOCAL_TZ)
            if end <= start:
                end += timedelta(days=1)
            if local_from:
                start = max(start, local_from)
            if local_to:
                end = min(end, local_to)
            return start, end

    return local_from, local_to


def _normalize_date(value: str) -> str:
    value = value.strip()
    if "/" in value:
        day, month, year = value.split("/")
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    return value


def _parse_time(value: str) -> time:
    parts = value.strip().replace("h", ":").split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    return time(hour, minute)


def _start_of_day(day: date) -> datetime:
    return datetime.combine(day, time.min, LOCAL_TZ)


def _end_of_day(day: date) -> datetime:
    return datetime.combine(day, time.max, LOCAL_TZ)


def _to_utc(value: datetime | None) -> datetime | None:
    return value.astimezone(timezone.utc) if value else None


def _add_month(day: date, months: int = 1) -> date:
    month = day.month - 1 + months
    year = day.year + month // 12
    month = month % 12 + 1
    return date(year, month, 1)
