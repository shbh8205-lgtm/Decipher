from datetime import date, datetime


def normalize_israeli_id(value: str | None) -> tuple[str | None, bool]:
    digits = "".join(char for char in (value or "") if char.isascii() and char.isdigit())
    if not digits or len(digits) > 9:
        return digits or None, False

    normalized = digits.zfill(9)
    checksum = 0
    for index, char in enumerate(normalized):
        product = int(char) * (1 if index % 2 == 0 else 2)
        checksum += product // 10 + product % 10
    return normalized, checksum % 10 == 0


def normalize_received_date(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, "Missing sample receipt date"

    raw = value.strip()
    parsed = None
    for pattern in ("%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            parsed = datetime.strptime(raw, pattern).date()
            break
        except ValueError:
            continue

    if parsed is None:
        return None, "Unrecognized sample receipt date"
    if parsed > date.today():
        return parsed.strftime("%d/%m/%Y"), "Sample receipt date is in the future"
    if parsed.year < 2000:
        return parsed.strftime("%d/%m/%Y"), "Sample receipt date is unusually old"
    return parsed.strftime("%d/%m/%Y"), None
