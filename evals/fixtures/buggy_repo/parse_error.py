"""Bug: parses a non-numeric string as int."""


def parse_count(value: str) -> int:
    return int(value)


if __name__ == "__main__":
    print(parse_count("not-a-number"))
