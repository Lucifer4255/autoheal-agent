"""Bug: dereferences None unconditionally."""


def get_user_name(user: dict | None) -> str:
    return user["name"]  # crashes when user is None


if __name__ == "__main__":
    print(get_user_name(None))
