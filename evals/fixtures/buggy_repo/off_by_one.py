"""Bug: accesses index one past the end of a list."""


def last_item(items: list) -> object:
    return items[len(items)]  # should be len(items) - 1


if __name__ == "__main__":
    print(last_item([1, 2, 3]))
