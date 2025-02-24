def parse_args_and_kwargs_to_kwargs(signature: list[str], args: list, kwargs: dict) -> dict:
    result = {}
    for key, value in zip(signature, args):
        result[key] = value

    for key, value in kwargs.items():
        result[key] = value

    return result
