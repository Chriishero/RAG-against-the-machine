import fire
from pydantic import ValidationError
from .CLI import CLI


def main() -> None:
    """
    Apply the module 'fire' on the model 'CLI' and catch exceptions.
    """
    try:
        fire.Fire(CLI)
    except ValidationError as e:
        print(e.errors()[0]["msg"].removeprefix("Value error, "))
    except ValueError as e:
        print(e)
    except KeyboardInterrupt:
        print("Keyboard interrupt.")


if __name__ == "__main__":
    main()
