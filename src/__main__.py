import fire
from .CLI import CLI


def main() -> None:
    """
    Apply the module 'fire' on the model 'CLI' and catch exceptions.
    """
    try:
        fire.Fire(CLI)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
