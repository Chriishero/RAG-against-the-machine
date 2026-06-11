import fire
from .CLI import CLI


def main() -> None:
    try:
        fire.Fire(CLI)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
