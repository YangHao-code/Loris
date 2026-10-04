"""List the three maintained, explicit workflow entry points."""
import argparse

def main():
    parser = argparse.ArgumentParser(description="Use python -m loris.discovery, loris.labeling, or loris.experiments.")
    parser.parse_args()
    parser.print_help()

if __name__ == "__main__":
    main()
