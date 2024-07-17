import argparse
import yaml

def format_value(value):
    if value == 'null':
        return None
    # check if string is numeric
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--name', type=str, required=True)
    arg_parser.add_argument('--set', nargs='+', metavar="KEY=VALUE")
    args = arg_parser.parse_args()

    name = args.name

    params = {key: format_value(value) for key, value in (p.split('=') for p in args.set)}

    cfg = {name: params}
    print(yaml.dump(cfg, default_style=None))


if __name__ == '__main__':
    _main()
