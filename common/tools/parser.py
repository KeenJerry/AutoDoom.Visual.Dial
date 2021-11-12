import json


def parse_json_file(file_path: str) -> list:
    with(open(file_path, mode="rb")) as json_file:
        return json.load(json_file)
