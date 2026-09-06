"""Vendored TUM RGB-D ``associate`` helper.

The Stage 3 ``eval_ate.py`` provided with the assignment does ``import
associate`` but the module is not shipped alongside it.  This is the standard
TUM benchmark helper (BSD, Juergen Sturm / TUM), vendored unchanged in behaviour
so the official evaluator can run.  The Stage 3 official-eval wrapper adds this
directory to ``PYTHONPATH`` when invoking ``eval_ate.py`` so the provided script
resolves this module without modifying the assignment files.
"""


def read_file_list(filename):
    """Read a trajectory/association file into a ``{timestamp: [values]}`` dict."""
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    parsed = [
        [value.strip() for value in line.split(" ") if value.strip() != ""]
        for line in lines
        if len(line) > 0 and line[0] != "#"
    ]
    parsed = [(float(entry[0]), entry[1:]) for entry in parsed if len(entry) > 1]
    return dict(parsed)


def associate(first_list, second_list, offset, max_difference):
    """Greedy nearest-timestamp association within ``max_difference`` seconds."""
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    potential_matches.sort()
    matches = []
    for _diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    matches.sort()
    return matches
