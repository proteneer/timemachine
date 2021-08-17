import os
import ast

from argparse import ArgumentParser



if __name__ == "__main__":
    parser = ArgumentParser(description="Compare Timemachine FFs")
    parser.add_argument("reference_ff")
    parser.add_argument("comp_ff")
    args = parser.parse_args()
    ref_path = os.path.expanduser(args.reference_ff)
    comp_path = os.path.expanduser(args.comp_ff)
    for path in [ref_path, comp_path]:
        if not os.path.isfile(path):
            print("No such path:", path)
            sys.exit(1)
    with open(ref_path, "r") as ifs:
        ref_ff = ast.literal_eval(ifs.read())
    with open(comp_path, "r") as ifs:
        comp_ff = ast.literal_eval(ifs.read())
    ref_keys = set(ref_ff.keys())
    comp_keys = set(comp_ff.keys())
    diff_keys = ref_keys.difference(comp_keys)
    if len(diff_keys):
        print("The top level sections differ, the following keys", diff_keys)
    for key in ref_keys:
        if key in diff_keys:
            continue
        for subkey in ref_ff[key]:
            if subkey not in comp_ff[key]:
                print(f"Section {key} has no {subkey} section for {comp_path}")
                continue
            if isinstance(ref_ff[key][subkey], dict):
                for dict_key, val in ref_ff[key][subkey].items():
                    comp_val = comp_ff[key][subkey].get(dict_key, None)
                    if val != comp_val:
                        print(f"Difference in {subkey} value for {dict_key}: Reference value {val} New Value {comp_val}")
            elif isinstance(ref_ff[key][subkey], (list, tuple)):
                for pattern in ref_ff[key][subkey]:
                    found = False
                    smirks, params = pattern[0], pattern[1:]
                    for comp_pattern in comp_ff[key][subkey]:
                        if smirks == comp_pattern[0]:
                            comp_params = comp_pattern[1:]
                            found = True
                            if len(pattern) != len(comp_pattern):
                                print(f"Mismatch of size for pattern {smirks}")
                            if any(ref_val != comp_val for ref_val, comp_val in zip(pattern, comp_pattern)):
                                print(f"{key} pattern {smirks} differs:")
                                print(f"Reference  {params}")
                                print(f"Comparison {comp_params}")
                                print()
                    if not found:
                        print(f"Comp FF has no pattern {smirks}")
            else:
                if ref_ff[key][subkey] != comp_ff[key][subkey]:
                    print(f"Difference in {subkey} value: Reference value {ref_ff[key][subkey]} New Value {comp_ff[key][subkey]}")

