import ast

class Forcefield():

    def __init__(self, handle):
        """
        Initialize the forcefield class.

        Parameters
        ----------
        handle: str or dict
            If str, then the handle is interpret as a path to be opened. If dict,
            then the handle will be used directly
        """

        if isinstance(handle, str):
            handle = open(handle).read()
            ff_raw = ast.literal_eval(handle)
        elif isinstance(handle, dict):
            ff_raw = handle

        global_params = []
        global_param_groups = []

        def add_param(p, p_group):
            assert isinstance(p_group, int)
            length = len(global_params)
            global_params.append(p)
            global_param_groups.append(p_group)
            return length

        # recursively replace parameters with indices and appending them into a global list.
        def recursive_replace(val, p_group):
            if isinstance(val, list):
                arr = []
                for f_idx, f in enumerate(val):
                    if isinstance(f, list):
                        fg = p_group
                    else:
                        fg = p_group[f_idx]
                    v = recursive_replace(f, fg)
                    arr.append(v)
                return arr
            elif isinstance(val, float) or isinstance(val, int):
                return add_param(val, p_group)
            else:
                raise Exception("Unsupported type")

        # (ytz): temporary useful debug code. remove later
        # print(recursive_replace([0.5, 0.6], (2,3)))
        # print(recursive_replace([[4.0, 2.0], [5.0, 1.0], [4.0, 233.0], [645.0, 1.0]], (11,12)))
        # print(global_params)
        # print(global_param_groups)
        # assert 0

        group_map = {
            "Angle": (0,1),
            "Bond": (2,3),
            "Improper": (4,5,6),
            "Proper": (7,8,9),
            "vdW": (10,11),
            "GBSA": (12,13),
            "SimpleCharges": (14,)
        }

        self.forcefield = {}
        # convert raw to proper
        for force_type, values in ff_raw.items():
            new_params = []
            for v in values["params"]:
                smirks = v[0]
                params = v[1:]
                param_idxs = recursive_replace(params, group_map[force_type])
                new_params.append([smirks, *param_idxs])
            self.forcefield[force_type] = {}
            self.forcefield[force_type]["params"] = new_params
            if "props" in values:
                self.forcefield[force_type]["props"] = values["props"]

        assert len(global_params) == len(global_param_groups)
        
        self.params = global_params
        self.param_groups = global_param_groups

    def serialize(self):
        """
        Serialize the forcefield to an python dictionary.
        """

        def recursive_lookup(val):
            if isinstance(val, list):
                arr = []
                for f in val:
                    v = recursive_lookup(f)
                    arr.append(v)
                return arr
            elif isinstance(val, int):
                return self.params[val]
            else:
                raise Exception("Unsupported type")

        raw_ff = {}

        for force_type, values in self.forcefield.items():
            raw_ff[force_type] = {}
            new_params = []
            for v in values["params"]:
                smirks = v[0]
                param_idxs = v[1:]
                # print(smirks, param_idxs)
                param_vals = recursive_lookup(param_idxs)
                new_params.append([smirks, *param_vals])

            raw_ff[force_type]["params"] = new_params
            if "props" in values:
                raw_ff[force_type]["props"] = values["props"]

        return raw_ff
