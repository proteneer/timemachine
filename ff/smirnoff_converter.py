# converts smirnoff xmls into python dictionaries.

from simtk import unit
from xml.dom import minidom
xmldoc = minidom.parse('ff/smirnoff_1.1.0.xml')

# (ytz): lol i think i wrote this originally
def _ast_eval(node):
    """
    Performs an algebraic syntax tree evaluation of a unit.
    Parameters
    ----------
    node : An ast parsing tree node
    """
    import ast
    import operator as op

    operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
        ast.USub: op.neg}

    if isinstance(node, ast.Num): # <number>
        return node.n
    elif isinstance(node, ast.BinOp): # <left> <operator> <right>
        return operators[type(node.op)](_ast_eval(node.left), _ast_eval(node.right))
    elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
        return operators[type(node.op)](_ast_eval(node.operand))
    elif isinstance(node, ast.Name):
        # see if this is a simtk unit
        b = getattr(unit, node.id)
        return b
    # TODO: This was a quick hack that surprisingly worked. We should validate this further.
    elif isinstance(node, ast.List):
        return ast.literal_eval(node)
    else:
        raise TypeError(node)


def to_md_units(q):
    return q.value_in_unit_system(unit.md_unit_system)

def string_to_unit(unit_string):
    """
    Deserializes a simtk.unit.Quantity from a string representation, for
    example: "kilocalories_per_mole / angstrom ** 2"

    Parameters
    ----------
    unit_string : dict
        Serialized representation of a simtk.unit.Quantity.

    Returns
    -------
    output_unit: simtk.unit.Quantity
        The deserialized unit from the string

    """
    import ast
    output_unit = _ast_eval(ast.parse(unit_string, mode='eval').body)
    return output_unit

def parse_quantity(number_string):
    """
    Parse a quantity into MD units.
    """ 
    pos = number_string.find("*")

    number = float(number_string[:pos])
    item = number_string[pos+2:]
    quantity = number*string_to_unit(item)
    return to_md_units(quantity)


BOND_TAG = "Bond"
ANGLE_TAG = "Angle"
PROPER_TAG = "Proper"
IMPROPER_TAG = "Improper"
VDW_TAG = "Atom"

tags = [
    BOND_TAG,
    ANGLE_TAG,
    PROPER_TAG,
    IMPROPER_TAG,
    VDW_TAG
]

forcefield = {}

for tag in tags:
    itemlist = xmldoc.getElementsByTagName(tag)
    if tag == BOND_TAG:
        params = []
        for s in itemlist:
            patt = s.attributes['smirks'].value
            b0 = parse_quantity(s.attributes['length'].value)
            kb = parse_quantity(s.attributes['k'].value)
            params.append([patt, kb, b0])
        bonds = {
            "params": params,
        }
        forcefield[tag] = bonds

    elif tag == ANGLE_TAG:
        params = []
        for s in itemlist:
            patt = s.attributes['smirks'].value
            a0 = parse_quantity(s.attributes['angle'].value)
            ka = parse_quantity(s.attributes['k'].value)
            params.append([patt, ka, a0])
        angles = {
            "params": params,
        }
        forcefield[tag] = angles
    elif tag == PROPER_TAG:
        params = []
        for s in itemlist:
            patt = s.attributes['smirks'].value
            counter = 1
            components = []
            while True:
                try:
                    k = parse_quantity(s.attributes['k'+str(counter)].value)
                    phase = parse_quantity(s.attributes['phase'+str(counter)].value)
                    period = float(s.attributes['periodicity'+str(counter)].value)
                    idivf = float(s.attributes['idivf'+str(counter)].value)
                    k = k/idivf
                    components.append([k, phase, period])
                    counter += 1
                except KeyError:
                    break
            params.append([patt, components])
        torsions = {
            "params": params,
        }
        forcefield[tag] = torsions
    elif tag == IMPROPER_TAG:
        params = []
        for s in itemlist:
            patt = s.attributes['smirks'].value
            impdivf = 3
            k = parse_quantity(s.attributes['k1'].value)/impdivf
            phase = parse_quantity(s.attributes['phase1'].value)
            period = float(s.attributes['periodicity1'].value)
            params.append([patt, k, phase, period])
        impropers = {
            "params": params
        }
        forcefield[tag] = impropers
    elif tag == VDW_TAG:
        params = []
        for s in itemlist:
            patt = s.attributes['smirks'].value
            epsilon = parse_quantity(s.attributes['epsilon'].value)
            rmin_half = parse_quantity(s.attributes['rmin_half'].value)
            sigma = 2. * rmin_half / (2.**(1. / 6.))
            params.append([patt, sigma, epsilon])
        vdws = {
            "params": params,
            "props": {}
        }
        for key, val in xmldoc.getElementsByTagName("vdW")[0].attributes.items():
            if key == 'cutoff':
                # we don't do cuttoffs.
                continue
            elif 'scale' in key:
                val = float(val)
            elif key == 'switch_width':
                continue
            if key == "version":
                continue
            vdws["props"][key] = val 
        forcefield['vdW'] = vdws


import pprint
import json
pp = pprint.PrettyPrinter(width=500, compact=False)
pp._sorted = lambda x:x

pp.pprint(forcefield)

# print(json.dumps(forcefield, indent=4))

# import yaml

# print(yaml.dump(forcefield))