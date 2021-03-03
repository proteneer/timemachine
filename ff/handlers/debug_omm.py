# Modified by ytz to add support for Differentiability. Taken initially from forcefield.py in OpenMM
"""
forcefield.py: Constructs OpenMM System objects based on a Topology and an XML force field description

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2012-2019 Stanford University and the Authors.
Authors: Peter Eastman, Mark Friedrichs
Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


from __future__ import absolute_import, print_function

import jax.numpy as np
import numpy as onp

__author__ = "Peter Eastman"
__version__ = "1.0"

import os
import itertools
import xml.etree.ElementTree as etree
import math
from math import sqrt, cos
from copy import deepcopy
from collections import defaultdict
import simtk.openmm as mm
import simtk.unit as unit
from simtk.openmm.app import element as elem
from simtk.openmm.app import Topology
from simtk.openmm.app.internal.singleton import Singleton
from simtk.openmm.app.internal import compiled

# Directories from which to load built in force fields.

_dataDirectories = None

def _getDataDirectories():
    global _dataDirectories
    if _dataDirectories is None:
        _dataDirectories = [os.path.join(os.path.dirname(__file__), 'data')]
        try:
            from pkg_resources import iter_entry_points
            for entry in iter_entry_points(group='openmm.forcefielddir'):
                _dataDirectories.append(entry.load()())
        except:
            pass # pkg_resources is not installed
    return _dataDirectories

def _convertParameterToNumber(param):
    if unit.is_quantity(param):
        if param.unit.is_compatible(unit.bar):
            return param / unit.bar
        return param.value_in_unit_system(unit.md_unit_system)
    return float(param)

def _parseFunctions(element):
    """Parse the attributes on an XML tag to find any tabulated functions it defines."""
    functions = []
    for function in element.findall('Function'):
        values = [float(x) for x in function.text.split()]
        if 'type' in function.attrib:
            functionType = function.attrib['type']
        else:
            functionType = 'Continuous1D'
        params = {}
        for key in function.attrib:
            if key.endswith('size'):
                params[key] = int(function.attrib[key])
            elif key.endswith('min') or key.endswith('max'):
                params[key] = float(function.attrib[key])
        functions.append((function.attrib['name'], functionType, values, params))
    return functions

def _createFunctions(force, functions):
    """Add TabulatedFunctions to a Force based on the information that was recorded by _parseFunctions()."""
    for (name, type, values, params) in functions:
        if type == 'Continuous1D':
            force.addTabulatedFunction(name, mm.Continuous1DFunction(values, params['min'], params['max']))
        elif type == 'Continuous2D':
            force.addTabulatedFunction(name, mm.Continuous2DFunction(params['xsize'], params['ysize'], values, params['xmin'], params['xmax'], params['ymin'], params['ymax']))
        elif type == 'Continuous3D':
            force.addTabulatedFunction(name, mm.Continuous2DFunction(params['xsize'], params['ysize'], params['zsize'], values, params['xmin'], params['xmax'], params['ymin'], params['ymax'], params['zmin'], params['zmax']))
        elif type == 'Discrete1D':
            force.addTabulatedFunction(name, mm.Discrete1DFunction(values))
        elif type == 'Discrete2D':
            force.addTabulatedFunction(name, mm.Discrete2DFunction(params['xsize'], params['ysize'], values))
        elif type == 'Discrete3D':
            force.addTabulatedFunction(name, mm.Discrete2DFunction(params['xsize'], params['ysize'], params['zsize'], values))

# Enumerated values for nonbonded method

class NoCutoff(Singleton):
    def __repr__(self):
        return 'NoCutoff'
NoCutoff = NoCutoff()

class CutoffNonPeriodic(Singleton):
    def __repr__(self):
        return 'CutoffNonPeriodic'
CutoffNonPeriodic = CutoffNonPeriodic()

class CutoffPeriodic(Singleton):
    def __repr__(self):
        return 'CutoffPeriodic'
CutoffPeriodic = CutoffPeriodic()

class Ewald(Singleton):
    def __repr__(self):
        return 'Ewald'
Ewald = Ewald()

class PME(Singleton):
    def __repr__(self):
        return 'PME'
PME = PME()

class LJPME(Singleton):
    def __repr__(self):
        return 'LJPME'
LJPME = LJPME()

# Enumerated values for constraint type

class HBonds(Singleton):
    def __repr__(self):
        return 'HBonds'
HBonds = HBonds()

class AllBonds(Singleton):
    def __repr__(self):
        return 'AllBonds'
AllBonds = AllBonds()

class HAngles(Singleton):
    def __repr__(self):
        return 'HAngles'
HAngles = HAngles()

# A map of functions to parse elements of the XML file.

parsers = {}

class ForceField(object):
    """A ForceField constructs OpenMM System objects based on a Topology."""

    def __init__(self, *files):
        """Load one or more XML files and create a ForceField object based on them.

        Parameters
        ----------
        files : list
            A list of XML files defining the force field.  Each entry may
            be an absolute file path, a path relative to the current working
            directory, a path relative to this module's data subdirectory
            (for built in force fields), or an open file-like object with a
            read() method from which the forcefield XML data can be loaded.
        """
        self._atomTypes = {}
        self._templates = {} # protein templates
        self._patches = {} # empty
        self._templatePatches = {} # empty
        self._templateSignatures = {None:[]}
        self._atomClasses = {'':set()}
        self._forces = []
        self._scripts = [] # empty
        self._templateGenerators = []
        self.loadFile(files)

    def loadFile(self, files):
        """Load an XML file and add the definitions from it to this ForceField.

        Parameters
        ----------
        files : string or file or tuple
            An XML file or tuple of XML files containing force field definitions.
            Each entry may be either an absolute file path, a path relative to the current working
            directory, a path relative to this module's data subdirectory (for
            built in force fields), or an open file-like object with a read()
            method from which the forcefield XML data can be loaded.
        """

        if isinstance(files, tuple):
            files = list(files)
        else:
            files = [files]

        trees = []

        i = 0
        while i < len(files):
            file = files[i]
            tree = None
            try:
                # this handles either filenames or open file-like objects
                tree = etree.parse(file)
            except IOError:
                for dataDir in _getDataDirectories():
                    f = os.path.join(dataDir, file)
                    if os.path.isfile(f):
                        tree = etree.parse(f)
                        break
            except Exception as e:
                # Fail with an error message about which file could not be read.
                # TODO: Also handle case where fallback to 'data' directory encounters problems,
                # but this is much less worrisome because we control those files.
                msg  = str(e) + '\n'
                if hasattr(file, 'name'):
                    filename = file.name
                else:
                    filename = str(file)
                msg += "ForceField.loadFile() encountered an error reading file '%s'\n" % filename
                raise Exception(msg)
            if tree is None:
                raise ValueError('Could not locate file "%s"' % file)

            trees.append(tree)
            i += 1

            # Process includes in this file.

            if isinstance(file, str):
                parentDir = os.path.dirname(file)
            else:
                parentDir = ''
            for included in tree.getroot().findall('Include'):
                includeFile = included.attrib['file']
                joined = os.path.join(parentDir, includeFile)
                if os.path.isfile(joined):
                    includeFile = joined
                if includeFile not in files:
                    files.append(includeFile)

        # Load the atom types.

        for tree in trees:
            if tree.getroot().find('AtomTypes') is not None:
                for type in tree.getroot().find('AtomTypes').findall('Type'):
                    self.registerAtomType(type.attrib)

        # Load the residue templates.

        for tree in trees:
            if tree.getroot().find('Residues') is not None:
                for residue in tree.getroot().find('Residues').findall('Residue'):
                    resName = residue.attrib['name']
                    template = ForceField._TemplateData(resName)
                    if 'override' in residue.attrib:
                        assert 0
                        template.overrideLevel = int(residue.attrib['override'])
                    atomIndices = template.atomIndices
                    for ia, atom in enumerate(residue.findall('Atom')):
                        params = {}
                        for key in atom.attrib:
                            if key not in ('name', 'type'):
                                assert 0
                                params[key] = _convertParameterToNumber(atom.attrib[key])
                        atomName = atom.attrib['name']
                        if atomName in atomIndices:
                            raise ValueError('Residue '+resName+' contains multiple atoms named '+atomName)
                        typeName = atom.attrib['type']
                        atomIndices[atomName] = ia
                        template.atoms.append(ForceField._TemplateAtomData(atomName, typeName, self._atomTypes[typeName].element, params))
                    for site in residue.findall('VirtualSite'):
                        assert 0
                        template.virtualSites.append(ForceField._VirtualSiteData(site, atomIndices))
                    for bond in residue.findall('Bond'):
                        if 'atomName1' in bond.attrib:
                            template.addBondByName(bond.attrib['atomName1'], bond.attrib['atomName2'])
                        else:
                            template.addBond(int(bond.attrib['from']), int(bond.attrib['to']))
                    for bond in residue.findall('ExternalBond'):
                        if 'atomName' in bond.attrib:
                            template.addExternalBondByName(bond.attrib['atomName'])
                        else:
                            template.addExternalBond(int(bond.attrib['from']))
                    for patch in residue.findall('AllowPatch'):
                        assert 0
                        patchName = patch.attrib['name']
                        if ':' in patchName:
                            colonIndex = patchName.find(':')
                            self.registerTemplatePatch(resName, patchName[:colonIndex], int(patchName[colonIndex+1:])-1)
                        else:
                            self.registerTemplatePatch(resName, patchName, 0)

                    self.registerResidueTemplate(template)

        # Load the patch defintions.

        for tree in trees:
            if tree.getroot().find('Patches') is not None:
                assert 0
                for patch in tree.getroot().find('Patches').findall('Patch'):
                    patchName = patch.attrib['name']
                    if 'residues' in patch.attrib:
                        numResidues = int(patch.attrib['residues'])
                    else:
                        numResidues = 1
                    patchData = ForceField._PatchData(patchName, numResidues)
                    for atom in patch.findall('AddAtom'):
                        params = {}
                        for key in atom.attrib:
                            if key not in ('name', 'type'):
                                params[key] = _convertParameterToNumber(atom.attrib[key])
                        atomName = atom.attrib['name']
                        if atomName in patchData.allAtomNames:
                            raise ValueError('Patch '+patchName+' contains multiple atoms named '+atomName)
                        patchData.allAtomNames.add(atomName)
                        atomDescription = ForceField._PatchAtomData(atomName)
                        typeName = atom.attrib['type']
                        patchData.addedAtoms[atomDescription.residue].append(ForceField._TemplateAtomData(atomDescription.name, typeName, self._atomTypes[typeName].element, params))
                    for atom in patch.findall('ChangeAtom'):
                        params = {}
                        for key in atom.attrib:
                            if key not in ('name', 'type'):
                                params[key] = _convertParameterToNumber(atom.attrib[key])
                        atomName = atom.attrib['name']
                        if atomName in patchData.allAtomNames:
                            raise ValueError('Patch '+patchName+' contains multiple atoms named '+atomName)
                        patchData.allAtomNames.add(atomName)
                        atomDescription = ForceField._PatchAtomData(atomName)
                        typeName = atom.attrib['type']
                        patchData.changedAtoms[atomDescription.residue].append(ForceField._TemplateAtomData(atomDescription.name, typeName, self._atomTypes[typeName].element, params))
                    for atom in patch.findall('RemoveAtom'):
                        atomName = atom.attrib['name']
                        if atomName in patchData.allAtomNames:
                            raise ValueError('Patch '+patchName+' contains multiple atoms named '+atomName)
                        patchData.allAtomNames.add(atomName)
                        atomDescription = ForceField._PatchAtomData(atomName)
                        patchData.deletedAtoms.append(atomDescription)
                    for bond in patch.findall('AddBond'):
                        atom1 = ForceField._PatchAtomData(bond.attrib['atomName1'])
                        atom2 = ForceField._PatchAtomData(bond.attrib['atomName2'])
                        patchData.addedBonds.append((atom1, atom2))
                    for bond in patch.findall('RemoveBond'):
                        atom1 = ForceField._PatchAtomData(bond.attrib['atomName1'])
                        atom2 = ForceField._PatchAtomData(bond.attrib['atomName2'])
                        patchData.deletedBonds.append((atom1, atom2))
                    for bond in patch.findall('AddExternalBond'):
                        atom = ForceField._PatchAtomData(bond.attrib['atomName'])
                        patchData.addedExternalBonds.append(atom)
                    for bond in patch.findall('RemoveExternalBond'):
                        atom = ForceField._PatchAtomData(bond.attrib['atomName'])
                        patchData.deletedExternalBonds.append(atom)
                    for residue in patch.findall('ApplyToResidue'):
                        name = residue.attrib['name']
                        if ':' in name:
                            colonIndex = name.find(':')
                            self.registerTemplatePatch(name[colonIndex+1:], patchName, int(name[:colonIndex])-1)
                        else:
                            self.registerTemplatePatch(name, patchName, 0)
                    self.registerPatch(patchData)

        # Load force definitions

        for tree in trees:
            for child in tree.getroot():
                if child.tag in parsers:
                    parsers[child.tag](child, self)

        # Load scripts

        for tree in trees:
            for node in tree.getroot().findall('Script'):
                assert 0
                self.registerScript(node.text)

    def getGenerators(self):
        """Get the list of all registered generators."""
        return self._forces

    def registerGenerator(self, generator):
        """Register a new generator."""
        self._forces.append(generator)

    def registerAtomType(self, parameters):
        """Register a new atom type."""
        name = parameters['name']
        if name in self._atomTypes:
            raise ValueError('Found multiple definitions for atom type: '+name)
        atomClass = parameters['class']
        mass = _convertParameterToNumber(parameters['mass'])
        element = None
        if 'element' in parameters:
            element = parameters['element']
            if not isinstance(element, elem.Element):
                element = elem.get_by_symbol(element)
        self._atomTypes[name] = ForceField._AtomType(name, atomClass, mass, element)
        if atomClass in self._atomClasses:
            typeSet = self._atomClasses[atomClass]
        else:
            typeSet = set()
            self._atomClasses[atomClass] = typeSet
        typeSet.add(name)
        self._atomClasses[''].add(name)

    def registerResidueTemplate(self, template):
        """Register a new residue template."""
        if template.name in self._templates:
            # There is already a template with this name, so check the override levels.

            existingTemplate = self._templates[template.name]
            if template.overrideLevel < existingTemplate.overrideLevel:
                # The existing one takes precedence, so just return.
                return
            if template.overrideLevel > existingTemplate.overrideLevel:
                # We need to delete the existing template.
                del self._templates[template.name]
                existingSignature = _createResidueSignature([atom.element for atom in existingTemplate.atoms])
                self._templateSignatures[existingSignature].remove(existingTemplate)
            else:
                raise ValueError('Residue template %s with the same override level %d already exists.' % (template.name, template.overrideLevel))

        # Register the template.

        self._templates[template.name] = template
        signature = _createResidueSignature([atom.element for atom in template.atoms])
        if signature in self._templateSignatures:
            self._templateSignatures[signature].append(template)
        else:
            self._templateSignatures[signature] = [template]

    def registerPatch(self, patch):
        """Register a new patch that can be applied to templates."""
        self._patches[patch.name] = patch

    def registerTemplatePatch(self, residue, patch, patchResidueIndex):
        """Register that a particular patch can be used with a particular residue."""
        if residue not in self._templatePatches:
            self._templatePatches[residue] = set()
        self._templatePatches[residue].add((patch, patchResidueIndex))

    def registerScript(self, script):
        """Register a new script to be executed after building the System."""
        self._scripts.append(script)

    def registerTemplateGenerator(self, generator):
        """Register a residue template generator that can be used to parameterize residues that do not match existing forcefield templates.

        This functionality can be used to add handlers to parameterize small molecules or unnatural/modified residues.

        .. CAUTION:: This method is experimental, and its API is subject to change.

        Parameters
        ----------
        generator : function
            A function that will be called when a residue is encountered that does not match an existing forcefield template.

        When a residue without a template is encountered, the ``generator`` function is called with:

        ::
           success = generator(forcefield, residue)

        where ``forcefield`` is the calling ``ForceField`` object and ``residue`` is a simtk.openmm.app.topology.Residue object.

        ``generator`` must conform to the following API:

        ::
           generator API

           Parameters
           ----------
           forcefield : simtk.openmm.app.ForceField
               The ForceField object to which residue templates and/or parameters are to be added.
           residue : simtk.openmm.app.Topology.Residue
               The residue topology for which a template is to be generated.

           Returns
           -------
           success : bool
               If the generator is able to successfully parameterize the residue, `True` is returned.
               If the generator cannot parameterize the residue, it should return `False` and not modify `forcefield`.

           The generator should either register a residue template directly with `forcefield.registerResidueTemplate(template)`
           or it should call `forcefield.loadFile(file)` to load residue definitions from an ffxml file.

           It can also use the `ForceField` programmatic API to add additional atom types (via `forcefield.registerAtomType(parameters)`)
           or additional parameters.

        """
        self._templateGenerators.append(generator)

    def _findAtomTypes(self, attrib, num):
        """Parse the attributes on an XML tag to find the set of atom types for each atom it involves.

        Parameters
        ----------
        attrib : dict of attributes
            The dictionary of attributes for an XML parameter tag.
        num : int
            The number of atom specifiers (e.g. 'class1' through 'class4') to extract.

        Returns
        -------
        types : list
            A list of atom types that match.

        """
        types = []
        for i in range(num):
            if num == 1:
                suffix = ''
            else:
                suffix = str(i+1)
            classAttrib = 'class'+suffix
            typeAttrib = 'type'+suffix
            if classAttrib in attrib:
                if typeAttrib in attrib:
                    raise ValueError('Specified both a type and a class for the same atom: '+str(attrib))
                if attrib[classAttrib] not in self._atomClasses:
                    types.append(None) # Unknown atom class
                else:
                    types.append(self._atomClasses[attrib[classAttrib]])
            elif typeAttrib in attrib:
                if attrib[typeAttrib] == '':
                    types.append(self._atomClasses[''])
                elif attrib[typeAttrib] not in self._atomTypes:
                    types.append(None) # Unknown atom type
                else:
                    types.append([attrib[typeAttrib]])
            else:
                types.append(None) # Unknown atom type
        return types

    def _parseTorsion(self, attrib):
        """Parse the node defining a torsion."""
        types = self._findAtomTypes(attrib, 4)
        if None in types:
            return None
        torsion = PeriodicTorsion(types)
        index = 1
        while 'phase%d'%index in attrib:
            torsion.periodicity.append(int(attrib['periodicity%d'%index]))
            torsion.phase.append(_convertParameterToNumber(attrib['phase%d'%index]))
            torsion.k.append(_convertParameterToNumber(attrib['k%d'%index]))
            index += 1
        return torsion

    class _SystemData(object):
        """Inner class used to encapsulate data about the system being created."""
        def __init__(self):
            self.atomType = {}
            self.atomParameters = {}
            self.atomTemplateIndexes = {}
            self.atoms = []
            self.excludeAtomWith = []
            self.virtualSites = {}
            self.bonds = []
            self.angles = []
            self.propers = []
            self.impropers = []
            self.atomBonds = []
            self.isAngleConstrained = []
            self.constraints = {}

        def addConstraint(self, system, atom1, atom2, distance):
            """Add a constraint to the system, avoiding duplicate constraints."""
            key = (min(atom1, atom2), max(atom1, atom2))
            if key in self.constraints:
                if self.constraints[key] != distance:
                    raise ValueError('Two constraints were specified between atoms %d and %d with different distances' % (atom1, atom2))
            else:
                self.constraints[key] = distance
                system.addConstraint(atom1, atom2, distance)

        def recordMatchedAtomParameters(self, residue, template, matches):
            """Record parameters for atoms based on having matched a residue to a template."""
            matchAtoms = dict(zip(matches, residue.atoms()))
            for atom, match in zip(residue.atoms(), matches):
                self.atomType[atom] = template.atoms[match].type
                self.atomParameters[atom] = template.atoms[match].parameters
                self.atomTemplateIndexes[atom] = match
                for site in template.virtualSites:
                    if match == site.index:
                        self.virtualSites[atom] = (site, [matchAtoms[i].index for i in site.atoms], matchAtoms[site.excludeWith].index)

    class _TemplateData(object):
        """Inner class used to encapsulate data about a residue template definition."""
        def __init__(self, name):
            self.name = name
            self.atoms = []
            self.atomIndices = {}
            self.virtualSites = []
            self.bonds = []
            self.externalBonds = []
            self.overrideLevel = 0

        def getAtomIndexByName(self, atom_name):
            """Look up an atom index by atom name, providing a helpful error message if not found."""
            index = self.atomIndices.get(atom_name, None)
            if index is not None:
                return index

            # Provide a helpful error message if atom name not found.
            msg =  "Atom name '%s' not found in residue template '%s'.\n" % (atom_name, self.name)
            msg += "Possible atom names are: %s" % str(list(map(lambda x: x.name, self.atoms)))
            raise ValueError(msg)

        def addAtom(self, atom):
            self.atoms.append(atom)
            self.atomIndices[atom.name] = len(self.atoms)-1

        def addBond(self, atom1, atom2):
            """Add a bond between two atoms in a template given their indices in the template."""
            self.bonds.append((atom1, atom2))
            self.atoms[atom1].bondedTo.append(atom2)
            self.atoms[atom2].bondedTo.append(atom1)

        def addBondByName(self, atom1_name, atom2_name):
            """Add a bond between two atoms in a template given their atom names."""
            atom1 = self.getAtomIndexByName(atom1_name)
            atom2 = self.getAtomIndexByName(atom2_name)
            self.addBond(atom1, atom2)

        def addExternalBond(self, atom_index):
            """Designate that an atom in a residue template has an external bond, using atom index within template."""
            self.externalBonds.append(atom_index)
            self.atoms[atom_index].externalBonds += 1

        def addExternalBondByName(self, atom_name):
            """Designate that an atom in a residue template has an external bond, using atom name within template."""
            atom = self.getAtomIndexByName(atom_name)
            self.addExternalBond(atom)

    class _TemplateAtomData(object):
        """Inner class used to encapsulate data about an atom in a residue template definition."""
        def __init__(self, name, type, element, parameters={}):
            self.name = name
            self.type = type
            self.element = element
            self.parameters = parameters
            self.bondedTo = []
            self.externalBonds = 0

    class _BondData(object):
        """Inner class used to encapsulate data about a bond."""
        def __init__(self, atom1, atom2):
            self.atom1 = atom1
            self.atom2 = atom2
            self.isConstrained = False
            self.length = 0.0

    class _VirtualSiteData(object):
        """Inner class used to encapsulate data about a virtual site."""
        def __init__(self, node, atomIndices):
            attrib = node.attrib
            self.type = attrib['type']
            if self.type == 'average2':
                numAtoms = 2
                self.weights = [float(attrib['weight1']), float(attrib['weight2'])]
            elif self.type == 'average3':
                numAtoms = 3
                self.weights = [float(attrib['weight1']), float(attrib['weight2']), float(attrib['weight3'])]
            elif self.type == 'outOfPlane':
                numAtoms = 3
                self.weights = [float(attrib['weight12']), float(attrib['weight13']), float(attrib['weightCross'])]
            elif self.type == 'localCoords':
                numAtoms = 0
                self.originWeights = []
                self.xWeights = []
                self.yWeights = []
                while ('wo%d' % (numAtoms+1)) in attrib:
                    numAtoms += 1
                    self.originWeights.append(float(attrib['wo%d' % numAtoms]))
                    self.xWeights.append(float(attrib['wx%d' % numAtoms]))
                    self.yWeights.append(float(attrib['wy%d' % numAtoms]))
                self.localPos = [float(attrib['p1']), float(attrib['p2']), float(attrib['p3'])]
            else:
                raise ValueError('Unknown virtual site type: %s' % self.type)
            if 'siteName' in attrib:
                self.index = atomIndices[attrib['siteName']]
                self.atoms = [atomIndices[attrib['atomName%d'%(i+1)]] for i in range(numAtoms)]
            else:
                self.index = int(attrib['index'])
                self.atoms = [int(attrib['atom%d'%(i+1)]) for i in range(numAtoms)]
            if 'excludeWith' in attrib:
                self.excludeWith = int(attrib['excludeWith'])
            else:
                self.excludeWith = self.atoms[0]

    class _PatchData(object):
        """Inner class used to encapsulate data about a patch definition."""
        def __init__(self, name, numResidues):
            self.name = name
            self.numResidues = numResidues
            self.addedAtoms = [[] for i in range(numResidues)]
            self.changedAtoms = [[] for i in range(numResidues)]
            self.deletedAtoms = []
            self.addedBonds = []
            self.deletedBonds = []
            self.addedExternalBonds = []
            self.deletedExternalBonds = []
            self.allAtomNames = set()

        def createPatchedTemplates(self, templates):
            """Apply this patch to a set of templates, creating new modified ones."""
            if len(templates) != self.numResidues:
                raise ValueError("Patch '%s' expected %d templates, received %d", (self.name, self.numResidues, len(templates)))

            # Construct a new version of each template.

            newTemplates = []
            for index, template in enumerate(templates):
                newTemplate = ForceField._TemplateData("%s-%s" % (template.name, self.name))
                newTemplates.append(newTemplate)

                # Build the list of atoms in it.

                for atom in template.atoms:
                    if not any(deleted.name == atom.name and deleted.residue == index for deleted in self.deletedAtoms):
                        newTemplate.addAtom(ForceField._TemplateAtomData(atom.name, atom.type, atom.element, atom.parameters))
                for atom in self.addedAtoms[index]:
                    if any(a.name == atom.name for a in newTemplate.atoms):
                        raise ValueError("Patch '%s' adds an atom with the same name as an existing atom: %s" % (self.name, atom.name))
                    newTemplate.addAtom(ForceField._TemplateAtomData(atom.name, atom.type, atom.element, atom.parameters))
                oldAtomIndex = dict([(atom.name, i) for i, atom in enumerate(template.atoms)])
                newAtomIndex = dict([(atom.name, i) for i, atom in enumerate(newTemplate.atoms)])
                for atom in self.changedAtoms[index]:
                    if atom.name not in newAtomIndex:
                        raise ValueError("Patch '%s' modifies nonexistent atom '%s' in template '%s'" % (self.name, atom.name, template.name))
                    newTemplate.atoms[newAtomIndex[atom.name]] = ForceField._TemplateAtomData(atom.name, atom.type, atom.element, atom.parameters)

                # Copy over the virtual sites, translating the atom indices.

                indexMap = dict([(oldAtomIndex[name], newAtomIndex[name]) for name in newAtomIndex if name in oldAtomIndex])
                for site in template.virtualSites:
                    if site.index in indexMap and all(i in indexMap for i in site.atoms):
                        newSite = deepcopy(site)
                        newSite.index = indexMap[site.index]
                        newSite.atoms = [indexMap[i] for i in site.atoms]
                        newTemplate.virtualSites.append(newSite)

                # Build the lists of bonds and external bonds.

                atomMap = dict([(template.atoms[i], indexMap[i]) for i in indexMap])
                deletedBonds = [(atom1.name, atom2.name) for atom1, atom2 in self.deletedBonds if atom1.residue == index and atom2.residue == index]
                for atom1, atom2 in template.bonds:
                    a1 = template.atoms[atom1]
                    a2 = template.atoms[atom2]
                    if a1 in atomMap and a2 in atomMap and (a1.name, a2.name) not in deletedBonds and (a2.name, a1.name) not in deletedBonds:
                        newTemplate.addBond(atomMap[a1], atomMap[a2])
                deletedExternalBonds = [atom.name for atom in self.deletedExternalBonds if atom.residue == index]
                for atom in template.externalBonds:
                    if template.atoms[atom].name not in deletedExternalBonds:
                        newTemplate.addExternalBond(indexMap[atom])
                for atom1, atom2 in self.addedBonds:
                    if atom1.residue == index and atom2.residue == index:
                        newTemplate.addBondByName(atom1.name, atom2.name)
                    elif atom1.residue == index:
                        newTemplate.addExternalBondByName(atom1.name)
                    elif atom2.residue == index:
                        newTemplate.addExternalBondByName(atom2.name)
                for atom in self.addedExternalBonds:
                    newTemplate.addExternalBondByName(atom.name)
            return newTemplates

    class _PatchAtomData(object):
        """Inner class used to encapsulate data about an atom in a patch definition."""
        def __init__(self, description):
            if ':' in description:
                colonIndex = description.find(':')
                self.residue = int(description[:colonIndex])-1
                self.name = description[colonIndex+1:]
            else:
                self.residue = 0
                self.name = description

    class _AtomType(object):
        """Inner class used to record atom types and associated properties."""
        def __init__(self, name, atomClass, mass, element):
            self.name = name
            self.atomClass = atomClass
            self.mass = mass
            self.element = element

    class _AtomTypeParameters(object):
        """Inner class used to record parameter values for atom types."""
        def __init__(self, forcefield, forceName, atomTag, paramNames):
            self.ff = forcefield
            self.forceName = forceName
            self.atomTag = atomTag
            self.paramNames = paramNames
            self.paramsForType = {}
            self.extraParamsForType = {}
            self.params = []

        def registerAtom(self, parameters, expectedParams=None):
            if expectedParams is None:
                expectedParams = self.paramNames
            types = self.ff._findAtomTypes(parameters, 1)
            if None not in types:
                values = {}
                extraValues = {}
                for key in parameters:
                    if key in expectedParams:
                        values[key] = _convertParameterToNumber(parameters[key])
                    else:
                        extraValues[key] = parameters[key]
                if len(values) < len(expectedParams):
                    for key in expectedParams:
                        if key not in values:
                            raise ValueError('%s: No value specified for "%s"' % (self.forceName, key))
                for t in types[0]:
                    print("REGISTERING", values, t, extraValues)
                    self.paramsForType[t] = len(self.params)
                    self.params.append([values['charge'], values['sigma'], values['epsilon']])
                    self.extraParamsForType[t] = extraValues

            print("DONE")

        def parseDefinitions(self, element):
            """"Load the definitions from an XML element."""
            expectedParams = list(self.paramNames)
            excludedParams = [node.attrib['name'] for node in element.findall('UseAttributeFromResidue')]
            for param in excludedParams:
                assert 0
                if param not in expectedParams:
                    raise ValueError('%s: <UseAttributeFromResidue> specified an invalid attribute: %s' % (self.forceName, param))
                expectedParams.remove(param)
            for atom in element.findall(self.atomTag):
                print("atom", atom.attrib)
                for param in excludedParams:
                    if param in atom.attrib:
                        raise ValueError('%s: The attribute "%s" appeared in both <%s> and <UseAttributeFromResidue> tags' % (self.forceName, param, self.atomTag))
                self.registerAtom(atom.attrib, expectedParams)
            print("DONE PARSE DEF")

        def getAtomParameters(self, atom, data):
            """Get the parameter values for a particular atom."""
            t = data.atomType[atom]
            p = data.atomParameters[atom]
            if t in self.paramsForType:
                param_idx = self.paramsForType[t]
                return param_idx
                # result = [None]*len(self.paramNames)
                # for i, name in enumerate(self.paramNames):
                #     if name in values:
                #         result[i] = values[name]
                #     elif name in p:
                #         result[i] = p[name]
                #     else:
                #         raise ValueError('%s: No value specified for "%s"' % (self.forceName, name))
                # return result
            else:
                raise ValueError('%s: No parameters defined for atom type %s' % (self.forceName, t))

        def getExtraParameters(self, atom, data):
            """Get extra parameter values for an atom that appeared in the <Atom> tag but were not included in paramNames."""
            t = data.atomType[atom]
            if t in self.paramsForType:
                return self.extraParamsForType[t]
            else:
                raise ValueError('%s: No parameters defined for atom type %s' % (self.forceName, t))


    def _getResidueTemplateMatches(self, res, bondedToAtom, templateSignatures=None, ignoreExternalBonds=False):
        """Return the residue template matches, or None if none are found.

        Parameters
        ----------
        res : Topology.Residue
            The residue for which template matches are to be retrieved.
        bondedToAtom : list of set of int
            bondedToAtom[i] is the set of atoms bonded to atom index i

        Returns
        -------
        template : _ForceFieldTemplate
            The matching forcefield residue template, or None if no matches are found.
        matches : list
            a list specifying which atom of the template each atom of the residue
            corresponds to, or None if it does not match the template

        """
        template = None
        matches = None
        if templateSignatures is None:
            templateSignatures = self._templateSignatures
        signature = _createResidueSignature([atom.element for atom in res.atoms()])
        if signature in templateSignatures:
            allMatches = []
            for t in templateSignatures[signature]:
                match = compiled.matchResidueToTemplate(res, t, bondedToAtom, ignoreExternalBonds)
                if match is not None:
                    allMatches.append((t, match))
            if len(allMatches) == 1:
                template = allMatches[0][0]
                matches = allMatches[0][1]
            elif len(allMatches) > 1:
                raise Exception('Multiple matching templates found for residue %d (%s): %s.' % (res.index+1, res.name, ', '.join(match[0].name for match in allMatches)))
        return [template, matches]

    def _buildBondedToAtomList(self, topology):
        """Build a list of which atom indices are bonded to each atom.

        Parameters
        ----------
        topology : Topology
            The Topology whose bonds are to be indexed.

        Returns
        -------
        bondedToAtom : list of set of int
            bondedToAtom[index] is the set of atom indices bonded to atom `index`

        """
        bondedToAtom = []
        for atom in topology.atoms():
            bondedToAtom.append(set())
        for (atom1, atom2) in topology.bonds():
            bondedToAtom[atom1.index].add(atom2.index)
            bondedToAtom[atom2.index].add(atom1.index)
        return bondedToAtom

    def getUnmatchedResidues(self, topology):
        """Return a list of Residue objects from specified topology for which no forcefield templates are available.

        .. CAUTION:: This method is experimental, and its API is subject to change.

        Parameters
        ----------
        topology : Topology
            The Topology whose residues are to be checked against the forcefield residue templates.

        Returns
        -------
        unmatched_residues : list of Residue
            List of Residue objects from `topology` for which no forcefield residue templates are available.
            Note that multiple instances of the same residue appearing at different points in the topology may be returned.

        This method may be of use in generating missing residue templates or diagnosing parameterization failures.
        """
        # Find the template matching each residue, compiling a list of residues for which no templates are available.
        bondedToAtom = self._buildBondedToAtomList(topology)
        unmatched_residues = list() # list of unmatched residues
        for res in topology.residues():
            # Attempt to match one of the existing templates.
            [template, matches] = self._getResidueTemplateMatches(res, bondedToAtom)
            if matches is None:
                # No existing templates match.
                unmatched_residues.append(res)

        return unmatched_residues

    def getMatchingTemplates(self, topology, ignoreExternalBonds=False):
        """Return a list of forcefield residue templates matching residues in the specified topology.

        .. CAUTION:: This method is experimental, and its API is subject to change.

        Parameters
        ----------
        topology : Topology
            The Topology whose residues are to be checked against the forcefield residue templates.
        ignoreExternalBonds : bool=False
            If true, ignore external bonds when matching residues to templates.
        Returns
        -------
        templates : list of _TemplateData
            List of forcefield residue templates corresponding to residues in the topology.
            templates[index] is template corresponding to residue `index` in topology.residues()

        This method may be of use in debugging issues related to parameter assignment.
        """
        # Find the template matching each residue, compiling a list of residues for which no templates are available.
        bondedToAtom = self._buildBondedToAtomList(topology)
        templates = list() # list of templates matching the corresponding residues
        for res in topology.residues():
            # Attempt to match one of the existing templates.
            [template, matches] = self._getResidueTemplateMatches(res, bondedToAtom, ignoreExternalBonds=ignoreExternalBonds)
            # Raise an exception if we have found no templates that match.
            if matches is None:
                raise ValueError('No template found for residue %d (%s).  %s' % (res.index+1, res.name, _findMatchErrors(self, res)))
            else:
                templates.append(template)

        return templates

    def generateTemplatesForUnmatchedResidues(self, topology):
        """Generate forcefield residue templates for residues in specified topology for which no forcefield templates are available.

        .. CAUTION:: This method is experimental, and its API is subject to change.

        Parameters
        ----------
        topology : Topology
            The Topology whose residues are to be checked against the forcefield residue templates.

        Returns
        -------
        templates : list of _TemplateData
            List of forcefield residue templates corresponding to residues in `topology` for which no forcefield templates are currently available.
            Atom types will be set to `None`, but template name, atom names, elements, and connectivity will be taken from corresponding Residue objects.
        residues : list of Residue
            List of Residue objects that were used to generate the templates.
            `residues[index]` is the Residue that was used to generate the template `templates[index]`

        """
        # Get a non-unique list of unmatched residues.
        unmatched_residues = self.getUnmatchedResidues(topology)
        # Generate a unique list of unmatched residues by comparing fingerprints.
        bondedToAtom = self._buildBondedToAtomList(topology)
        unique_unmatched_residues = list() # list of unique unmatched Residue objects from topology
        templates = list() # corresponding _TemplateData templates
        signatures = set()
        for residue in unmatched_residues:
            signature = _createResidueSignature([ atom.element for atom in residue.atoms() ])
            template = _createResidueTemplate(residue)
            is_unique = True
            if signature in signatures:
                # Signature is the same as an existing residue; check connectivity.
                for check_residue in unique_unmatched_residues:
                    matches = compiled.matchResidueToTemplate(check_residue, template, bondedToAtom, False)
                    if matches is not None:
                        is_unique = False
            if is_unique:
                # Residue is unique.
                unique_unmatched_residues.append(residue)
                signatures.add(signature)
                templates.append(template)

        return [templates, unique_unmatched_residues]

    def createSystemData(self, topology, nonbondedMethod=NoCutoff, nonbondedCutoff=1.0*unit.nanometer,
                     constraints=None, rigidWater=True, removeCMMotion=True, hydrogenMass=None, residueTemplates=dict(),
                     ignoreExternalBonds=False, switchDistance=None, flexibleConstraints=False, **args):
        """Construct an OpenMM System representing a Topology with this force field.

        Parameters
        ----------
        topology : Topology
            The Topology for which to create a System
        nonbondedMethod : object=NoCutoff
            The method to use for nonbonded interactions.  Allowed values are
            NoCutoff, CutoffNonPeriodic, CutoffPeriodic, Ewald, PME, or LJPME.
        nonbondedCutoff : distance=1*nanometer
            The cutoff distance to use for nonbonded interactions
        constraints : object=None
            Specifies which bonds and angles should be implemented with constraints.
            Allowed values are None, HBonds, AllBonds, or HAngles.
        rigidWater : boolean=True
            If true, water molecules will be fully rigid regardless of the value
            passed for the constraints argument
        removeCMMotion : boolean=True
            If true, a CMMotionRemover will be added to the System
        hydrogenMass : mass=None
            The mass to use for hydrogen atoms bound to heavy atoms.  Any mass
            added to a hydrogen is subtracted from the heavy atom to keep
            their total mass the same.
        residueTemplates : dict=dict()
            Key: Topology Residue object
            Value: string, name of _TemplateData residue template object to use for (Key) residue.
            This allows user to specify which template to apply to particular Residues
            in the event that multiple matching templates are available (e.g Fe2+ and Fe3+
            templates in the ForceField for a monoatomic iron ion in the topology).
        ignoreExternalBonds : boolean=False
            If true, ignore external bonds when matching residues to templates.  This is
            useful when the Topology represents one piece of a larger molecule, so chains are
            not terminated properly.  This option can create ambiguities where multiple
            templates match the same residue.  If that happens, use the residueTemplates
            argument to specify which one to use.
        switchDistance : float=None
            The distance at which the potential energy switching function is turned on for
            Lennard-Jones interactions. If this is None, no switching function will be used.
        flexibleConstraints : boolean=False
            If True, parameters for constrained degrees of freedom will be added to the System
        args
            Arbitrary additional keyword arguments may also be specified.
            This allows extra parameters to be specified that are specific to
            particular force fields.

        Returns
        -------
        system
            the newly created System
        """
        args['switchDistance'] = switchDistance
        args['flexibleConstraints'] = flexibleConstraints
        data = ForceField._SystemData()
        data.atoms = list(topology.atoms())
        for atom in data.atoms:
            data.excludeAtomWith.append([])

        # Make a list of all bonds

        for bond in topology.bonds():
            data.bonds.append(ForceField._BondData(bond[0].index, bond[1].index))

        # Record which atoms are bonded to each other atom

        bondedToAtom = []
        for i in range(len(data.atoms)):
            bondedToAtom.append(set())
            data.atomBonds.append([])
        for i in range(len(data.bonds)):
            bond = data.bonds[i]
            bondedToAtom[bond.atom1].add(bond.atom2)
            bondedToAtom[bond.atom2].add(bond.atom1)
            data.atomBonds[bond.atom1].append(i)
            data.atomBonds[bond.atom2].append(i)

        # Find the template matching each residue and assign atom types.

        unmatchedResidues = []
        for chain in topology.chains():
            for res in chain.residues():
                if res in residueTemplates:
                    tname = residueTemplates[res]
                    template = self._templates[tname]
                    matches = compiled.matchResidueToTemplate(res, template, bondedToAtom, ignoreExternalBonds)
                    if matches is None:
                        raise Exception('User-supplied template %s does not match the residue %d (%s)' % (tname, res.index+1, res.name))
                else:
                    # Attempt to match one of the existing templates.
                    [template, matches] = self._getResidueTemplateMatches(res, bondedToAtom, ignoreExternalBonds=ignoreExternalBonds)
                if matches is None:
                    unmatchedResidues.append(res)
                else:
                    data.recordMatchedAtomParameters(res, template, matches)


        # Try to apply patches to find matches for any unmatched residues.

        if len(unmatchedResidues) > 0:
            unmatchedResidues = _applyPatchesToMatchResidues(self, data, unmatchedResidues, bondedToAtom, ignoreExternalBonds)

        # If we still haven't found a match for a residue, attempt to use residue template generators to create
        # new templates (and potentially atom types/parameters).

        for res in unmatchedResidues:
            # A template might have been generated on an earlier iteration of this loop.
            [template, matches] = self._getResidueTemplateMatches(res, bondedToAtom, ignoreExternalBonds=ignoreExternalBonds)
            if matches is None:
                # Try all generators.
                for generator in self._templateGenerators:
                    assert 0
                    # print("TRYING GENERATOR", generator)
                    if generator(self, res):
                        # This generator has registered a new residue template that should match.
                        [template, matches] = self._getResidueTemplateMatches(res, bondedToAtom, ignoreExternalBonds=ignoreExternalBonds)
                        if matches is None:
                            # Something went wrong because the generated template does not match the residue signature.
                            raise Exception('The residue handler %s indicated it had correctly parameterized residue %s, but the generated template did not match the residue signature.' % (generator.__class__.__name__, str(res)))
                        else:
                            # We successfully generated a residue template.  Break out of the for loop.
                            break
            if matches is None:
                raise ValueError('No template found for residue %d (%s).  %s' % (res.index+1, res.name, _findMatchErrors(self, res)))
            else:
                data.recordMatchedAtomParameters(res, template, matches)

        # Create the System and add atoms

        sys = mm.System()
        for atom in topology.atoms():
            # Look up the atom type name, returning a helpful error message if it cannot be found.
            if atom not in data.atomType:
                raise Exception("Could not identify atom type for atom '%s'." % str(atom))
            typename = data.atomType[atom]

            # Look up the type name in the list of registered atom types, returning a helpful error message if it cannot be found.
            if typename not in self._atomTypes:
                msg  = "Could not find typename '%s' for atom '%s' in list of known atom types.\n" % (typename, str(atom))
                msg += "Known atom types are: %s" % str(self._atomTypes.keys())
                raise Exception(msg)

            # Add the particle to the OpenMM system.
            mass = self._atomTypes[typename].mass
            sys.addParticle(mass)

        # Adjust hydrogen masses if requested.

        if hydrogenMass is not None:
            if not unit.is_quantity(hydrogenMass):
                hydrogenMass *= unit.dalton
            for atom1, atom2 in topology.bonds():
                if atom1.element is elem.hydrogen:
                    (atom1, atom2) = (atom2, atom1)
                if atom2.element is elem.hydrogen and atom1.element not in (elem.hydrogen, None):
                    transferMass = hydrogenMass-sys.getParticleMass(atom2.index)
                    sys.setParticleMass(atom2.index, hydrogenMass)
                    sys.setParticleMass(atom1.index, sys.getParticleMass(atom1.index)-transferMass)

        # Set periodic boundary conditions.

        boxVectors = topology.getPeriodicBoxVectors()
        if boxVectors is not None:
            sys.setDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2])
        elif nonbondedMethod not in [NoCutoff, CutoffNonPeriodic]:
            raise ValueError('Requested periodic boundary conditions for a Topology that does not specify periodic box dimensions')

        # Make a list of all unique angles

        uniqueAngles = set()
        for bond in data.bonds:
            for atom in bondedToAtom[bond.atom1]:
                if atom != bond.atom2:
                    if atom < bond.atom2:
                        uniqueAngles.add((atom, bond.atom1, bond.atom2))
                    else:
                        uniqueAngles.add((bond.atom2, bond.atom1, atom))
            for atom in bondedToAtom[bond.atom2]:
                if atom != bond.atom1:
                    if atom > bond.atom1:
                        uniqueAngles.add((bond.atom1, bond.atom2, atom))
                    else:
                        uniqueAngles.add((atom, bond.atom2, bond.atom1))
        data.angles = sorted(list(uniqueAngles))

        # Make a list of all unique proper torsions

        uniquePropers = set()
        for angle in data.angles:
            for atom in bondedToAtom[angle[0]]:
                if atom not in angle:
                    if atom < angle[2]:
                        uniquePropers.add((atom, angle[0], angle[1], angle[2]))
                    else:
                        uniquePropers.add((angle[2], angle[1], angle[0], atom))
            for atom in bondedToAtom[angle[2]]:
                if atom not in angle:
                    if atom > angle[0]:
                        uniquePropers.add((angle[0], angle[1], angle[2], atom))
                    else:
                        uniquePropers.add((atom, angle[2], angle[1], angle[0]))
        data.propers = sorted(list(uniquePropers))

        # Make a list of all unique improper torsions

        for atom in range(len(bondedToAtom)):
            bondedTo = bondedToAtom[atom]
            if len(bondedTo) > 2:
                for subset in itertools.combinations(bondedTo, 3):
                    data.impropers.append((atom, subset[0], subset[1], subset[2]))

        # Identify bonds that should be implemented with constraints

        if constraints == AllBonds or constraints == HAngles:
            for bond in data.bonds:
                bond.isConstrained = True
        elif constraints == HBonds:
            for bond in data.bonds:
                atom1 = data.atoms[bond.atom1]
                atom2 = data.atoms[bond.atom2]
                bond.isConstrained = atom1.element is elem.hydrogen or atom2.element is elem.hydrogen
        if rigidWater:
            for bond in data.bonds:
                atom1 = data.atoms[bond.atom1]
                atom2 = data.atoms[bond.atom2]
                if atom1.residue.name == 'HOH' and atom2.residue.name == 'HOH':
                    bond.isConstrained = True

        # Identify angles that should be implemented with constraints

        if constraints == HAngles:
            for angle in data.angles:
                atom1 = data.atoms[angle[0]]
                atom2 = data.atoms[angle[1]]
                atom3 = data.atoms[angle[2]]
                numH = 0
                if atom1.element is elem.hydrogen:
                    numH += 1
                if atom3.element is elem.hydrogen:
                    numH += 1
                data.isAngleConstrained.append(numH == 2 or (numH == 1 and atom2.element is elem.oxygen))
        else:
            data.isAngleConstrained = len(data.angles)*[False]
        if rigidWater:
            for i in range(len(data.angles)):
                angle = data.angles[i]
                atom1 = data.atoms[angle[0]]
                atom2 = data.atoms[angle[1]]
                atom3 = data.atoms[angle[2]]
                if atom1.residue.name == 'HOH' and atom2.residue.name == 'HOH' and atom3.residue.name == 'HOH':
                    data.isAngleConstrained[i] = True

        # Add virtual sites

        for atom in data.virtualSites:
            (site, atoms, excludeWith) = data.virtualSites[atom]
            index = atom.index
            data.excludeAtomWith[excludeWith].append(index)
            if site.type == 'average2':
                sys.setVirtualSite(index, mm.TwoParticleAverageSite(atoms[0], atoms[1], site.weights[0], site.weights[1]))
            elif site.type == 'average3':
                sys.setVirtualSite(index, mm.ThreeParticleAverageSite(atoms[0], atoms[1], atoms[2], site.weights[0], site.weights[1], site.weights[2]))
            elif site.type == 'outOfPlane':
                sys.setVirtualSite(index, mm.OutOfPlaneSite(atoms[0], atoms[1], atoms[2], site.weights[0], site.weights[1], site.weights[2]))
            elif site.type == 'localCoords':
                sys.setVirtualSite(index, mm.LocalCoordinatesSite(atoms, site.originWeights, site.xWeights, site.yWeights, site.localPos))

        return data

        # Add forces to the System

        for force in self._forces:
            force.createForce(sys, data, nonbondedMethod, nonbondedCutoff, args)
        if removeCMMotion:
            sys.addForce(mm.CMMotionRemover())

        # Let force generators do postprocessing

        for force in self._forces:
            if 'postprocessSystem' in dir(force):
                force.postprocessSystem(sys, data, args)

        # Execute scripts found in the XML files.

        for script in self._scripts:
            exec(script, locals())
        return sys


def _findBondsForExclusions(data, sys):
    """Create a list of bonds to use when identifying exclusions."""
    bondIndices = []
    for bond in data.bonds:
        bondIndices.append((bond.atom1, bond.atom2))

    # If a virtual site does *not* share exclusions with another atom, add a bond between it and its first parent atom.

    for i in range(sys.getNumParticles()):
        if sys.isVirtualSite(i):
            (site, atoms, excludeWith) = data.virtualSites[data.atoms[i]]
            if excludeWith is None:
                bondIndices.append((i, site.getParticle(0)))

    # Certain particles, such as lone pairs and Drude particles, share exclusions with a parent atom.
    # If the parent atom does not interact with an atom, the child particle does not either.

    for atom1, atom2 in bondIndices:
        for child1 in data.excludeAtomWith[atom1]:
            bondIndices.append((child1, atom2))
            for child2 in data.excludeAtomWith[atom2]:
                bondIndices.append((child1, child2))
        for child2 in data.excludeAtomWith[atom2]:
            bondIndices.append((atom1, child2))
    for atom in data.atoms:
        for child in data.excludeAtomWith[atom.index]:
            bondIndices.append((child, atom.index))
    return bondIndices

def _findExclusions(bondIndices, maxSeparation, numAtoms):
    """Identify pairs of atoms in the same molecule separated by no more than maxSeparation bonds."""
    bondedTo = [set() for i in range(numAtoms)]
    for i, j in bondIndices:
        bondedTo[i].add(j)
        bondedTo[j].add(i)

    # Identify all neighbors of each atom with each separation.

    bondedWithSeparation = [bondedTo]
    for i in range(maxSeparation-1):
        lastBonds = bondedWithSeparation[-1]
        newBonds = deepcopy(lastBonds)
        for atom in range(numAtoms):
            for a1 in lastBonds[atom]:
                for a2 in bondedTo[a1]:
                    newBonds[atom].add(a2)
        bondedWithSeparation.append(newBonds)

    # Build the list of pairs.

    pairs = []
    for atom in range(numAtoms):
        for otherAtom in bondedWithSeparation[-1][atom]:
            if otherAtom > atom:
                # Determine the minimum number of bonds between them.
                sep = maxSeparation
                for i in reversed(range(maxSeparation-1)):
                    if otherAtom in bondedWithSeparation[i][atom]:
                        sep -= 1
                    else:
                        break
                pairs.append((atom, otherAtom, sep))
    return pairs


def _findGroups(bondedTo):
    """Given bonds that connect atoms, identify the connected groups."""
    atomGroup = [None]*len(bondedTo)
    numGroups = 0
    for i in range(len(bondedTo)):
        if atomGroup[i] is None:
            # Start a new group.

            atomStack = [i]
            neighborStack = [0]
            group = numGroups
            numGroups += 1

            # Recursively tag all the bonded atoms.

            while len(atomStack) > 0:
                atom = atomStack[-1]
                atomGroup[atom] = group
                while neighborStack[-1] < len(bondedTo[atom]) and atomGroup[bondedTo[atom][neighborStack[-1]]] is not None:
                    neighborStack[-1] += 1
                if neighborStack[-1] < len(bondedTo[atom]):
                    atomStack.append(bondedTo[atom][neighborStack[-1]])
                    neighborStack.append(0)
                else:
                    atomStack.pop()
                    neighborStack.pop()
    return atomGroup

def _countResidueAtoms(elements):
    """Count the number of atoms of each element in a residue."""
    counts = {}
    for element in elements:
        if element in counts:
            counts[element] += 1
        else:
            counts[element] = 1
    return counts


def _createResidueSignature(elements):
    """Create a signature for a residue based on the elements of the atoms it contains."""
    counts = _countResidueAtoms(elements)
    sig = []
    for c in counts:
        if c is not None:
            sig.append((c, counts[c]))
    sig.sort(key=lambda x: -x[0].mass)

    # Convert it to a string.

    s = ''
    for element, count in sig:
        s += element.symbol+str(count)
    return s


def _applyPatchesToMatchResidues(forcefield, data, residues, bondedToAtom, ignoreExternalBonds):
    """Try to apply patches to find matches for residues."""
    # Start by creating all templates than can be created by applying a combination of one-residue patches
    # to a single template.  The number of these is usually not too large, and they often cover a large fraction
    # of residues.

    patchedTemplateSignatures = {}
    patchedTemplates = {}
    for name, template in forcefield._templates.items():
        if name in forcefield._templatePatches:
            patches = [forcefield._patches[patchName] for patchName, patchResidueIndex in forcefield._templatePatches[name] if forcefield._patches[patchName].numResidues == 1]
            if len(patches) > 0:
                newTemplates = []
                patchedTemplates[name] = newTemplates
                _generatePatchedSingleResidueTemplates(template, patches, 0, newTemplates, set())
                for patchedTemplate in newTemplates:
                    signature = _createResidueSignature([atom.element for atom in patchedTemplate.atoms])
                    if signature in patchedTemplateSignatures:
                        patchedTemplateSignatures[signature].append(patchedTemplate)
                    else:
                        patchedTemplateSignatures[signature] = [patchedTemplate]

    # Now see if any of those templates matches any of the residues.

    unmatchedResidues = []
    for res in residues:
        [template, matches] = forcefield._getResidueTemplateMatches(res, bondedToAtom, patchedTemplateSignatures, ignoreExternalBonds)
        if matches is None:
            unmatchedResidues.append(res)
        else:
            data.recordMatchedAtomParameters(res, template, matches)
    if len(unmatchedResidues) == 0:
        return []

    # We need to consider multi-residue patches.  This can easily lead to a combinatorial explosion, so we make a simplifying
    # assumption: that no residue is affected by more than one multi-residue patch (in addition to any number of single-residue
    # patches).  Record all multi-residue patches, and the templates they can be applied to.

    patches = {}
    maxPatchSize = 0
    for patch in forcefield._patches.values():
        if patch.numResidues > 1:
            patches[patch.name] = [[] for i in range(patch.numResidues)]
            maxPatchSize = max(maxPatchSize, patch.numResidues)
    if maxPatchSize == 0:
        return unmatchedResidues # There aren't any multi-residue patches
    for templateName in forcefield._templatePatches:
        for patchName, patchResidueIndex in forcefield._templatePatches[templateName]:
            if patchName in patches:
                # The patch should accept this template, *and* all patched versions of it generated above.
                patches[patchName][patchResidueIndex].append(forcefield._templates[templateName])
                if templateName in patchedTemplates:
                    patches[patchName][patchResidueIndex] += patchedTemplates[templateName]

    # Record which unmatched residues are bonded to each other.

    bonds = set()
    topology = residues[0].chain.topology
    for atom1, atom2 in topology.bonds():
        if atom1.residue != atom2.residue:
            res1 = atom1.residue
            res2 = atom2.residue
            if res1 in unmatchedResidues and res2 in unmatchedResidues:
                bond = tuple(sorted((res1, res2), key=lambda x: x.index))
                if bond not in bonds:
                    bonds.add(bond)

    # Identify clusters of unmatched residues that are all bonded to each other.  These are the ones we'll
    # try to apply multi-residue patches to.

    clusterSize = 2
    clusters = bonds
    while clusterSize <= maxPatchSize:
        # Try to apply patches to clusters of this size.

        for patchName in patches:
            patch = forcefield._patches[patchName]
            if patch.numResidues == clusterSize:
                matchedClusters = _matchToMultiResiduePatchedTemplates(data, clusters, patch, patches[patchName], bondedToAtom, ignoreExternalBonds)
                for cluster in matchedClusters:
                    for residue in cluster:
                        unmatchedResidues.remove(residue)
                bonds = set(bond for bond in bonds if bond[0] in unmatchedResidues and bond[1] in unmatchedResidues)

        # Now extend the clusters to find ones of the next size up.

        largerClusters = set()
        for cluster in clusters:
            for bond in bonds:
                if bond[0] in cluster and bond[1] not in cluster:
                    newCluster = tuple(sorted(cluster+(bond[1],), key=lambda x: x.index))
                    largerClusters.add(newCluster)
                elif bond[1] in cluster and bond[0] not in cluster:
                    newCluster = tuple(sorted(cluster+(bond[0],), key=lambda x: x.index))
                    largerClusters.add(newCluster)
        if len(largerClusters) == 0:
            # There are no clusters of this size or larger
            break
        clusters = largerClusters
        clusterSize += 1

    return unmatchedResidues


def _generatePatchedSingleResidueTemplates(template, patches, index, newTemplates, alteredAtoms):
    """Apply all possible combinations of a set of single-residue patches to a template."""
    try:
        if len(alteredAtoms.intersection(patches[index].allAtomNames)) > 0:
            # This patch would alter an atom that another patch has already altered,
            # so don't apply it.
            patchedTemplate = None
        else:
            patchedTemplate = patches[index].createPatchedTemplates([template])[0]
            newTemplates.append(patchedTemplate)
    except:
        # This probably means the patch is inconsistent with another one that has already been applied,
        # so just ignore it.
        patchedTemplate = None

    # Call this function recursively to generate combinations of patches.

    if index+1 < len(patches):
        _generatePatchedSingleResidueTemplates(template, patches, index+1, newTemplates, alteredAtoms)
        if patchedTemplate is not None:
            newAlteredAtoms = alteredAtoms.union(patches[index].allAtomNames)
            _generatePatchedSingleResidueTemplates(patchedTemplate, patches, index+1, newTemplates, newAlteredAtoms)


def _matchToMultiResiduePatchedTemplates(data, clusters, patch, residueTemplates, bondedToAtom, ignoreExternalBonds):
    """Apply a multi-residue patch to templates, then try to match them against clusters of residues."""
    matchedClusters = []
    selectedTemplates = [None]*patch.numResidues
    _applyMultiResiduePatch(data, clusters, patch, residueTemplates, selectedTemplates, 0, matchedClusters, bondedToAtom, ignoreExternalBonds)
    return matchedClusters


def _applyMultiResiduePatch(data, clusters, patch, candidateTemplates, selectedTemplates, index, matchedClusters, bondedToAtom, ignoreExternalBonds):
    """This is called recursively to apply a multi-residue patch to all possible combinations of templates."""

    if index < patch.numResidues:
        for template in candidateTemplates[index]:
            selectedTemplates[index] = template
            _applyMultiResiduePatch(data, clusters, patch, candidateTemplates, selectedTemplates, index+1, matchedClusters, bondedToAtom, ignoreExternalBonds)
    else:
        # We're at the deepest level of the recursion.  We've selected a template for each residue, so apply the patch,
        # then try to match it against clusters.

        try:
            patchedTemplates = patch.createPatchedTemplates(selectedTemplates)
        except:
            # This probably means the patch is inconsistent with another one that has already been applied,
            # so just ignore it.
            return
        newlyMatchedClusters = []
        for cluster in clusters:
            for residues in itertools.permutations(cluster):
                residueMatches = []
                for residue, template in zip(residues, patchedTemplates):
                    matches = compiled.matchResidueToTemplate(residue, template, bondedToAtom, ignoreExternalBonds)
                    if matches is None:
                        residueMatches = None
                        break
                    else:
                        residueMatches.append(matches)
                if residueMatches is not None:
                    # Each residue individually matches.  Now make sure they're bonded in the correct way.

                    bondsMatch = True
                    for a1, a2 in patch.addedBonds:
                        res1 = a1.residue
                        res2 = a2.residue
                        if res1 != res2:
                            # The patch adds a bond between residues.  Make sure that bond exists.

                            atoms1 = patchedTemplates[res1].atoms
                            atoms2 = patchedTemplates[res2].atoms
                            index1 = next(i for i in range(len(atoms1)) if atoms1[residueMatches[res1][i]].name == a1.name)
                            index2 = next(i for i in range(len(atoms2)) if atoms2[residueMatches[res2][i]].name == a2.name)
                            atom1 = list(residues[res1].atoms())[index1]
                            atom2 = list(residues[res2].atoms())[index2]
                            bondsMatch &= atom2.index in bondedToAtom[atom1.index]
                    if bondsMatch:
                        # We successfully matched the template to the residues.  Record the parameters.

                        for i in range(patch.numResidues):
                            data.recordMatchedAtomParameters(residues[i], patchedTemplates[i], residueMatches[i])
                        newlyMatchedClusters.append(cluster)
                        break

        # Record which clusters were successfully matched.

        matchedClusters += newlyMatchedClusters
        for cluster in newlyMatchedClusters:
            clusters.remove(cluster)


def _findMatchErrors(forcefield, res):
    """Try to guess why a residue failed to match any template and return an error message."""
    residueCounts = _countResidueAtoms([atom.element for atom in res.atoms()])
    numResidueAtoms = sum(residueCounts.values())
    numResidueHeavyAtoms = sum(residueCounts[element] for element in residueCounts if element not in (None, elem.hydrogen))

    # Loop over templates and see how closely each one might match.

    bestMatchName = None
    numBestMatchAtoms = 3*numResidueAtoms
    numBestMatchHeavyAtoms = 2*numResidueHeavyAtoms
    for templateName in forcefield._templates:
        template = forcefield._templates[templateName]
        templateCounts = _countResidueAtoms([atom.element for atom in template.atoms])

        # Does the residue have any atoms that clearly aren't in the template?

        if any(element not in templateCounts or templateCounts[element] < residueCounts[element] for element in residueCounts):
            continue

        # If there are too many missing atoms, discard this template.

        numTemplateAtoms = sum(templateCounts.values())
        numTemplateHeavyAtoms = sum(templateCounts[element] for element in templateCounts if element not in (None, elem.hydrogen))
        if numTemplateAtoms > numBestMatchAtoms:
            continue
        if numTemplateHeavyAtoms > numBestMatchHeavyAtoms:
            continue

        # If this template has the same number of missing atoms as our previous best one, look at the name
        # to decide which one to use.

        if numTemplateAtoms == numBestMatchAtoms:
            if bestMatchName == res.name or res.name not in templateName:
                continue

        # Accept this as our new best match.

        bestMatchName = templateName
        numBestMatchAtoms = numTemplateAtoms
        numBestMatchHeavyAtoms = numTemplateHeavyAtoms
        numBestMatchExtraParticles = len([atom for atom in template.atoms if atom.element is None])

    # Return an appropriate error message.

    if numBestMatchAtoms == numResidueAtoms:
        chainResidues = list(res.chain.residues())
        if len(chainResidues) > 1 and (res == chainResidues[0] or res == chainResidues[-1]):
            return 'The set of atoms matches %s, but the bonds are different.  Perhaps the chain is missing a terminal group?' % bestMatchName
        return 'The set of atoms matches %s, but the bonds are different.' % bestMatchName
    if bestMatchName is not None:
        if numBestMatchHeavyAtoms == numResidueHeavyAtoms:
            numResidueExtraParticles = len([atom for atom in res.atoms() if atom.element is None])
            if numResidueExtraParticles == 0 and numBestMatchExtraParticles == 0:
                return 'The set of atoms is similar to %s, but it is missing %d hydrogen atoms.' % (bestMatchName, numBestMatchAtoms-numResidueAtoms)
            if numBestMatchExtraParticles-numResidueExtraParticles == numBestMatchAtoms-numResidueAtoms:
                return 'The set of atoms is similar to %s, but it is missing %d extra particles.  You can add them with Modeller.addExtraParticles().' % (bestMatchName, numBestMatchAtoms-numResidueAtoms)
        return 'The set of atoms is similar to %s, but it is missing %d atoms.' % (bestMatchName, numBestMatchAtoms-numResidueAtoms)
    return 'This might mean your input topology is missing some atoms or bonds, or possibly that you are using the wrong force field.'

def _createResidueTemplate(residue):
    """Create a _TemplateData template from a Residue object.

    Parameters
    ----------
    residue : Residue
        The Residue from which the template is to be constructed.

    Returns
    -------
    template : _TemplateData
        The residue template, with atom types set to None.

    This method may be useful in creating new residue templates for residues without templates defined by the ForceField.

    """
    template = ForceField._TemplateData(residue.name)
    for atom in residue.atoms():
        template.addAtom(ForceField._TemplateAtomData(atom.name, None, atom.element))
    for (atom1,atom2) in residue.internal_bonds():
        template.addBondByName(atom1.name, atom2.name)
    residue_atoms = [ atom for atom in residue.atoms() ]
    for (atom1,atom2) in residue.external_bonds():
        if atom1 in residue_atoms:
            template.addExternalBondByName(atom1.name)
        elif atom2 in residue_atoms:
            template.addExternalBondByName(atom2.name)
    return template

def _matchImproper(data, torsion, generator):
    type1 = data.atomType[data.atoms[torsion[0]]]
    type2 = data.atomType[data.atoms[torsion[1]]]
    type3 = data.atomType[data.atoms[torsion[2]]]
    type4 = data.atomType[data.atoms[torsion[3]]]
    wildcard = generator.ff._atomClasses['']
    match = None
    for impr_idx, tordef in enumerate(generator.improper):
        types1 = tordef.types1
        types2 = tordef.types2
        types3 = tordef.types3
        types4 = tordef.types4
        hasWildcard = (wildcard in (types1, types2, types3, types4))
        if match is not None and hasWildcard:
            # Prefer specific definitions over ones with wildcards
            continue
        if type1 in types1:
            for (t2, t3, t4) in itertools.permutations(((type2, 1), (type3, 2), (type4, 3))):
                if t2[0] in types2 and t3[0] in types3 and t4[0] in types4:
                    if tordef.ordering == 'default':
                        # Workaround to be more consistent with AMBER.  It uses wildcards to define most of its
                        # impropers, which leaves the ordering ambiguous.  It then follows some bizarre rules
                        # to pick the order.
                        a1 = torsion[t2[1]]
                        a2 = torsion[t3[1]]
                        e1 = data.atoms[a1].element
                        e2 = data.atoms[a2].element
                        if e1 == e2 and a1 > a2:
                            (a1, a2) = (a2, a1)
                        elif e1 != elem.carbon and (e2 == elem.carbon or e1.mass < e2.mass):
                            (a1, a2) = (a2, a1)
                        match = (a1, a2, torsion[0], torsion[t4[1]], tordef)
                        break
                    elif tordef.ordering == 'charmm':
                        if hasWildcard:
                            # Workaround to be more consistent with AMBER.  It uses wildcards to define most of its
                            # impropers, which leaves the ordering ambiguous.  It then follows some bizarre rules
                            # to pick the order.
                            a1 = torsion[t2[1]]
                            a2 = torsion[t3[1]]
                            e1 = data.atoms[a1].element
                            e2 = data.atoms[a2].element
                            if e1 == e2 and a1 > a2:
                                (a1, a2) = (a2, a1)
                            elif e1 != elem.carbon and (e2 == elem.carbon or e1.mass < e2.mass):
                                (a1, a2) = (a2, a1)
                            match = (a1, a2, torsion[0], torsion[t4[1]], tordef)
                        else:
                            # There are no wildcards, so the order is unambiguous.
                            match = (torsion[0], torsion[t2[1]], torsion[t3[1]], torsion[t4[1]], tordef)
                        break
                    elif tordef.ordering == 'amber':
                        # topology atom indexes
                        a2 = torsion[t2[1]]
                        a3 = torsion[t3[1]]
                        a4 = torsion[t4[1]]
                        # residue indexes
                        r2 = data.atoms[a2].residue.index
                        r3 = data.atoms[a3].residue.index
                        r4 = data.atoms[a4].residue.index
                        # template atom indexes
                        ta2 = data.atomTemplateIndexes[data.atoms[a2]]
                        ta3 = data.atomTemplateIndexes[data.atoms[a3]]
                        ta4 = data.atomTemplateIndexes[data.atoms[a4]]
                        # elements
                        e2 = data.atoms[a2].element
                        e3 = data.atoms[a3].element
                        e4 = data.atoms[a4].element
                        if not hasWildcard:
                            if t2[0] == t4[0] and (r2 > r4 or (r2 == r4 and ta2 > ta4)):
                                (a2, a4) = (a4, a2)
                                r2 = data.atoms[a2].residue.index
                                r4 = data.atoms[a4].residue.index
                                ta2 = data.atomTemplateIndexes[data.atoms[a2]]
                                ta4 = data.atomTemplateIndexes[data.atoms[a4]]
                            if t3[0] == t4[0] and (r3 > r4 or (r3 == r4 and ta3 > ta4)):
                                (a3, a4) = (a4, a3)
                                r3 = data.atoms[a3].residue.index
                                r4 = data.atoms[a4].residue.index
                                ta3 = data.atomTemplateIndexes[data.atoms[a3]]
                                ta4 = data.atomTemplateIndexes[data.atoms[a4]]
                            if t2[0] == t3[0] and (r2 > r3 or (r2 == r3 and ta2 > ta3)):
                                (a2, a3) = (a3, a2)
                        else:
                            if e2 == e4 and (r2 > r4 or (r2 == r4 and ta2 > ta4)):
                                (a2, a4) = (a4, a2)
                                r2 = data.atoms[a2].residue.index
                                r4 = data.atoms[a4].residue.index
                                ta2 = data.atomTemplateIndexes[data.atoms[a2]]
                                ta4 = data.atomTemplateIndexes[data.atoms[a4]]
                            if e3 == e4 and (r3 > r4 or (r3 == r4 and ta3 > ta4)):
                                (a3, a4) = (a4, a3)
                                r3 = data.atoms[a3].residue.index
                                r4 = data.atoms[a4].residue.index
                                ta3 = data.atomTemplateIndexes[data.atoms[a3]]
                                ta4 = data.atomTemplateIndexes[data.atoms[a4]]
                            if r2 > r3 or (r2 == r3 and ta2 > ta3):
                                (a2, a3) = (a3, a2)
                        match = (a2, a3, torsion[0], a4, tordef)
                        break
    return match, impr_idx


# The following classes are generators that know how to create Force subclasses and add them to a System that is being
# created.  Each generator class must define two methods: 1) a static method that takes an etree Element and a ForceField,
# and returns the corresponding generator object; 2) a createForce() method that constructs the Force object and adds it
# to the System.  The static method should be added to the parsers map.

## @private
class HarmonicBondGenerator(object):
    """A HarmonicBondGenerator constructs a HarmonicBondForce."""

    def __init__(self, forcefield):
        self.ff = forcefield
        self.bondsForAtomType = defaultdict(set)
        self.types1 = []
        self.types2 = []
        self.params = [] # (ytz): new!
        self.length = []
        self.k = []

    def registerBond(self, parameters):
        types = self.ff._findAtomTypes(parameters, 2)
        if None not in types:
            index = len(self.types1)
            self.types1.append(types[0])
            self.types2.append(types[1])
            for t in types[0]:
                self.bondsForAtomType[t].add(index)
            for t in types[1]:
                self.bondsForAtomType[t].add(index)

            b = _convertParameterToNumber(parameters['length'])
            k = _convertParameterToNumber(parameters['k'])

            self.params.append((k, b))
            self.length.append(b)
            self.k.append(k)

    @staticmethod
    def parseElement(element, ff):
        existing = [f for f in ff._forces if isinstance(f, HarmonicBondGenerator)]
        if len(existing) == 0:
            generator = HarmonicBondGenerator(ff)
            ff.registerGenerator(generator)
        else:
            generator = existing[0]
        for bond in element.findall('Bond'):
            # print("registering", bond)
            generator.registerBond(bond.attrib)

    def parameterize(self, params, data):
        bond_param_idxs = []
        bond_idxs = []

        for bond in data.bonds:
            type1 = data.atomType[data.atoms[bond.atom1]]
            type2 = data.atomType[data.atoms[bond.atom2]]
            for p_idx, i in enumerate(self.bondsForAtomType[type1]):
                types1 = self.types1[i]
                types2 = self.types2[i]
                if (type1 in types1 and type2 in types2) or (type1 in types2 and type2 in types1):
                    bond.length = self.length[i]
                    bond_idxs.append((bond.atom1, bond.atom2))
                    bond_param_idxs.append(p_idx)
                    break

        bond_params = params[onp.array(bond_param_idxs)]
        bond_idxs = onp.array(bond_idxs, dtype=np.int32)

        return bond_params, bond_idxs

    # def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
    #     existing = [sys.getForce(i) for i in range(sys.getNumForces())]
    #     existing = [f for f in existing if type(f) == mm.HarmonicBondForce]
    #     if len(existing) == 0:
    #         force = mm.HarmonicBondForce()
    #         sys.addForce(force)
    #     else:
    #         force = existing[0]
    #     for bond in data.bonds:
    #         type1 = data.atomType[data.atoms[bond.atom1]]
    #         type2 = data.atomType[data.atoms[bond.atom2]]
    #         for i in self.bondsForAtomType[type1]:
    #             types1 = self.types1[i]
    #             types2 = self.types2[i]
    #             if (type1 in types1 and type2 in types2) or (type1 in types2 and type2 in types1):
    #                 bond.length = self.length[i]
    #                 if bond.isConstrained:
    #                     data.addConstraint(sys, bond.atom1, bond.atom2, self.length[i])
    #                 if self.k[i] != 0:
    #                     # flexibleConstraints allows us to add parameters even if the DOF is
    #                     # constrained
    #                     if not bond.isConstrained or args.get('flexibleConstraints', False):
    #                         force.addBond(bond.atom1, bond.atom2, self.length[i], self.k[i])
    #                 break

parsers["HarmonicBondForce"] = HarmonicBondGenerator.parseElement


## @private
class HarmonicAngleGenerator(object):
    """A HarmonicAngleGenerator constructs a HarmonicAngleForce."""

    def __init__(self, forcefield):
        self.ff = forcefield
        self.anglesForAtom2Type = defaultdict(list)
        self.types1 = []
        self.types2 = []
        self.types3 = []
        self.angle = []
        self.k = []
        self.params = []

    def registerAngle(self, parameters):
        types = self.ff._findAtomTypes(parameters, 3)
        if None not in types:
            index = len(self.types1)
            self.types1.append(types[0])
            self.types2.append(types[1])
            self.types3.append(types[2])
            for t in types[1]:
                self.anglesForAtom2Type[t].append(index)

            angle = _convertParameterToNumber(parameters['angle'])
            k = _convertParameterToNumber(parameters['k'])

            self.angle.append(angle)
            self.k.append(k)
            self.params.append((k, angle))


    @staticmethod
    def parseElement(element, ff):
        existing = [f for f in ff._forces if isinstance(f, HarmonicAngleGenerator)]
        if len(existing) == 0:
            generator = HarmonicAngleGenerator(ff)
            ff.registerGenerator(generator)
        else:
            generator = existing[0]
        for angle in element.findall('Angle'):
            generator.registerAngle(angle.attrib)

    def parameterize(self, params, data):
        angle_param_idxs = []
        angle_idxs = []

        for (angle, isConstrained) in zip(data.angles, data.isAngleConstrained):
            type1 = data.atomType[data.atoms[angle[0]]]
            type2 = data.atomType[data.atoms[angle[1]]]
            type3 = data.atomType[data.atoms[angle[2]]]
            for p_idx, i in enumerate(self.anglesForAtom2Type[type2]):
                types1 = self.types1[i]
                types2 = self.types2[i]
                types3 = self.types3[i]
                if (type1 in types1 and type2 in types2 and type3 in types3) or (type1 in types3 and type2 in types2 and type3 in types1):
                    a = self.angle[i]
                    k = self.k[i]
                    angle_param_idxs.append(p_idx)
                    angle_idxs.append((angle[0], angle[1], angle[2]))
                    break

        angle_params = params[onp.array(angle_param_idxs)]
        angle_idxs = onp.array(angle_idxs, dtype=np.int32)
        return angle_params, angle_idxs

        # return params[onp.array(bond_param_idxs)], onp.array(bond_idxs, dtype=np.int32)

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        existing = [sys.getForce(i) for i in range(sys.getNumForces())]
        existing = [f for f in existing if type(f) == mm.HarmonicAngleForce]
        if len(existing) == 0:
            force = mm.HarmonicAngleForce()
            sys.addForce(force)
        else:
            force = existing[0]
        for (angle, isConstrained) in zip(data.angles, data.isAngleConstrained):
            type1 = data.atomType[data.atoms[angle[0]]]
            type2 = data.atomType[data.atoms[angle[1]]]
            type3 = data.atomType[data.atoms[angle[2]]]
            for i in self.anglesForAtom2Type[type2]:
                types1 = self.types1[i]
                types2 = self.types2[i]
                types3 = self.types3[i]
                if (type1 in types1 and type2 in types2 and type3 in types3) or (type1 in types3 and type2 in types2 and type3 in types1):
                    if isConstrained:
                        # Find the two bonds that make this angle.

                        bond1 = None
                        bond2 = None
                        for bond in data.atomBonds[angle[1]]:
                            atom1 = data.bonds[bond].atom1
                            atom2 = data.bonds[bond].atom2
                            if atom1 == angle[0] or atom2 == angle[0]:
                                bond1 = bond
                            elif atom1 == angle[2] or atom2 == angle[2]:
                                bond2 = bond

                        # Compute the distance between atoms and add a constraint

                        if bond1 is not None and bond2 is not None:
                            l1 = data.bonds[bond1].length
                            l2 = data.bonds[bond2].length
                            if l1 is not None and l2 is not None:
                                length = sqrt(l1*l1 + l2*l2 - 2*l1*l2*cos(self.angle[i]))
                                data.addConstraint(sys, angle[0], angle[2], length)
                    if self.k[i] != 0:
                        if not isConstrained or args.get('flexibleConstraints', False):
                            force.addAngle(angle[0], angle[1], angle[2], self.angle[i], self.k[i])
                    break

parsers["HarmonicAngleForce"] = HarmonicAngleGenerator.parseElement

## @private
class PeriodicTorsion(object):
    """A PeriodicTorsion records the information for a periodic torsion definition."""

    def __init__(self, types):
        self.types1 = types[0]
        self.types2 = types[1]
        self.types3 = types[2]
        self.types4 = types[3]
        self.periodicity = []
        self.phase = []
        self.k = []
        self.ordering = 'default'

## @private
class PeriodicTorsionGenerator(object):
    """A PeriodicTorsionGenerator constructs a PeriodicTorsionForce."""

    # (ytz): we need to prefix-sum and unroll this ourselves
    def __init__(self, forcefield):
        self.ff = forcefield
        self.proper = []
        self.improper = []

        self.params = []
        self.start_end = []

        self.propersForAtomType = defaultdict(set)

    def _insert_torsion(self, torsion):

        assert len(torsion.k) == len(torsion.phase)
        assert len(torsion.phase) == len(torsion.periodicity)

        start = len(self.params)
        for k, phase, period in zip(torsion.k, torsion.phase, torsion.periodicity):
            self.params.append((k, phase, period))
        end = len(self.params)

        self.start_end.append((start, end))

    def registerProperTorsion(self, parameters):
        torsion = self.ff._parseTorsion(parameters)
        if torsion is not None:
            index = len(self.proper)
            self.proper.append(torsion)
            for t in torsion.types2:
                self.propersForAtomType[t].add(index)
            for t in torsion.types3:
                self.propersForAtomType[t].add(index)

            self._insert_torsion(torsion)

    def registerImproperTorsion(self, parameters, ordering='default'):
        torsion = self.ff._parseTorsion(parameters)
        if torsion is not None:
            if ordering in ['default', 'charmm', 'amber']:
                torsion.ordering = ordering
            else:
                raise ValueError('Illegal ordering type %s for improper torsion %s' % (ordering, torsion))
            self.improper.append(torsion)

            self._insert_torsion(torsion)

    @staticmethod
    def extract_params(tag_dict):
        idx = 1
        params = []
        while 'periodicity'+str(idx) in tag_dict:
            period, phase, k = tag_dict['periodicity'+str(idx)], tag_dict['phase'+str(idx)], tag_dict['k'+str(idx)]
            print(period, phase, k)
            idx += 1

    @staticmethod
    def parseElement(element, ff):
        existing = [f for f in ff._forces if isinstance(f, PeriodicTorsionGenerator)]
        if len(existing) == 0:
            generator = PeriodicTorsionGenerator(ff)
            ff.registerGenerator(generator)
        else:
            generator = existing[0]

        torsion_idx = 0
        for torsion in element.findall('Proper'):
            generator.registerProperTorsion(torsion.attrib)

        for torsion in element.findall('Improper'):
            if 'ordering' in element.attrib:
                generator.registerImproperTorsion(torsion.attrib, element.attrib['ordering'])
            else:
                generator.registerImproperTorsion(torsion.attrib)

        generator.params = onp.array(generator.params, dtype=np.float64)
        generator.start_end = onp.array(generator.start_end, dtype=np.int32)

    def parameterize(self, params, data):
        torsion_param_idxs = []
        torsion_idxs = []

        wildcard = self.ff._atomClasses['']
        proper_cache = {}
        for torsion in data.propers:
            type1, type2, type3, type4 = [data.atomType[data.atoms[torsion[i]]] for i in range(4)]
            sig = (type1, type2, type3, type4)
            sig = frozenset((sig, sig[::-1]))
            match = proper_cache.get(sig, (None, None))
            if match[0] == -1:
                continue
            if match[0] is None:
                for index in self.propersForAtomType[type2]:
                    tordef = self.proper[index]
                    types1 = tordef.types1
                    types2 = tordef.types2
                    types3 = tordef.types3
                    types4 = tordef.types4
                    if (type2 in types2 and type3 in types3 and type4 in types4 and type1 in types1) or (type2 in types3 and type3 in types2 and type4 in types1 and type1 in types4):
                        hasWildcard = (wildcard in (types1, types2, types3, types4))
                        if match[0] is None or not hasWildcard: # Prefer specific definitions over ones with wildcards
                            match = (tordef, index)
                        if not hasWildcard:
                            break
                if match[0] is None:
                    proper_cache[sig] = (-1, None)
                else:
                    proper_cache[sig] = (match, index)
            if match[0] is not None:
                index = match[1]
                start, end = self.start_end[index]
                for i in range(start, end):
                    torsion_param_idxs.append(i)
                    torsion_idxs.append((torsion[0], torsion[1], torsion[2], torsion[3]))

        impr_cache = {}
        for torsion in data.impropers:
            t1, t2, t3, t4 = tatoms = [data.atomType[data.atoms[torsion[i]]] for i in range(4)]
            sig = (t1, t2, t3, t4)
            match_tuple = impr_cache.get(sig, (None, None))
            match, impr_idx = match_tuple
            if match == -1:
                # Previously checked, and doesn't appear in the database
                continue
            elif match:
                i1, i2, i3, i4, tordef = match
                a1, a2, a3, a4 = (torsion[i] for i in (i1, i2, i3, i4))
                match = (a1, a2, a3, a4, tordef)

            if match is None:
                match, impr_idx = _matchImproper(data, torsion, self)
                if match is not None:
                    order = match[:4]
                    i1, i2, i3, i4 = tuple(torsion.index(a) for a in order)
                    impr_cache[sig] = ((i1, i2, i3, i4, match[-1]), impr_idx)
                else:
                    impr_cache[sig] = (-1, impr_idx)

            if match is not None:
                (a1, a2, a3, a4, tordef) = match
                start, end = self.start_end[impr_idx + len(self.proper)]
                for i in range(start, end):
                    torsion_param_idxs.append(i)
                    torsion_idxs.append((a1, a2, a3, a4))

        return self.params[onp.array(torsion_param_idxs)], onp.array(torsion_idxs, dtype=np.int32)


    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        existing = [sys.getForce(i) for i in range(sys.getNumForces())]
        existing = [f for f in existing if type(f) == mm.PeriodicTorsionForce]
        if len(existing) == 0:
            force = mm.PeriodicTorsionForce()
            sys.addForce(force)
        else:
            force = existing[0]
        wildcard = self.ff._atomClasses['']
        proper_cache = {}
        for torsion in data.propers:
            type1, type2, type3, type4 = [data.atomType[data.atoms[torsion[i]]] for i in range(4)]
            sig = (type1, type2, type3, type4)
            sig = frozenset((sig, sig[::-1]))
            match = proper_cache.get(sig, None)
            if match == -1:
                continue
            if match is None:
                for index in self.propersForAtomType[type2]:
                    tordef = self.proper[index]
                    types1 = tordef.types1
                    types2 = tordef.types2
                    types3 = tordef.types3
                    types4 = tordef.types4
                    if (type2 in types2 and type3 in types3 and type4 in types4 and type1 in types1) or (type2 in types3 and type3 in types2 and type4 in types1 and type1 in types4):
                        hasWildcard = (wildcard in (types1, types2, types3, types4))
                        if match is None or not hasWildcard: # Prefer specific definitions over ones with wildcards
                            match = tordef
                        if not hasWildcard:
                            break
                if match is None:
                    proper_cache[sig] = -1
                else:
                    proper_cache[sig] = match
            if match is not None:
                for i in range(len(match.phase)):
                    if match.k[i] != 0:
                        force.addTorsion(torsion[0], torsion[1], torsion[2], torsion[3], match.periodicity[i], match.phase[i], match.k[i])
        impr_cache = {}
        for torsion in data.impropers:
            t1, t2, t3, t4 = tatoms = [data.atomType[data.atoms[torsion[i]]] for i in range(4)]
            sig = (t1, t2, t3, t4)
            match = impr_cache.get(sig, None)
            if match == -1:
                # Previously checked, and doesn't appear in the database
                continue
            elif match:
                i1, i2, i3, i4, tordef = match
                a1, a2, a3, a4 = (torsion[i] for i in (i1, i2, i3, i4))
                match = (a1, a2, a3, a4, tordef)
            if match is None:
                match = _matchImproper(data, torsion, self)
                if match is not None:
                    order = match[:4]
                    i1, i2, i3, i4 = tuple(torsion.index(a) for a in order)
                    impr_cache[sig] = (i1, i2, i3, i4, match[-1])
                else:
                    impr_cache[sig] = -1
            if match is not None:
                (a1, a2, a3, a4, tordef) = match
                for i in range(len(tordef.phase)):
                    if tordef.k[i] != 0:
                        force.addTorsion(a1, a2, a3, a4, tordef.periodicity[i], tordef.phase[i], tordef.k[i])
parsers["PeriodicTorsionForce"] = PeriodicTorsionGenerator.parseElement

## @private
class NonbondedGenerator(object):
    """A NonbondedGenerator constructs a NonbondedForce."""

    SCALETOL = 1e-5

    def __init__(self, forcefield, coulomb14scale, lj14scale):
        self.ff = forcefield
        self.coulomb14scale = coulomb14scale
        self.lj14scale = lj14scale
        self.typed_params = ForceField._AtomTypeParameters(forcefield, 'NonbondedForce', 'Atom', ('charge', 'sigma', 'epsilon'))
        self.params = self.typed_params.params

    def registerAtom(self, parameters):
        self.typed_params.registerAtom(parameters)

    @staticmethod
    def parseElement(element, ff):
        existing = [f for f in ff._forces if isinstance(f, NonbondedGenerator)]
        if len(existing) == 0:
            generator = NonbondedGenerator(ff, float(element.attrib['coulomb14scale']), float(element.attrib['lj14scale']))
            ff.registerGenerator(generator)
        else:
            # Multiple <NonbondedForce> tags were found, probably in different files.  Simply add more types to the existing one.
            # eg water + protein nonbonded tags
            generator = existing[0]
            if abs(generator.coulomb14scale - float(element.attrib['coulomb14scale'])) > NonbondedGenerator.SCALETOL or \
                    abs(generator.lj14scale - float(element.attrib['lj14scale'])) > NonbondedGenerator.SCALETOL:
                raise ValueError('Found multiple NonbondedForce tags with different 1-4 scales')
        generator.typed_params.parseDefinitions(element)

    def parameterize(self, params, data):
        param_idxs = []
        for atom in data.atoms:
            param_idx = self.typed_params.getAtomParameters(atom, data)
            param_idxs.append(param_idx)
        nb_params = params[onp.array(param_idxs, dtype=np.int32)]
        return nb_params

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        methodMap = {NoCutoff:mm.NonbondedForce.NoCutoff,
                     CutoffNonPeriodic:mm.NonbondedForce.CutoffNonPeriodic,
                     CutoffPeriodic:mm.NonbondedForce.CutoffPeriodic,
                     Ewald:mm.NonbondedForce.Ewald,
                     PME:mm.NonbondedForce.PME,
                     LJPME:mm.NonbondedForce.LJPME}
        if nonbondedMethod not in methodMap:
            raise ValueError('Illegal nonbonded method for NonbondedForce')
        force = mm.NonbondedForce()
        for atom in data.atoms:
            values = self.typed_params.getAtomParameters(atom, data)
            force.addParticle(values[0], values[1], values[2])
        force.setNonbondedMethod(methodMap[nonbondedMethod])
        force.setCutoffDistance(nonbondedCutoff)
        if args['switchDistance'] is not None:
            force.setUseSwitchingFunction(True)
            force.setSwitchingDistance(args['switchDistance'])
        if 'ewaldErrorTolerance' in args:
            force.setEwaldErrorTolerance(args['ewaldErrorTolerance'])
        if 'useDispersionCorrection' in args:
            force.setUseDispersionCorrection(bool(args['useDispersionCorrection']))
        sys.addForce(force)

    def postprocessSystem(self, sys, data, args):
        # Create the exceptions.

        bondIndices = _findBondsForExclusions(data, sys)
        nonbonded = [f for f in sys.getForces() if isinstance(f, mm.NonbondedForce)][0]
        nonbonded.createExceptionsFromBonds(bondIndices, self.coulomb14scale, self.lj14scale)

parsers["NonbondedForce"] = NonbondedGenerator.parseElement

