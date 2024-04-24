# Timemachine Forcefields

Timemachine requires its own Forcefield definitions, that can be constructed by converting an XML forcefield.

To convert a Forcefield and include AM1CCC charges you can run the following:

```bash
python smirnoff_converter.py smirnoff_1.1.0.xml --add_am1ccc_charges --output params/smirnoff_1-1-0_ccc.py
```

After conversion, it is possible to compare two converted forcefield to see the differences by running the following:

```bash
python compare_forcefields.py params/smirnoff_1_1_0_ccc.py params/smirnoff_2_0_0_ccc.py
```

## Built in Forcefields

The forcefields that are bundled with Timemachine come from the [Open Forcefield Initiative](https://openforcefield.org/), using the constrained definitions. The following forcefields are currently included:

* OpenForceField 1.1.0 - [original](https://github.com/openforcefield/openff-forcefields/blob/master/openforcefields/offxml/openff-1.1.0.offxml)
* OpenForceField 1.3.1 - [original](https://github.com/openforcefield/openff-forcefields/blob/master/openforcefields/offxml/openff-1.3.1.offxml)
* OpenForceField 2.0.0 - [original](https://github.com/openforcefield/openff-forcefields/blob/master/openforcefields/offxml/openff-2.0.0.offxml)
* OpenForceField 2.1.0 - [original](https://github.com/openforcefield/openff-forcefields/blob/main/openforcefields/offxml/openff-2.1.0.offxml)
* OpenForceField 2.2.0 - [original](https://github.com/openforcefield/openff-forcefields/blob/main/openforcefields/offxml/openff-2.2.0.offxml)

## Forcefield Support

Timemachine does not currently support the full OpenForceField definition. The following are some features that are not currently supported:

* [Virtual Sites](https://open-forcefield-toolkit.readthedocs.io/en/latest/virtualsites.html)
* Fractional bond order Interpolation for [bonds](https://open-forcefield-toolkit.readthedocs.io/en/0.10.0/users/smirnoff.html#fractional-bond-orders) and [torsions](https://open-forcefield-toolkit.readthedocs.io/en/0.10.0/users/smirnoff.html#fractional-torsion-bond-orders)
* [Library Charges](https://open-forcefield-toolkit.readthedocs.io/en/latest/smirnoff.html#librarycharges-library-charges-for-polymeric-residues-and-special-solvent-models)


## AM1BCC Charges

The AM1BCC charges used as the basis for our CCC originally come from the [recharge](https://github.com/openforcefield/openff-recharge) project, a port of the original AM1BCC parameters by Christopher Bayly. The source of the charges comes directly from the charges defined [here](https://github.com/openforcefield/openff-recharge/blob/cf18f1920d35af0025ce90c4e1a7f7280b4bd76d/openff/recharge/data/bcc/original-am1-bcc.json).
