"""The chemistry layer (chemistry-engine-spec.md).

The layer *below* the materials engine: root **atoms** (distilled quantum descriptors)
bond into molecules/compounds, react under conditions, and the resulting crystal
structures are measured by the existing materials engine. Chemistry generates structure;
the materials engine measures it (spec §1).

Built bottom-up along the C0..C5 milestone ladder (spec §18). Public surface grows as the
ladder is climbed; C0 ships :mod:`chemistry.atoms` (the atom model + derived valence).
"""
