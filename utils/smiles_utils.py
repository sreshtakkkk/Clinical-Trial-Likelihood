"""
utils/smiles_utils.py
=====================
SMILES molecular feature extraction utilities.
"""

import re
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def basic_smiles_features(smi: str) -> dict:
    if not isinstance(smi, str) or not smi.strip():
        return {}
    feat = {
        "smi_length": len(smi),
        "smi_ring_count": smi.count("1") + smi.count("2") + smi.count("3"),
        "smi_branch_count": smi.count("("),
        "smi_double_bonds": smi.count("="),
        "smi_triple_bonds": smi.count("#"),
        "smi_aromatic_atoms": sum(1 for c in smi if c.islower()),
        "smi_nitrogen": smi.upper().count("N"),
        "smi_oxygen": smi.upper().count("O"),
        "smi_fluorine": smi.upper().count("F"),
        "smi_heteroatom_count": smi.upper().count("N") + smi.upper().count("O"),
    }
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mol = Chem.MolFromSmiles(smi)
        if mol:
            feat["mol_weight"] = Descriptors.MolWt(mol)
            feat["mol_logp"] = Descriptors.MolLogP(mol)
            feat["mol_hbd"] = rdMolDescriptors.CalcNumHBD(mol)
            feat["mol_hba"] = rdMolDescriptors.CalcNumHBA(mol)
            feat["mol_tpsa"] = Descriptors.TPSA(mol)
            feat["mol_rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
    except Exception:
        pass
    return feat


def extract_batch_smiles_features(smiles_series: pd.Series) -> pd.DataFrame:
    features = [basic_smiles_features(s) for s in smiles_series]
    return pd.DataFrame(features).fillna(0)
