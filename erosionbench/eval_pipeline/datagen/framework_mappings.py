# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""OWASP ASI + MAStrike framework enums and cross-reference tables."""

from __future__ import annotations

from enum import Enum
from typing import Dict


class OwaspASI(str, Enum):
    """OWASP Agentic Security Incidents (ASI) framework categories."""

    ASI_01 = "ASI-01 Prompt Manipulation"
    ASI_02 = "ASI-02 Data Exfiltration"
    ASI_03 = "ASI-03 Unauthorized Actions"
    ASI_04 = "ASI-04 State Manipulation"
    ASI_05 = "ASI-05 Trust Exploitation"
    ASI_06 = "ASI-06 Identity Spoofing"
    ASI_07 = "ASI-07 Metric Manipulation"
    ASI_08 = "ASI-08 Resource Abuse"


class MAStrikeTactic(str, Enum):
    """MAStrike framework tactic IDs for multi-agent attack classification."""

    TA0003_001 = "TA0003.001"  # State Poisoning — parameter drift
    TA0003_002 = "TA0003.002"  # State Poisoning — temporal exploitation
    TA0003_004 = "TA0003.004"  # State Poisoning — shared memory pollution
    TA0003_005 = "TA0003.005"  # State Poisoning — policy reframing via context manipulation
    TA0003_007 = "TA0003.007"  # State Poisoning — reasoning influence via trusted channel
    TA0003_008 = "TA0003.008"  # State Poisoning — warranty/service contract fabrication
    TA0004_001 = "TA0004.001"  # Identity Manipulation — authority escalation
    TA0004_002 = "TA0004.002"  # Identity Manipulation — cross-session forgery
    TA0004_003 = "TA0004.003"  # Identity Manipulation — entitlement escalation
    TA0004_004 = "TA0004.004"  # Identity Manipulation — crisis authority escalation
    TA0005_001 = "TA0005.001"  # Trust Cascade — split evidence channels
    TA0005_002 = "TA0005.002"  # Trust Cascade — compliance theater
    TA0005_003 = "TA0005.003"  # Trust Cascade — artifact provenance
    TA0006_002 = "TA0006.002"  # Lineage Attack — lineage forgery
    TA0007_001 = "TA0007.001"  # Metric Gaming — SLA manipulation
    TA0009_003 = "TA0009.003"  # Exfiltration — intermediate step leakage


# Cross-reference: attack_class -> (OwaspASI, MAStrikeTactic)
ATTACK_CLASS_FRAMEWORK_MAP: Dict[str, Dict[str, str]] = {
    "SEMANTIC_STATE_POISONING": {
        "owasp_asi": OwaspASI.ASI_04.value,
        "mastrike": MAStrikeTactic.TA0003_001.value,
    },
    "IDENTITY_SPOOFING_GASLIGHT": {
        "owasp_asi": OwaspASI.ASI_06.value,
        "mastrike": MAStrikeTactic.TA0004_002.value,
    },
    "SPLIT_EVIDENCE_CORROBORATION": {
        "owasp_asi": OwaspASI.ASI_05.value,
        "mastrike": MAStrikeTactic.TA0005_001.value,
    },
    "SUPPLY_CHAIN_POISONING": {
        "owasp_asi": OwaspASI.ASI_05.value,
        "mastrike": MAStrikeTactic.TA0005_003.value,
    },
    "AUTHORITY_ESCALATION": {
        "owasp_asi": OwaspASI.ASI_06.value,
        "mastrike": MAStrikeTactic.TA0004_001.value,
    },
    "POLICY_REFRAMING": {
        "owasp_asi": OwaspASI.ASI_04.value,
        "mastrike": MAStrikeTactic.TA0003_005.value,
    },
    "TEMPORAL_EXPLOITATION": {
        "owasp_asi": OwaspASI.ASI_04.value,
        "mastrike": MAStrikeTactic.TA0003_002.value,
    },
    "COMPLIANCE_THEATER": {
        "owasp_asi": OwaspASI.ASI_05.value,
        "mastrike": MAStrikeTactic.TA0005_002.value,
    },
    "REASONING_INFLUENCE": {
        "owasp_asi": OwaspASI.ASI_04.value,
        "mastrike": MAStrikeTactic.TA0003_007.value,
    },
    "INTERMEDIATE_STEP_LEAKAGE": {
        "owasp_asi": OwaspASI.ASI_02.value,
        "mastrike": MAStrikeTactic.TA0009_003.value,
    },
    "SHARED_MEMORY_POLLUTION": {
        "owasp_asi": OwaspASI.ASI_04.value,
        "mastrike": MAStrikeTactic.TA0003_004.value,
    },
    "LINEAGE_FORGERY": {
        "owasp_asi": OwaspASI.ASI_05.value,
        "mastrike": MAStrikeTactic.TA0006_002.value,
    },
    "SLA_METRIC_GAMING": {
        "owasp_asi": OwaspASI.ASI_07.value,
        "mastrike": MAStrikeTactic.TA0007_001.value,
    },
    "ENTITLEMENT_MANIPULATION": {
        "owasp_asi": OwaspASI.ASI_06.value,
        "mastrike": MAStrikeTactic.TA0004_003.value,
    },
    "WARRANTY_FRAUD_INJECTION": {
        "owasp_asi": OwaspASI.ASI_04.value,
        "mastrike": MAStrikeTactic.TA0003_008.value,
    },
    "CRISIS_ESCALATION_HIJACK": {
        "owasp_asi": OwaspASI.ASI_06.value,
        "mastrike": MAStrikeTactic.TA0004_004.value,
    },
}


def get_framework_mapping(attack_class: str) -> Dict[str, str]:
    """Return OWASP ASI + MAStrike mapping for an attack class."""
    return ATTACK_CLASS_FRAMEWORK_MAP.get(attack_class, {})
