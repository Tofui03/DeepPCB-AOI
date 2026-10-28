"""Pixel-area limits used as a stand-in for IPC-A-600 Class 2 and Class 3.

Open and Short always violate (``max_allowed_px = 0``). Other classes compare the
box area with a fixed pixel cap that is tighter for Class 3. The numbers in
``IPC_A_600_SPEC_TABLE`` are project limits, not a certified copy of the standard.
"""

from __future__ import annotations

from typing import Any, get_args

from core.agent.schemas import (
    BoundingBoxDetection,
    DefectClass,
    IPCClass,
    IPCComplianceResult,
)

VALID_DEFECT_CLASSES: set[str] = set(get_args(DefectClass))
VALID_IPC_CLASSES: set[str] = set(get_args(IPCClass))

# Canonical IPC-A-600 specification table covering all 6 DeepPCB defect classes
# across IPC Class 2 (General/Dedicated Service Electronics) and Class 3 (Aerospace/Automotive/Medical).
IPC_A_600_SPEC_TABLE: dict[DefectClass, dict[str, Any]] = {
    "Open": {
        "ipc_clause": "IPC-A-600 Section 2.2.1 (Open Circuit)",
        "Class 2": {
            "ipc_clause": "IPC-A-600 Section 2.2.1 (Open Circuit)",
            "max_allowed_px": 0,
        },
        "Class 3": {
            "ipc_clause": "IPC-A-600 Section 2.2.1 (Open Circuit)",
            "max_allowed_px": 0,
        },
        "electrical_risk": (
            "Complete circuit discontinuity causing open signal net or interrupted power delivery."
        ),
        "root_cause": (
            "Over-etching, photoresist flaking, or mechanical scratch on photomask film."
        ),
        "corrective_action": (
            "Scrap defective PCB immediately; inspect photoresist lamination rollers and etchant spray nozzle pressure."
        ),
    },
    "Short": {
        "ipc_clause": "IPC-A-600 Section 2.2.2 (Short / Bridging)",
        "Class 2": {
            "ipc_clause": "IPC-A-600 Section 2.2.2 (Short / Bridging)",
            "max_allowed_px": 0,
        },
        "Class 3": {
            "ipc_clause": "IPC-A-600 Section 2.2.2 (Short / Bridging)",
            "max_allowed_px": 0,
        },
        "electrical_risk": (
            "Unintended conductive bridge between traces risking dead short, overcurrent, and thermal runaway."
        ),
        "root_cause": (
            "Under-etching, exhausted etchant bath chemistry, or developer residue between adjacent traces."
        ),
        "corrective_action": (
            "Scrap or isolate bridged board; titrate copper etchant solution and verify conveyor dwell speed."
        ),
    },
    "Mousebite": {
        "ipc_clause": "IPC-A-600 Section 2.2.3 (Mousebite / Edge Nick)",
        "Class 2": {
            "ipc_clause": "IPC-A-600 Section 2.2.3 (Mousebite / Edge Nick)",
            "max_allowed_px": 250,
        },
        "Class 3": {
            "ipc_clause": "IPC-A-600 Section 2.2.3 (Mousebite / Edge Nick)",
            "max_allowed_px": 120,
        },
        "electrical_risk": (
            "Localized trace cross-sectional width reduction causing elevated current density, "
            "high-frequency impedance mismatch, and thermal electromigration."
        ),
        "root_cause": (
            "Marginal photoresist edge adhesion or excessive lateral undercut during acid etching."
        ),
        "corrective_action": (
            "Route to rework/verification station; calibrate UV exposure energy and copper pre-clean roughness."
        ),
    },
    "Spur": {
        "ipc_clause": "IPC-A-600 Section 2.2.4 (Spur / Protrusion)",
        "Class 2": {
            "ipc_clause": "IPC-A-600 Section 2.2.4 (Spur / Protrusion)",
            "max_allowed_px": 200,
        },
        "Class 3": {
            "ipc_clause": "IPC-A-600 Section 2.2.4 (Spur / Protrusion)",
            "max_allowed_px": 100,
        },
        "electrical_risk": (
            "Sharp copper protrusion violating minimum dielectric spacing, increasing risk of "
            "high-voltage arcing or dendritic growth under humidity."
        ),
        "root_cause": (
            "Pinholes in negative photomask or localized developer stagnation along trace wall."
        ),
        "corrective_action": (
            "Perform micro-abrasion rework to remove copper protrusion and clean photomask artwork."
        ),
    },
    "Spurious Copper": {
        "ipc_clause": "IPC-A-600 Section 2.2.5 (Spurious Copper)",
        "Class 2": {
            "ipc_clause": "IPC-A-600 Section 2.2.5 (Spurious Copper)",
            "max_allowed_px": 180,
        },
        "Class 3": {
            "ipc_clause": "IPC-A-600 Section 2.2.5 (Spurious Copper)",
            "max_allowed_px": 80,
        },
        "electrical_risk": (
            "Isolated residual copper island introducing parasitic capacitive coupling or "
            "bridging hazard during solder mask application."
        ),
        "root_cause": (
            "Incomplete photoresist development or redeposited copper particulates in rinse chamber."
        ),
        "corrective_action": (
            "Rework laminate surface to remove residual copper speck and replace developer filtration cartridges."
        ),
    },
    "Pin-hole": {
        "ipc_clause": "IPC-A-600 Section 2.2.6 (Pin-hole / Void)",
        "Class 2": {
            "ipc_clause": "IPC-A-600 Section 2.2.6 (Pin-hole / Void)",
            "max_allowed_px": 160,
        },
        "Class 3": {
            "ipc_clause": "IPC-A-600 Section 2.2.6 (Pin-hole / Void)",
            "max_allowed_px": 75,
        },
        "electrical_risk": (
            "Internal void within conductive trace or land pad increasing localized DC resistance "
            "and solder joint voiding risk."
        ),
        "root_cause": (
            "Particulate contamination or air micro-bubble trapped during dry-film resist lamination."
        ),
        "corrective_action": (
            "Inspect trace continuity and cleanroom lamination roll cleanliness."
        ),
    },
}


def get_ipc_metadata(defect_class: DefectClass) -> dict[str, str]:
    """Return deterministic fallback engineering metadata for `defect_class`."""
    if defect_class not in IPC_A_600_SPEC_TABLE:
        raise ValueError(
            f"Unsupported defect_class={defect_class!r}. Expected one of {sorted(VALID_DEFECT_CLASSES)}."
        )
    entry = IPC_A_600_SPEC_TABLE[defect_class]
    return {
        "ipc_clause": str(entry["ipc_clause"]),
        "electrical_risk": str(entry["electrical_risk"]),
        "root_cause": str(entry["root_cause"]),
        "corrective_action": str(entry["corrective_action"]),
    }


def check_ipc_rule(
    defect_class: DefectClass,
    measured_px: int,
    ipc_class: IPCClass = "Class 2",
    *,
    relative_width_pct: float | None = None,
) -> IPCComplianceResult:
    """Compare ``measured_px`` with the class limit for ``defect_class``.

    Open and Short always violate. Other classes violate when the area is above
    ``max_allowed_px``. If ``relative_width_pct`` is passed, they also violate above
    25% (Class 2) or 20% (Class 3). The inspection pipeline does not pass that argument.
    """
    if defect_class not in IPC_A_600_SPEC_TABLE:
        raise ValueError(
            f"Unsupported defect_class={defect_class!r}. Expected one of {sorted(VALID_DEFECT_CLASSES)}."
        )
    if ipc_class not in VALID_IPC_CLASSES:
        raise ValueError(
            f"Unsupported ipc_class={ipc_class!r}. Expected one of {sorted(VALID_IPC_CLASSES)}."
        )
    if measured_px < 0:
        raise ValueError(f"measured_px must be >= 0, got {measured_px}.")

    spec_entry = IPC_A_600_SPEC_TABLE[defect_class]
    class_spec = spec_entry[ipc_class]
    max_allowed_px = int(class_spec["max_allowed_px"])
    ipc_clause = str(class_spec["ipc_clause"])

    if defect_class in ("Open", "Short"):
        violates_ipc = True
    else:
        violates_ipc = measured_px > max_allowed_px
        if relative_width_pct is not None:
            max_width_pct = 20.0 if ipc_class == "Class 3" else 25.0
            if relative_width_pct > max_width_pct:
                violates_ipc = True

    return IPCComplianceResult(
        defect_class=defect_class,
        ipc_class=ipc_class,
        violates_ipc=violates_ipc,
        ipc_clause=ipc_clause,
        max_allowed_px=max_allowed_px,
        measured_px=measured_px,
    )


def evaluate_detection_ipc(
    bbox: BoundingBoxDetection,
    ipc_class: IPCClass = "Class 2",
) -> IPCComplianceResult:
    """Evaluate a Stage 1 `BoundingBoxDetection` directly against IPC-A-600 thresholds."""
    return check_ipc_rule(
        defect_class=bbox.defect_class,
        measured_px=bbox.area_px,
        ipc_class=ipc_class,
    )
