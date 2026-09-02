# ─── Functional Brain Region Mapping ─────────────────────────────────────────
# Knowledge-based layer: maps approximate brain lobe (derived from Grad-CAM
# centroid position) to functional area and potential impacts.
#
# IMPORTANT LIMITATION:
# Lobe assignment is based on a coarse spatial grid applied to a 2-D MRI slice.
# It is NOT a validated anatomical segmentation. This is an educational /
# decision-support approximation only.

FUNCTIONAL_MAPPING = {
    "frontal": {
        "functional_area": "Frontal Lobe — Executive & Motor Control",
        "functions": [
            "Planning and decision-making",
            "Voluntary motor control",
            "Personality and behaviour regulation",
            "Speech production (Broca's area, left hemisphere)",
            "Working memory",
        ],
        "potential_impacts": [
            "Difficulty with planning or problem-solving",
            "Personality or behavioural changes",
            "Weakness or paralysis on the opposite side of the body",
            "Speech or language difficulties (if left-sided)",
            "Impaired concentration or attention",
        ],
    },
    "parietal": {
        "functional_area": "Parietal Lobe — Sensory & Spatial Processing",
        "functions": [
            "Processing touch, pain, and temperature",
            "Spatial awareness and navigation",
            "Integration of sensory information",
            "Reading and arithmetic (dominant hemisphere)",
        ],
        "potential_impacts": [
            "Numbness or tingling on the opposite side of the body",
            "Difficulty with spatial orientation or navigation",
            "Problems with reading or arithmetic",
            "Impaired ability to recognise objects by touch",
        ],
    },
    "temporal": {
        "functional_area": "Temporal Lobe — Memory, Hearing & Language",
        "functions": [
            "Auditory processing and hearing",
            "Memory formation and retrieval",
            "Language comprehension (Wernicke's area, left hemisphere)",
            "Emotion processing (amygdala)",
        ],
        "potential_impacts": [
            "Memory difficulties or amnesia",
            "Hearing problems or auditory hallucinations",
            "Language comprehension difficulties",
            "Seizures (temporal lobe epilepsy)",
            "Emotional or mood changes",
        ],
    },
    "occipital": {
        "functional_area": "Occipital Lobe — Visual Processing",
        "functions": [
            "Primary visual cortex — processing visual input",
            "Colour and motion perception",
            "Object and face recognition",
        ],
        "potential_impacts": [
            "Visual disturbances or blurred vision",
            "Visual field defects (partial or complete loss of vision in one area)",
            "Difficulty recognising objects or faces",
            "Visual hallucinations",
        ],
    },
    "cerebellum": {
        "functional_area": "Cerebellum — Balance & Coordination",
        "functions": [
            "Fine motor coordination",
            "Balance and posture",
            "Timing of movements",
            "Some cognitive and language functions",
        ],
        "potential_impacts": [
            "Loss of balance or unsteady gait (ataxia)",
            "Poor coordination of limb movements",
            "Tremor or involuntary movements",
            "Slurred speech (dysarthria)",
            "Dizziness or vertigo",
        ],
    },
    "brainstem": {
        "functional_area": "Brainstem — Vital Functions",
        "functions": [
            "Breathing and heart rate regulation",
            "Consciousness and arousal",
            "Cranial nerve functions (eye movement, swallowing, facial sensation)",
        ],
        "potential_impacts": [
            "Difficulty swallowing or breathing",
            "Double vision or eye movement problems",
            "Facial numbness or weakness",
            "Altered consciousness or coma (severe cases)",
        ],
    },
    "pituitary": {
        "functional_area": "Pituitary / Sellar Region — Hormonal & Visual Pathway",
        "functions": [
            "Hormone regulation (growth, thyroid, adrenal, reproductive)",
            "Adjacent to optic chiasm (visual pathway)",
        ],
        "potential_impacts": [
            "Hormonal imbalances (fatigue, weight changes, infertility)",
            "Bitemporal visual field loss (tunnel vision) due to optic chiasm compression",
            "Headaches from mass effect",
            "Diabetes insipidus (excessive thirst/urination)",
        ],
    },
    "unknown": {
        "functional_area": "Region Undetermined",
        "functions": ["Could not reliably determine the affected brain region from the available data."],
        "potential_impacts": ["Functional impact cannot be estimated without reliable localisation."],
    },
}


def infer_lobe_from_location(location: str, tumor_class: str, cx: float, cy: float, img_size: int = 224) -> str:
    """
    Map a coarse hemisphere location + pixel centroid to a brain lobe name.

    Args:
        location:   e.g. "upper left hemisphere"
        tumor_class: e.g. "pituitary"
        cx, cy:     centroid pixel coordinates (0..img_size)
        img_size:   image dimension (default 224)

    Returns:
        lobe name key matching FUNCTIONAL_MAPPING
    """
    # Pituitary tumors are almost always in the sellar/pituitary region
    if tumor_class.lower() == "pituitary":
        return "pituitary"

    # Normalise centroid to [0, 1]
    nx = cx / img_size
    ny = cy / img_size

    # Coarse spatial grid for axial MRI slice (approximate):
    #   Top ~30 %  → frontal
    #   Middle vertical band, left/right → parietal or temporal
    #   Bottom ~25 % → occipital or cerebellum
    #
    # Horizontal split: left 40 % / centre 20 % / right 40 %
    # These thresholds are heuristic and intentionally conservative.

    if ny < 0.30:
        return "frontal"
    elif ny > 0.72:
        if nx < 0.35 or nx > 0.65:
            return "cerebellum"
        else:
            return "occipital"
    else:
        # Middle band — distinguish parietal (upper-mid) vs temporal (lower-mid)
        if ny < 0.52:
            return "parietal"
        else:
            return "temporal"


def get_functional_impact(lobe: str) -> dict:
    """Return the functional mapping entry for a given lobe."""
    return FUNCTIONAL_MAPPING.get(lobe, FUNCTIONAL_MAPPING["unknown"])
