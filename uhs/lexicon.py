"""Transparent rule-based tagger (independent of Laya).

Purpose:
  1. Extra reservoir-engineering / porous-media dimensions that Laya's 9 questions do not cover.
  2. An INDEPENDENT baseline: where Laya and the rules disagree, papers go to a manual review queue.
  3. Cheap, deterministic pre-filter (no hydrogen term at all -> cannot be about UHS).

Tags are multi-label and evaluated on title + abstract only. They mean "the abstract mentions this
topic", not "the paper is mainly about it" - the report states this limitation.
"""
from __future__ import annotations

import re

_H2 = re.compile(r"\bhydrogen\b|\bh2\b|\bh_2\b|\bh-2\b", re.I)
_STORE = re.compile(r"\bstor(?:age|e|ed|ing)\b|\bstorag", re.I)
_SUB = re.compile(r"underground|subsurface|sub-surface|geolog|aquifer|cavern|depleted (?:gas|oil|field|reservoir|hydrocarbon)|"
                  r"porous (?:media|medium|rock|formation)|reservoir (?:storage|scale|simulation|pressure)|"
                  r"saline (?:aquifer|formation)|salt dome|rock formation|geo-?storage", re.I)
_STRONG = re.compile(
    r"(?:underground|geological|geologic|subsurface|subterranean)[\s-]+(?:hydrogen|h2)[\s-]+(?:storage|storing)"
    r"|(?:hydrogen|h2)[\s-]+(?:underground|subsurface|geological|geologic)[\s-]+storage"
    r"|\buhs\b"
    r"|(?:storage|storing|store)\s+(?:of\s+)?(?:pure\s+)?(?:hydrogen|h2)\s+(?:in|within)\s+"
    r"(?:underground|subsurface|geological|geologic|saline|salt|depleted|porous|aquifer|reservoir)"
    r"|(?:hydrogen|h2)[\s-]+(?:storage|injection)\s+(?:in|into|within)\s+"
    r"(?:underground|subsurface|geological|geologic|saline|salt|depleted|porous|aquifer|reservoir|caverns?)", re.I)

# name -> (group, regex). Groups drive the coverage matrix and the report.
TAGS: dict[str, tuple[str, str]] = {
    # ---- storage setting
    "depleted_hydrocarbon_reservoir": ("setting", r"depleted (?:natural )?(?:gas|oil|hydrocarbon)|(?:gas|oil|hydrocarbon) (?:reservoir|field)s?|depleted (?:reservoir|field)s?|exhausted (?:gas|oil)"),
    "saline_aquifer": ("setting", r"aquifer|saline formation|saline porous|deep saline|brine[- ]saturated formation"),
    "salt_cavern": ("setting", r"salt cavern|salt dome|salt formation|bedded salt|rock salt|salt rock|salt cavit|solution[- ]mined|halite formation"),
    "hard_rock_cavern": ("setting", r"lined rock cavern|\blrc\b|hard[- ]rock cavern|unlined rock cavern|abandoned mine|mined cavern|crystalline rock"),
    "basalt_mafic": ("setting", r"basalt|mafic|ultramafic|volcanic rock"),
    "coal_seam": ("setting", r"coal seam|coalbed|coal bed"),
    # ---- lithology / seal
    "sandstone": ("lithology", r"sandstone|sand stone|berea|bentheimer|fontainebleau"),
    "carbonate": ("lithology", r"carbonate|limestone|dolomite|calcite|indiana limestone|edwards"),
    "shale_clay_seal": ("lithology", r"shale|mudstone|claystone|clay mineral|\bclay\b|illite|kaolinite|montmorillonite|smectite"),
    "caprock_seal": ("lithology", r"cap ?rock|seal (?:integrity|rock|capacity)|sealing capacity|breakthrough pressure|top seal"),
    "evaporite": ("lithology", r"evaporite|anhydrite|gypsum"),
    # ---- gases / fluids
    "cushion_gas": ("fluid", r"cushion gas|base gas|buffer gas"),
    "gas_mixing_blending": ("fluid", r"gas mixing|mixing (?:of|between)|hydrogen[- ]methane|blend(?:ed|ing)|h2[- ](?:ch4|methane)|mixture of (?:hydrogen|h2)"),
    "co2_involved": ("fluid", r"carbon dioxide|\bco2\b|\bco_2\b"),
    "methane_natural_gas": ("fluid", r"natural gas|methane|\bch4\b"),
    "nitrogen": ("fluid", r"nitrogen|\bn2\b"),
    # ---- pore-scale & flow physics
    "multiphase_flow": ("flow_physics", r"multiphase|multi-phase|two[- ]phase|gas[- ]water|gas[- ]brine|immiscible|drainage|imbibition|(?:fluid|gas|brine|water|immiscible|piston-like) displacement"),
    "relative_permeability": ("flow_physics", r"relative permeabilit|\bkr\b|corey"),
    "capillary_pressure": ("flow_physics", r"capillary"),
    "wettability_contact_angle": ("flow_physics", r"wettabilit|contact angle|wetting"),
    "interfacial_tension": ("flow_physics", r"interfacial tension|\bift\b|surface tension"),
    "hysteresis": ("flow_physics", r"hysteresis"),
    "trapping": ("flow_physics", r"residual (?:gas )?(?:trapping|saturation)|structural trapping|solubility trapping|capillary trapping|snap[- ]?off|trapped gas|trapping"),
    "diffusion_dispersion": ("flow_physics", r"diffusi|dispersi"),
    "gas_solubility": ("flow_physics", r"solubilit|dissolved (?:hydrogen|h2)|dissolution of (?:hydrogen|h2)"),
    "viscous_fingering_gravity": ("flow_physics", r"fingering|instabilit|gravity (?:override|segregation|drainage)|buoyan"),
    "petrophysics": ("flow_physics", r"porosity|permeability|pore (?:size|throat|structure|space)|tortuosity|\bnmr\b|mercury intrusion"),
    "heterogeneity_upscaling": ("flow_physics", r"heterogene|upscal|facies|geostatist|permeability field|stochastic"),
    # ---- coupled processes
    "geochemistry": ("coupled_process", r"geochemi|mineral (?:reaction|dissolution|precipitation)|(?:calcite|carbonate|pyrite|siderite|mineral) (?:dissolution|precipitation)|dissolution[- ]precipitation|reactive transport|pyrite|redox|abiotic|water[- ]rock|hydrogen sulfide|\bh2s\b|siderite"),
    "microbial_methanogenesis": ("coupled_process", r"microb|methanogen|sulfate[- ]reduc|sulphate[- ]reduc|acetogen|archaea|bacteri|biogeochem|biofilm|biodegrad"),
    "geomechanics": ("coupled_process", r"geomechanic|(?:in[- ]situ|effective|mechanical|thermal|differential|shear) stress|stress (?:field|state|path|change|concentration|redistribution)|fault reactivation|induced seismic|subsidence|salt creep|\bcreep\b|rock mechanic|thermo-hydro-mechanical|\bthm\b|mechanical integrity|deformation|fracture propagation"),
    "thermal_effects": ("coupled_process", r"thermal|non-isothermal|temperature (?:effect|change|variation)|joule[- ]thomson|thermodynamic"),
    "well_integrity_corrosion": ("coupled_process", r"well integrity|wellbore|cement|casing|corrosion|embrittle"),
    "hydrogen_loss": ("coupled_process", r"hydrogen loss|h2 loss|loss of (?:hydrogen|h2)|hydrogen consumption|hydrogen depletion|lost hydrogen"),
    "leakage_containment": ("coupled_process", r"leak|containment|migration|seal failure|loss of (?:hydrogen|h2)"),
    # ---- operations / performance
    "cyclic_operation": ("operations", r"cyclic|injection[- ](?:and|-)?[- ]?(?:withdrawal|production|extraction)|working gas|deliverab|injectivity|seasonal|storage efficiency|hydrogen recovery|recovery factor|\bcycle"),
    "purity_recovery": ("operations", r"purity|recovery efficiency|gas recovery|recoverable"),
    "field_pilot_case_study": ("operations", r"field (?:test|pilot|trial|scale)|pilot (?:project|test)|demonstration|case study|underground sun|town gas|hyuspre|hychico|hystorage|site (?:selection|screening)|feasibility"),
    "techno_economic_lca": ("operations", r"techno[- ]?economic|levelized|\blcoh\b|capex|opex|life[- ]cycle|\blca\b|carbon footprint|economic (?:analysis|assessment|evaluation|feasibility|viability)|cost (?:analysis|assessment|estimation|model|reduction|comparison|effective)|cost-effective|investment cost|storage cost"),
    "policy_regulation_safety": ("operations", r"regulat|policy|safety|risk assessment|hazard|public acceptance|permitting"),
    # ---- methods
    "core_flooding": ("method", r"core[- ]?flood|coreflood|steady[- ]state|unsteady[- ]state|centrifuge|core plug|core sample|core-scale"),
    "micro_ct_imaging": ("method", r"micro[- ]?ct|x-ray|tomograph|micromodel|microfluidic|digital rock|imaging"),
    "molecular_simulation": ("method", r"molecular dynamic|molecular simulation|density functional|\bdft\b|monte carlo simulation|\bgcmc\b"),
    "pore_scale_modeling": ("method", r"pore[- ]network|lattice[- ]boltzmann|\blbm\b|pore[- ]scale (?:model|simulation)|direct numerical simulation|volume of fluid|\bvof\b"),
    "reservoir_simulation": ("method", r"reservoir simulation|reservoir model|numerical simulation|numerical model|simulator|finite element|finite volume|finite difference|\bcfd\b|reactive transport model"),
    "analytical_theory": ("method", r"analytical (?:model|solution)|semi-analytical|closed[- ]form|theoretical (?:model|analysis)"),
    "machine_learning": ("method", r"machine learning|deep learning|neural network|artificial intelligence|random forest|xgboost|gaussian process|data[- ]driven|physics[- ]informed|surrogate|proxy model"),
    "optimization": ("method", r"optimi[sz]"),
    "sensitivity_uncertainty": ("method", r"sensitivity (?:analysis|study)|uncertainty|monte carlo|design of experiments|response surface|sobol|parametric study|global sensitivity|tornado"),
    "experimental_lab": ("method", r"experiment|laborator|core[- ]?flood|bench[- ]scale|in-house measurement|measured (?:in|at|using|with|by)"),
    "review_perspective": ("method", r"\breview\b|state[- ]of[- ]the[- ]art|perspective|overview|bibliometric|roadmap|systematic|challenges and"),
}
_COMPILED = {k: (g, re.compile(p, re.I)) for k, (g, p) in TAGS.items()}
GROUPS = {k: g for k, (g, _) in TAGS.items()}

TOOLS = {
    "CMG (GEM/STARS/IMEX)": r"\bcmg\b|computer modelling group|cmg[- ]gem",
    "Eclipse": r"eclipse[- ](?:100|300|simulat\w*|reservoir|software|black[- ]oil|compositional)|(?:using|with|by|via|in|and) (?:the )?(?:schlumberger )?eclipse\b",
    "tNavigator": r"tnavigator",
    "MRST": r"\bmrst\b",
    "TOUGH/TOUGHREACT": r"\btough\d?\b|toughreact|tough2",
    "PFLOTRAN": r"pflotran",
    "OpenGoSim": r"opengosim|pflotran-ogs",
    "DuMux": r"dumux",
    "OpenGeoSys": r"opengeosys|\bogs[- ](?:simulat\w*|code|software|framework)",
    "COMSOL": r"comsol",
    "Petrel": r"petrel",
    "OpenFOAM": r"openfoam",
    "MOOSE/GEOS": r"\bmoose (?:framework|simulat\w*|application|multiphysics)|moose[- ]based|\bgeosx\b|\bgeos (?:simulat\w*|code|framework|software)",
    "OPM Flow": r"open porous media|\bopm[- ]flow\b",
    "DARTS": r"\bopen[- ]?darts\b|\bdarts (?:simulat\w*|framework)",
    "STOMP": r"\bstomp[- ](?:h2|co2|eor|simulat\w*)|\bstomp code",
    "Python/ML frameworks": r"tensorflow|pytorch|scikit",
}
_TOOLS = {k: re.compile(p, re.I) for k, p in TOOLS.items()}

# tag -> Laya storage label
SETTING_TO_LAYA = {"depleted_hydrocarbon_reservoir": "depleted_gas_reservoir", "saline_aquifer": "aquifer",
                   "salt_cavern": "salt_cavern", "hard_rock_cavern": "hard_rock_cavern", "basalt_mafic": "other_geology",
                   "coal_seam": "other_geology"}
_SCALE = {
    "pore": re.compile(r"pore[- ]scale|micromodel|microfluidic|pore network|digital rock|lattice[- ]boltzmann|molecular|nano|micro-?ct|pore[- ]level", re.I),
    "core": re.compile(r"core[- ]?flood|core[- ]scale|core sample|core plug|laboratory|column|bench[- ]scale|sample", re.I),
    "reservoir": re.compile(r"reservoir[- ]scale|reservoir simulation|reservoir model|reservoir performance|numerical simulation|injection well|well[- ]scale|formation[- ]scale", re.I),
    "field": re.compile(r"field[- ]scale|field pilot|field test|regional|basin|site selection|case study|national|europe|potential|capacity assessment", re.I),
}


def normalize(text: str) -> str:
    t = (text or "").replace("₂", "2").replace("–", "-").replace("—", "-").replace("‑", "-")
    return t.lower()


def has_hydrogen(text: str) -> bool:
    return bool(_H2.search(normalize(text)))


def tag_record(title: str, abstract: str) -> dict:
    text = normalize(f"{title or ''}. {abstract or ''}")
    tags = [k for k, (_, rx) in _COMPILED.items() if rx.search(text)]
    tools = [k for k, rx in _TOOLS.items() if rx.search(text)]
    h2 = bool(_H2.search(text))

    # --- rule-based UHS relevance: strong / likely / unlikely / none
    if not h2:
        uhs = "none"
    elif _STRONG.search(text):
        uhs = "strong"
    elif _STORE.search(text) and _SUB.search(text):
        uhs = "likely"
    else:
        uhs = "unlikely"

    # --- rule-based storage setting = most frequently matched setting
    counts = {}
    for k, lab in SETTING_TO_LAYA.items():
        n = len(_COMPILED[k][1].findall(text))
        if n:
            counts[lab] = counts.get(lab, 0) + n
    if not counts:
        storage = "none"
    else:
        top = sorted(counts.items(), key=lambda kv: -kv[1])
        storage = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else "multiple"

    sc = {k: len(rx.findall(text)) for k, rx in _SCALE.items()}
    top = sorted(sc.items(), key=lambda kv: -kv[1])
    scale = "none" if top[0][1] == 0 else (top[0][0] if top[0][1] > top[1][1] else "unclear")

    exp = "experimental_lab" in tags or "core_flooding" in tags or "micro_ct_imaging" in tags
    num = any(t in tags for t in ("reservoir_simulation", "pore_scale_modeling", "molecular_simulation"))
    if "review_perspective" in tags and not (exp and num):
        method = "review"
    elif exp and num:
        method = "mixed"
    elif exp:
        method = "experimental"
    elif num:
        method = "numerical"
    elif "analytical_theory" in tags:
        method = "analytical"
    elif "machine_learning" in tags:
        method = "data_driven"
    elif "techno_economic_lca" in tags:
        method = "techno_economic"
    else:
        method = "none"
    return dict(tags=tags, tools=tools, has_hydrogen=h2, rule_uhs=uhs, rule_storage=storage,
                rule_scale=scale, rule_method=method)
