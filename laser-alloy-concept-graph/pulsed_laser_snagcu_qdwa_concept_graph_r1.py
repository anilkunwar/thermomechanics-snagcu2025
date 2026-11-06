#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pulsed Laser Soldering of Sn-Ag-Cu (SAC305) — Two-Domain Concept Graph
=======================================================================
Domain layer for a concept-graph / QDWA application, converted from the
"Battery Optimization in Drone Systems" concept graph (v6.2) to the
pulsed-laser-soldering of Sn-3.0Ag-0.5Cu (SAC305) subject of the manuscript
"Understanding the influence of pulsed laser soldering on thermomechanical
behavior of SAC305 microbumps through AI-assisted computational modeling".

TWO DOMAINS (replaces the drone 4-category QDWA):
    laser               -> "Laser Heat Source"        (cause / energy-input side)
    alloy_microstructure-> "Alloy Microstructure"    (effect / structure side)

The central thesis encoded by the cross-domain edges mirrors the manuscript's
Fig. N1 cause-effect pair: the pulsed Gaussian *fluence* (laser domain) drives
the *enthalpy / temperature field*, which in turn governs the *microstructure*
(beta-Sn grains, (Cu,Ni)6Sn5 IMC) and ultimately the *thermomechanical response*
(shear strength, hardness, creep) of the SAC305 alloy.

------------------------------------------------------------------------------
DROP-IN REPLACEMENT — original symbol  ->  new symbol (this module)
------------------------------------------------------------------------------
  BATTERY_EXPERT_LABELS            ->  LASER_SOLDERING_EXPERT_LABELS
  EXPERT_CATEGORY                  ->  EXPERT_DOMAIN
  BatteryProblem                   ->  LaserSolderingProblem
  BatteryProblemDefinition         ->  LaserSolderingProblemDefinition
  BATTERY_PROBLEM_DEFINITIONS      ->  LASER_SOLDERING_PROBLEM_DEFINITIONS
  FOUR_CATEGORIES                  ->  TWO_DOMAINS
  CATEGORY_DISPLAY                 ->  DOMAIN_DISPLAY
  _LABEL2KEY                       ->  _LABEL2KEY
  CATEGORY_SEEDS                   ->  DOMAIN_SEEDS
  PROBLEM_TO_CATEGORY              ->  PROBLEM_TO_DOMAIN
  _HIERARCHY_PARENTS               ->  _HIERARCHY_PARENTS
  BATTERY_KEYWORDS                 ->  LASER_SOLDERING_KEYWORDS
  BATTERY_DESCRIPTOR_MAPPING       ->  DESCRIPTOR_MAPPING
  BATTERY_QUANTITATIVE_PATTERNS    ->  QUANTITATIVE_PATTERNS
  extract_battery_concepts_from_text -> extract_laser_soldering_concepts_from_text
  is_valid_battery_concept         ->  is_valid_laser_soldering_concept
  normalize_battery_concept        ->  normalize_laser_soldering_concept
  categorize_battery_concept       ->  categorize_laser_soldering_concept
  get_battery_category_color      ->  get_concept_domain_color
  DomainOntology (battery)         ->  DomainOntology (this module, 2-domain)

ConceptType / RelationshipType / ConceptNode / Relationship keep the SAME names
and the SAME dataclass shape as the drone app, so the generic app shell
(Streamlit UI, LatentMoEKGExtractor, QDWAWeightAllocator, renderers) is reused
unchanged EXCEPT for the 4->2 category count. See "INTEGRATION NOTES" below.

Run standalone to print a verified structural summary:
    python pulsed_laser_snagcu_concept_graph.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

try:  # networkx is used by the host app; guarded so this module stays importable.
    import networkx as nx
    _HAS_NX = True
except Exception:  # pragma: no cover
    _HAS_NX = False


# ============================================================================
# CONCEPT & RELATIONSHIP TAXONOMY  (names kept identical to the drone app)
# ============================================================================
class ConceptType(Enum):
    MATERIAL = "material"
    PROCESS = "process"
    PROPERTY = "property"
    PHENOMENON = "phenomenon"
    METHOD = "method"
    PARAMETER = "parameter"
    MICROSTRUCTURE = "microstructure"
    MODEL = "model"
    GENERAL = "general"


class RelationshipType(Enum):
    # --- semantic / structural ---
    SYNONYM = "synonym"
    HYPERNYM = "hypernym"
    HYPONYM = "hyponym"
    PART_OF = "part_of"
    HAS_PART = "has_part"
    CO_OCCURS = "co_occurs"
    # --- causal / directional ---
    CAUSES = "causes"
    RESULTS_IN = "results_in"
    INFLUENCES = "influences"
    DEPENDS_ON = "depends_on"
    CONSTRAINS = "constrains"
    MODIFIES = "modifies"
    DRIVES = "drives"
    ENABLES = "enables"
    # --- phase / thermodynamic transitions ---
    TRANSITIONS_TO = "transitions_to"
    FORMS = "forms"
    STABILIZES = "stabilizes"
    PRESERVES = "preserves"
    ABSORBS = "absorbs"
    # --- computation / modeling ---
    COMPUTES = "computes"
    SIMULATES = "simulates"
    MODELS = "models"
    APPROXIMATES = "approximates"
    MAPS = "maps"
    COUPLES = "couples"
    INTERPOLATES = "interpolates"
    # --- analysis / evaluation ---
    QUANTIFIES = "quantifies"
    EVALUATES = "evaluates"
    COMPARES = "compares"
    VALIDATES = "validates"
    CORRELATES = "correlates"
    # --- strengthening / microstructure-mechanics ---
    STRENGTHENS = "strengthens"
    REFINE = "refines"
    IMPEDS = "impeds"
    # --- generic ---
    SEMANTIC = "semantic"
    INFERRED = "inferred"
    BRIDGE = "bridge"


RELATIONSHIP_TO_IDX = {rel.name: i for i, rel in enumerate(RelationshipType)}
NUM_EDGE_TYPES = len(RelationshipType)

EDGE_COLOR_REGISTRY: Dict[RelationshipType, str] = {
    RelationshipType.SYNONYM: "#AAAAAA",      RelationshipType.HYPERNYM: "#5B9BD5",
    RelationshipType.HYPONYM: "#5B9BD5",       RelationshipType.PART_OF: "#70AD47",
    RelationshipType.HAS_PART: "#70AD47",      RelationshipType.CO_OCCURS: "#BFBFBF",
    RelationshipType.CAUSES: "#FF4444",        RelationshipType.RESULTS_IN: "#E06040",
    RelationshipType.INFLUENCES: "#FF8C00",    RelationshipType.DEPENDS_ON: "#DAA520",
    RelationshipType.CONSTRAINS: "#CC5500",    RelationshipType.MODIFIES: "#FF6347",
    RelationshipType.DRIVES: "#DC143C",         RelationshipType.ENABLES: "#FF7F50",
    RelationshipType.TRANSITIONS_TO: "#8A2BE2", RelationshipType.FORMS: "#9370DB",
    RelationshipType.STABILIZES: "#7B68EE",    RelationshipType.PRESERVES: "#6A5ACD",
    RelationshipType.ABSORBS: "#6495ED",
    RelationshipType.COMPUTES: "#008B8B",      RelationshipType.SIMULATES: "#5F9EA0",
    RelationshipType.MODELS: "#4682B4",         RelationshipType.APPROXIMATES: "#87CEEB",
    RelationshipType.MAPS: "#00BFFF",          RelationshipType.COUPLES: "#CD950C",
    RelationshipType.INTERPOLATES: "#48D1CC",
    RelationshipType.QUANTIFIES: "#32CD32",     RelationshipType.EVALUATES: "#228B22",
    RelationshipType.COMPARES: "#3CB371",      RelationshipType.VALIDATES: "#2E8B57",
    RelationshipType.CORRELATES: "#00FA9A",
    RelationshipType.STRENGTHENS: "#7FFF00",    RelationshipType.REFINE: "#ADFF2F",
    RelationshipType.IMPEDS: "#556B2F",
    RelationshipType.SEMANTIC: "#808080",      RelationshipType.INFERRED: "#A9A9A9",
    RelationshipType.BRIDGE: "#C0C0C0",
}
EDGE_COLOR_FALLBACK = "#888888"

# Domain palette — laser (warm, energy) vs alloy microstructure (cool, structure).
DOMAIN_COLORS = {
    "laser": "#E8743B",                 # warm orange — heat source
    "alloy_microstructure": "#3F6FB0",  # cool blue   — structure
}


def get_edge_color(rel_type: RelationshipType) -> str:
    return EDGE_COLOR_REGISTRY.get(rel_type, EDGE_COLOR_FALLBACK)


def get_edge_width(rel_type: RelationshipType) -> float:
    STRONG = {RelationshipType.CAUSES, RelationshipType.DRIVES, RelationshipType.FORMS,
              RelationshipType.STABILIZES, RelationshipType.DEPENDS_ON,
              RelationshipType.CONSTRAINS, RelationshipType.STRENGTHENS}
    MEDIUM = {RelationshipType.INFLUENCES, RelationshipType.RESULTS_IN,
              RelationshipType.MODIFIES, RelationshipType.ENABLES,
              RelationshipType.TRANSITIONS_TO, RelationshipType.COMPUTES,
              RelationshipType.REFINE, RelationshipType.ABSORBS}
    if rel_type in STRONG:
        return 3.0
    if rel_type in MEDIUM:
        return 2.0
    return 1.0


def get_edge_style(rel_type: RelationshipType) -> str:
    DASHED = {RelationshipType.INFERRED, RelationshipType.CO_OCCURS,
              RelationshipType.SEMANTIC, RelationshipType.BRIDGE,
              RelationshipType.INTERPOLATES, RelationshipType.APPROXIMATES}
    return "dashed" if rel_type in DASHED else "solid"


def lighten_hex_color(hex_color: str, factor: float) -> str:
    if not hex_color.startswith("#"):
        return hex_color
    r, g, b = (int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16))
    r = int(r + (255 - r) * factor); g = int(g + (255 - g) * factor); b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


# ============================================================================
# NODE / EDGE DATACLASSES  (same shape as the drone app)
# ============================================================================
@dataclass
class ConceptNode:
    canonical_name: str
    concept_type: ConceptType
    domain: str                       # "laser" | "alloy_microstructure"
    synonyms: Set[str] = field(default_factory=set)
    hypernyms: Set[str] = field(default_factory=set)
    hyponyms: Set[str] = field(default_factory=set)
    related_processes: Set[str] = field(default_factory=set)
    related_properties: Set[str] = field(default_factory=set)
    definition: str = ""
    embedding: Optional["np.ndarray"] = None

    def add_synonym(self, synonym: str) -> None:
        self.synonyms.add(synonym.lower().strip())

    def is_match(self, text: str) -> bool:
        t = text.lower().strip()
        return t == self.canonical_name.lower() or t in self.synonyms


@dataclass
class Relationship:
    source: str
    target: str
    rel_type: RelationshipType
    confidence: float = 1.0
    evidence: str = ""
    inferred: bool = False


# ============================================================================
# DOMAIN ONTOLOGY  — two domains: laser heat source + alloy microstructure
# ============================================================================
LASER = "laser"
ALLOY = "alloy_microstructure"


class DomainOntology:
    """Two-domain ontology for pulsed laser soldering of SAC305.

    Domain `laser`               : the pulsed Gaussian heat source (cause side).
    Domain `alloy_microstructure`: SAC305 phase change, microstructure & mechanics
                                   (effect / structure side).
    Cross-domain edges encode the manuscript thesis:
        pulse parameters -> thermal field -> microstructure -> mechanical response.
    """

    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptNode] = {}
        self.relationships: List[Relationship] = []
        self._build_ontology()

    # ------------------------------------------------------------------ build
    def _build_ontology(self) -> None:
        # === DOMAIN A: LASER (HEAT SOURCE) ===
        self._add_concept("pulsed_laser_source", ConceptType.METHOD, LASER,
            synonyms={"pulsed laser", "laser soldering source", "laser heat source"},
            definition="Nd:YAG solid-state pulsed laser used for localized, non-contact "
                       "reflow of SAC305 solder balls (experimental E_pulse = 7.5 mJ).")
        self._add_concept("ndyag_laser", ConceptType.MATERIAL, LASER,
            synonyms={"nd:yag", "yag laser", "ultra-sb2300"},
            definition="Neodymium-doped YAG solid-state laser source emitting pulsed IR radiation.")
        self._add_concept("pulse_energy", ConceptType.PARAMETER, LASER,
            synonyms={"e_pulse", "pulse energy", "laser energy"},
            definition="Energy per pulse (mJ); experimental 7.5 mJ, simulation sweep 0.5-8.0 mJ.")
        self._add_concept("pulse_width", ConceptType.PARAMETER, LASER,
            synonyms={"tau_pulse", "pulse duration", "pulse on width"},
            definition="Temporal width of a single pulse, 2.0-7.0 ns in the simulation sweep.")
        self._add_concept("repetition_rate", ConceptType.PARAMETER, LASER,
            synonyms={"f_rep", "pulse repetition", "prf"},
            definition="Pulse repetition frequency f_rep (Hz) setting the pulse-train period 1/f_rep.")
        self._add_concept("beam_waist", ConceptType.PARAMETER, LASER,
            synonyms={"gaussian half-width", "alpha", "beam radius"},
            definition="Gaussian beam half-width parameter alpha governing radial decay.")
        self._add_concept("gaussian_profile", ConceptType.PHENOMENON, LASER,
            synonyms={"gaussian beam", "radial gaussian", "spatial profile"},
            definition="Radial Gaussian intensity G(r;alpha)=exp(-2 r^2/alpha^2).")
        self._add_concept("fluence", ConceptType.PROPERTY, LASER,
            synonyms={"laser fluence", "energy density", "spatial fluence"},
            definition="Spatial fluence F=E/(pi alpha^2) exp(-2 r^2/alpha^2), in J/m^2.")
        self._add_concept("temporal_modulation", ConceptType.PROCESS, LASER,
            synonyms={"fourier modulation", "harmonic modulation", "phi_harmonic"},
            definition="Pulse-on temporal waveform built from the first three odd Fourier harmonics.")
        self._add_concept("pulse_on_condition", ConceptType.PROCESS, LASER,
            synonyms={"pulse gating", "pulse on phase", "duty cycle"},
            definition="mod(t,1/f_rep) < tau_pulse condition delimiting the pulse-on window.")
        self._add_concept("average_power", ConceptType.PARAMETER, LASER,
            synonyms={"p_avg", "mean laser power"},
            definition="P_avg = 1e-3 E_pulse f_rep (W), the mean dissipated power.")
        self._add_concept("effective_power", ConceptType.PARAMETER, LASER,
            synonyms={"p_effective", "geometry-scaled power"},
            definition="P_effective = 2 P_avg (R_ball/alpha)^2, scaled to ball geometry.")
        self._add_concept("normalization_coefficient", ConceptType.PARAMETER, LASER,
            synonyms={"power coefficient", "source coefficient c"},
            definition="C = 2 P_effective / (pi R_ball^2), normalizing the source to the ball surface.")
        self._add_concept("laser_absorption", ConceptType.PHENOMENON, LASER,
            synonyms={"surface absorption", "absorptivity"},
            definition="Fraction of incident laser energy absorbed at the solder-ball surface.")
        self._add_concept("laser_heat_source_term", ConceptType.MODEL, LASER,
            synonyms={"i_laser", "source term", "volumetric heat source"},
            definition="Assembled source I_laser = phi_harmonic(t) * C * G(r;alpha) applied to the top surface.")

        # === DOMAIN B: ALLOY MICROSTRUCTURE (STRUCTURE) ===
        # -- material & thermodynamics --
        self._add_concept("sac305_alloy", ConceptType.MATERIAL, ALLOY,
            synonyms={"sn-3.0ag-0.5cu", "sac305", "snagcu solder", "sac305 solder"},
            definition="Sn-3.0Ag-0.5Cu lead-free solder alloy, melting point 217 C, ball dia 100 um.")
        self._add_concept("melting_point", ConceptType.PROPERTY, ALLOY,
            synonyms={"liquidus", "melting temperature", "solidus"},
            definition="Melting temperature ~217 C (490 K); mushy zone spans 217-225 C.")
        self._add_concept("latent_heat", ConceptType.PROPERTY, ALLOY,
            synonyms={"heat of fusion", "l_sac"},
            definition="Latent heat of fusion L_SAC ~= 59 kJ/kg absorbed across the mushy zone.")
        self._add_concept("mushy_zone", ConceptType.PHENOMENON, ALLOY,
            synonyms={"mushy region", "two-phase zone"},
            definition="Solid-liquid coexistence interval 490-498 K where latent heat is absorbed.")
        self._add_concept("enthalpy", ConceptType.PROPERTY, ALLOY,
            synonyms={"specific enthalpy", "h_sac", "enthalpy temperature"},
            definition="H_SAC(T) integrating cp + L df/dT; differentiated to apparent heat capacity.")
        self._add_concept("apparent_heat_capacity", ConceptType.MODEL, ALLOY,
            synonyms={"apparent cp", "effective heat capacity", "c_bar"},
            definition="C_bar = (dh/dt)/(dT/dt); folds latent heat into the transient heat equation.")
        self._add_concept("solid_liquid_transition", ConceptType.PROCESS, ALLOY,
            synonyms={"phase change", "melting-solidification", "solidification"},
            definition="Phase change governed by liquid fraction f_SAC(T) via the enthalpy method.")
        self._add_concept("thermal_conductivity", ConceptType.PROPERTY, ALLOY,
            synonyms={"k_th", "thermal conductivity"},
            definition="k_th(T): solid linear in T; liquid = 1.16 x solid value.")
        self._add_concept("density", ConceptType.PROPERTY, ALLOY,
            synonyms={"rho", "sac305 density"},
            definition="Piecewise density rho(T) for solid and liquid SAC305 regimes.")
        self._add_concept("cte", ConceptType.PROPERTY, ALLOY,
            synonyms={"coefficient of thermal expansion", "alpha_cte"},
            definition="Solid CTE alpha_s(T), non-linear; liquid set to 0 to suppress non-physical stress.")
        self._add_concept("poissons_ratio", ConceptType.PROPERTY, ALLOY,
            synonyms={"nu", "poisson ratio"},
            definition="Non-linear temperature-dependent Poisson ratio nu_s(T).")
        self._add_concept("youngs_modulus", ConceptType.PROPERTY, ALLOY,
            synonyms={"e modulus", "elastic modulus model", "e_s"},
            definition="Quadratic temperature dependence E_s(T) of the solid alloy.")

        # -- melt pool hydrodynamics --
        self._add_concept("melt_pool", ConceptType.PHENOMENON, ALLOY,
            synonyms={"molten pool", "liquid region"},
            definition="Molten SAC305 region where Navier-Stokes flow and Marangoni convection act.")
        self._add_concept("navier_stokes_flow", ConceptType.PROCESS, ALLOY,
            synonyms={"incompressible flow", "melt flow", "fluid flow"},
            definition="Incompressible Navier-Stokes for velocity u and pressure p in the melt pool.")
        self._add_concept("boussinesq_buoyancy", ConceptType.PROCESS, ALLOY,
            synonyms={"thermal buoyancy", "buoyancy force"},
            definition="Boussinesq body-force approximation for thermal buoyancy in the melt.")
        self._add_concept("marangoni_convection", ConceptType.PHENOMENON, ALLOY,
            synonyms={"thermocapillary flow", "marangoni flow", "thermocapillary convection"},
            definition="Surface-tension-gradient driven flow; d(sigma_ST)/dT = -1.101e-4 N/m/K.")
        self._add_concept("melt_viscosity", ConceptType.PROPERTY, ALLOY,
            synonyms={"viscosity", "mu(t)"},
            definition="Arrhenius-type viscosity mu(T) ~2 mPa.s near liquidus; raised in solid to stop flow.")

        # -- microstructure --
        self._add_concept("beta_sn_grain", ConceptType.MICROSTRUCTURE, ALLOY,
            synonyms={"beta-sn", "sn grain", "beta-tin grain"},
            definition="beta-Sn grains: 20-30 striated (laser) vs 2-5 large with twinning (reflow).")
        self._add_concept("grain_refinement", ConceptType.PROCESS, ALLOY,
            synonyms={"grain refinement", "grain refining"},
            definition="Laser-induced rapid cooling refines beta-Sn grains (Hall-Petch).")
        self._add_concept("grain_size", ConceptType.PROPERTY, ALLOY,
            synonyms={"mean grain diameter", "d"},
            definition="Mean beta-Sn grain diameter d controlling Hall-Petch strengthening.")
        self._add_concept("imc_morphology", ConceptType.MICROSTRUCTURE, ALLOY,
            synonyms={"intermetallic morphology", "imc shape"},
            definition="(Cu,Ni)6Sn5 morphology: granular (laser, 0.6 um) vs needle-like (reflow, 1.3 um).")
        self._add_concept("imc_thickness", ConceptType.PROPERTY, ALLOY,
            synonyms={"imc layer thickness"},
            definition="Interfacial (Cu,Ni)6Sn5 layer thickness; laser 0.6 um vs reflow 1.3 um.")
        self._add_concept("imc_volume_fraction", ConceptType.PROPERTY, ALLOY,
            synonyms={"imc volume fraction", "f_imc"},
            definition="Volume fraction f of (Cu,Ni)6Sn5 precipitates: 2.13 (laser) vs 1.42 (reflow).")
        self._add_concept("precipitate", ConceptType.MICROSTRUCTURE, ALLOY,
            synonyms={"(cu,ni)6sn5 particle", "imc particle", "second phase"},
            definition="(Cu,Ni)6Sn5 precipitate particles impeding crack propagation (Orowan).")
        self._add_concept("cooling_rate", ConceptType.PARAMETER, ALLOY,
            synonyms={"r", "quench rate"},
            definition="Cooling rate R; laser soldering yields a far higher R than reflow.")
        self._add_concept("supercooling", ConceptType.PHENOMENON, ALLOY,
            synonyms={"undercooling", "delta t"},
            definition="Degree of supercooling dT; increases with cooling rate, lowers critical nucleation radius.")
        self._add_concept("critical_nucleation_radius", ConceptType.PARAMETER, ALLOY,
            synonyms={"nucleation radius", "r*"},
            definition="r* = 2 alpha T_e / ((N0 Vm^-1)^(1/3) dT); decreases as supercooling grows.")
        self._add_concept("nucleation_rate", ConceptType.PROCESS, ALLOY,
            synonyms={"nucleation", "grain nucleation"},
            definition="Nucleation rate rises as r* falls, promoting finer beta-Sn grains.")

        # -- thermomechanical response --
        self._add_concept("thermal_stress", ConceptType.PHENOMENON, ALLOY,
            synonyms={"thermomechanical stress", "residual stress"},
            definition="Stress from steep, non-uniform temperature gradients during the pulse.")
        self._add_concept("elastoplastic_deformation", ConceptType.PROCESS, ALLOY,
            synonyms={"thermo-elastoplastic deformation", "displacement field"},
            definition="Displacement s from elastoplastic momentum balance (rho d2s/dt2 - div sigma).")
        self._add_concept("cauchy_stress", ConceptType.PROPERTY, ALLOY,
            synonyms={"stress tensor", "sigma"},
            definition="Cauchy stress sigma_ij = C_ijkl eps_kl - Omega_ij (T - T_ref).")
        self._add_concept("ramberg_osgood_model", ConceptType.MODEL, ALLOY,
            synonyms={"ramberg-osgood", "ro plasticity"},
            definition="eps_eq = sigma_eq/E + K_H (sigma_eq/E)^q describing plastic flow.")
        self._add_concept("von_mises_stress", ConceptType.PROPERTY, ALLOY,
            synonyms={"equivalent stress", "sigma_eq"},
            definition="von Mises equivalent stress sigma_eq gating elastic/plastic switching.")
        self._add_concept("yield_strength", ConceptType.PROPERTY, ALLOY,
            synonyms={"sigma_y", "yield stress"},
            definition="Yield strength sigma_Y = 37.5 MPa at 298 K; Hollomon sigma_0=232 MPa, 1/n=5.4.")
        self._add_concept("shear_strength", ConceptType.PROPERTY, ALLOY,
            synonyms={"shear property", "bond shear strength"},
            definition="Shear strength 75.4 MPa (laser) vs 69.2 MPa (reflow); ductile dimple fracture.")
        self._add_concept("hardness", ConceptType.PROPERTY, ALLOY,
            synonyms={"micro-vickers hardness", "nanoindentation hardness"},
            definition="Hardness 279.75 MPa (laser) vs 221.39 MPa (reflow) at 10 mN load.")
        self._add_concept("elastic_modulus_indent", ConceptType.PROPERTY, ALLOY,
            synonyms={"indentation modulus", "oliver-parr modulus"},
            definition="Indentation elastic modulus 53.26 GPa (laser) vs 34.15 GPa (reflow).")
        self._add_concept("creep_resistance", ConceptType.PROPERTY, ALLOY,
            synonyms={"creep performance", "creep displacement"},
            definition="Laser microbumps show 24% lower creep displacement than reflow at 10 mN hold.")
        self._add_concept("ductile_fracture", ConceptType.PHENOMENON, ALLOY,
            synonyms={"dimple fracture", "ductile failure"},
            definition="Ductile fracture within the bulk solder with dimples under shear.")
        self._add_concept("hall_petch_strengthening", ConceptType.MODEL, ALLOY,
            synonyms={"grain boundary strengthening", "hall-petch"},
            definition="sigma_G = sigma_0 + k/sqrt(d); finer laser grains raise strength.")
        self._add_concept("precipitation_strengthening", ConceptType.MODEL, ALLOY,
            synonyms={"orowan strengthening", "particle strengthening"},
            definition="sigma_P from (Cu,Ni)6Sn5 precipitates impeding dislocation motion.")
        self._add_concept("joint_reliability", ConceptType.PROPERTY, ALLOY,
            synonyms={"microbump reliability", "solder joint reliability"},
            definition="Aggregate reliability outcome: shear + hardness + creep resistance.")

        # -- numerical / AI framework (cross-cutting, tagged alloy_microstructure
        #    as the field they reconstruct lives in the structure domain) --
        self._add_concept("fem_simulation", ConceptType.MODEL, ALLOY,
            synonyms={"elmer multiphysics", "finite element model", "fem"},
            definition="Coupled heat/flow/stress FEM in Elmer Multiphysics; dt=1e-9 s, 23-run sweep.")
        self._add_concept("temperature_field", ConceptType.PHENOMENON, ALLOY,
            synonyms={"thermal field", "temperature distribution", "t(x,t)"},
            definition="Spatiotemporal temperature field T(x,t) — the bridge from laser to microstructure.")
        self._add_concept("thermal_gradient", ConceptType.PROPERTY, ALLOY,
            synonyms={"temperature gradient", "grad t"},
            definition="Steep thermal gradients driving stress, buoyancy and Marangoni flow.")
        self._add_concept("naif_framework", ConceptType.MODEL, ALLOY,
            synonyms={"numerically attentive interpolation framework", "naif"},
            definition="Physics-regularized transformer interpolation from FEM runs to continuous fields.")
        self._add_concept("cross_attention", ConceptType.METHOD, ALLOY,
            synonyms={"st-dgpa", "spatio-temporal gated physics attention", "transformer attention"},
            definition="Spatio-temporal gated physics attention mapping pulse parameters to fields.")
        self._add_concept("heat_flux", ConceptType.PROPERTY, ALLOY,
            synonyms={"q_heat", "thermal flux"},
            definition="Volumetric/source heat flux Q entering the energy equation.")

        self._build_synonym_index()
        self._build_causal_chains()

    # ----------------------------------------------------------------- helpers
    def _add_concept(self, canonical_name: str, concept_type: ConceptType, domain: str,
                     synonyms: Set[str] = None, hypernyms: Set[str] = None,
                     hyponyms: Set[str] = None, definition: str = "",
                     related_processes: Set[str] = None,
                     related_properties: Set[str] = None) -> None:
        self.concepts[canonical_name] = ConceptNode(
            canonical_name=canonical_name, concept_type=concept_type, domain=domain,
            synonyms=synonyms or set(), hypernyms=hypernyms or set(),
            hyponyms=hyponyms or set(), definition=definition,
            related_processes=related_processes or set(),
            related_properties=related_properties or set(),
        )

    def _build_synonym_index(self) -> None:
        self.synonym_to_canonical: Dict[str, str] = {}
        for canonical, node in self.concepts.items():
            self.synonym_to_canonical[canonical.lower()] = canonical
            for syn in node.synonyms:
                self.synonym_to_canonical[syn.lower()] = canonical

    def _build_causal_chains(self) -> None:
        # Each tuple: (source, rel_type, target, confidence)
        chains: List[Tuple[str, RelationshipType, str, float]] = [
            # --- LASER domain: assemble the pulsed heat source term ---
            ("ndyag_laser", RelationshipType.PART_OF, "pulsed_laser_source", 0.95),
            ("pulse_energy", RelationshipType.DRIVES, "fluence", 0.95),
            ("beam_waist", RelationshipType.CONSTRAINS, "gaussian_profile", 0.90),
            ("gaussian_profile", RelationshipType.FORMS, "fluence", 0.92),
            ("repetition_rate", RelationshipType.INFLUENCES, "pulse_on_condition", 0.85),
            ("pulse_width", RelationshipType.CONSTRAINS, "pulse_on_condition", 0.88),
            ("pulse_on_condition", RelationshipType.ENABLES, "temporal_modulation", 0.85),
            ("temporal_modulation", RelationshipType.MODIFIES, "laser_heat_source_term", 0.90),
            ("pulse_energy", RelationshipType.RESULTS_IN, "average_power", 0.93),
            ("repetition_rate", RelationshipType.INFLUENCES, "average_power", 0.90),
            ("average_power", RelationshipType.RESULTS_IN, "effective_power", 0.88),
            ("effective_power", RelationshipType.RESULTS_IN, "normalization_coefficient", 0.85),
            ("normalization_coefficient", RelationshipType.COMPUTES, "laser_heat_source_term", 0.85),
            ("gaussian_profile", RelationshipType.PART_OF, "laser_heat_source_term", 0.80),
            ("laser_absorption", RelationshipType.MODIFIES, "laser_heat_source_term", 0.70),

            # --- ALLOY domain: thermodynamics & phase change ---
            ("sac305_alloy", RelationshipType.HAS_PART, "latent_heat", 0.90),
            ("sac305_alloy", RelationshipType.HAS_PART, "melting_point", 0.92),
            ("melting_point", RelationshipType.CONSTRAINS, "mushy_zone", 0.88),
            ("latent_heat", RelationshipType.ABSORBS, "mushy_zone", 0.92),
            ("enthalpy", RelationshipType.MODELS, "solid_liquid_transition", 0.90),
            ("apparent_heat_capacity", RelationshipType.APPROXIMATES, "enthalpy", 0.85),
            ("solid_liquid_transition", RelationshipType.RESULTS_IN, "melt_pool", 0.88),
            ("sac305_alloy", RelationshipType.HAS_PART, "thermal_conductivity", 0.80),
            ("sac305_alloy", RelationshipType.HAS_PART, "density", 0.80),
            ("sac305_alloy", RelationshipType.HAS_PART, "cte", 0.80),
            ("sac305_alloy", RelationshipType.HAS_PART, "youngs_modulus", 0.80),
            ("sac305_alloy", RelationshipType.HAS_PART, "poissons_ratio", 0.78),

            # --- ALLOY domain: melt-pool hydrodynamics ---
            ("melt_pool", RelationshipType.ENABLES, "navier_stokes_flow", 0.90),
            ("boussinesq_buoyancy", RelationshipType.PART_OF, "navier_stokes_flow", 0.80),
            ("thermal_gradient", RelationshipType.DRIVES, "boussinesq_buoyancy", 0.82),
            ("thermal_gradient", RelationshipType.DRIVES, "marangoni_convection", 0.90),
            ("marangoni_convection", RelationshipType.INFLUENCES, "navier_stokes_flow", 0.85),
            ("melt_viscosity", RelationshipType.CONSTRAINS, "navier_stokes_flow", 0.78),

            # --- ALLOY domain: solidification & microstructure ---
            ("cooling_rate", RelationshipType.CAUSES, "supercooling", 0.90),
            ("supercooling", RelationshipType.CONSTRAINS, "critical_nucleation_radius", 0.88),
            ("critical_nucleation_radius", RelationshipType.INFLUENCES, "nucleation_rate", 0.85),
            ("nucleation_rate", RelationshipType.RESULTS_IN, "grain_refinement", 0.88),
            ("grain_refinement", RelationshipType.REFINE, "beta_sn_grain", 0.90),
            ("grain_refinement", RelationshipType.RESULTS_IN, "grain_size", 0.85),
            ("cooling_rate", RelationshipType.INFLUENCES, "imc_morphology", 0.80),
            ("imc_morphology", RelationshipType.CORRELATES, "imc_thickness", 0.82),
            ("imc_volume_fraction", RelationshipType.QUANTIFIES, "precipitate", 0.85),
            ("solid_liquid_transition", RelationshipType.TRANSITIONS_TO, "beta_sn_grain", 0.78),
            ("solid_liquid_transition", RelationshipType.FORMS, "precipitate", 0.75),

            # --- ALLOY domain: thermomechanical response ---
            ("thermal_gradient", RelationshipType.CAUSES, "thermal_stress", 0.88),
            ("thermal_stress", RelationshipType.DRIVES, "elastoplastic_deformation", 0.85),
            ("elastoplastic_deformation", RelationshipType.RESULTS_IN, "cauchy_stress", 0.85),
            ("ramberg_osgood_model", RelationshipType.MODELS, "elastoplastic_deformation", 0.88),
            ("cauchy_stress", RelationshipType.COMPUTES, "von_mises_stress", 0.85),
            ("von_mises_stress", RelationshipType.CONSTRAINS, "yield_strength", 0.80),
            ("grain_size", RelationshipType.DEPENDS_ON, "hall_petch_strengthening", 0.88),
            ("hall_petch_strengthening", RelationshipType.STRENGTHENS, "shear_strength", 0.85),
            ("precipitate", RelationshipType.DEPENDS_ON, "precipitation_strengthening", 0.85),
            ("precipitation_strengthening", RelationshipType.STRENGTHENS, "shear_strength", 0.82),
            ("grain_refinement", RelationshipType.STRENGTHENS, "hardness", 0.84),
            ("imc_volume_fraction", RelationshipType.STRENGTHENS, "hardness", 0.78),
            ("hall_petch_strengthening", RelationshipType.STRENGTHENS, "elastic_modulus_indent", 0.75),
            ("grain_refinement", RelationshipType.STRENGTHENS, "creep_resistance", 0.80),
            ("shear_strength", RelationshipType.RESULTS_IN, "ductile_fracture", 0.70),
            ("shear_strength", RelationshipType.PART_OF, "joint_reliability", 0.90),
            ("hardness", RelationshipType.PART_OF, "joint_reliability", 0.88),
            ("creep_resistance", RelationshipType.PART_OF, "joint_reliability", 0.88),

            # --- NUMERICAL / AI framework ---
            ("fem_simulation", RelationshipType.SIMULATES, "temperature_field", 0.92),
            ("laser_heat_source_term", RelationshipType.DRIVES, "heat_flux", 0.92),
            ("temperature_field", RelationshipType.HAS_PART, "thermal_gradient", 0.90),
            ("naif_framework", RelationshipType.INTERPOLATES, "temperature_field", 0.88),
            ("cross_attention", RelationshipType.PART_OF, "naif_framework", 0.90),
            ("naif_framework", RelationshipType.MAPS, "joint_reliability", 0.70),

            # === CROSS-DOMAIN BRIDGES (laser -> alloy_microstructure) ===
            # Fig. N1 cause-effect pair: fluence (cause)  <->  enthalpy (effect)
            ("pulse_energy", RelationshipType.DRIVES, "enthalpy", 0.82),
            ("fluence", RelationshipType.CAUSES, "enthalpy", 0.85),
            ("laser_absorption", RelationshipType.INFLUENCES, "enthalpy", 0.72),
            ("fluence", RelationshipType.DRIVES, "temperature_field", 0.85),
            ("average_power", RelationshipType.DRIVES, "heat_flux", 0.80),
            ("repetition_rate", RelationshipType.INFLUENCES, "cooling_rate", 0.72),
            ("beam_waist", RelationshipType.CONSTRAINS, "thermal_gradient", 0.70),
            ("temporal_modulation", RelationshipType.MODIFIES, "temperature_field", 0.72),
            ("laser_heat_source_term", RelationshipType.DRIVES, "temperature_field", 0.95),
            ("laser_heat_source_term", RelationshipType.DRIVES, "heat_flux", 0.92),
            ("temperature_field", RelationshipType.DRIVES, "melt_pool", 0.90),
            ("temperature_field", RelationshipType.CAUSES, "thermal_gradient", 0.88),
            ("temperature_field", RelationshipType.DRIVES, "cooling_rate", 0.85),
            ("heat_flux", RelationshipType.ABSORBS, "enthalpy", 0.85),
            ("pulsed_laser_source", RelationshipType.ENABLES, "grain_refinement", 0.75),
        ]
        for source, rel_type, target, conf in chains:
            # Skip any edge whose endpoints are not both defined (safety).
            if source in self.concepts and target in self.concepts:
                self.relationships.append(
                    Relationship(source, target, rel_type, abs(float(conf)))
                )

    # --------------------------------------------------------------- queries
    def resolve_concept(self, text: str) -> Optional[str]:
        t = text.lower().strip()
        if t in self.synonym_to_canonical:
            return self.synonym_to_canonical[t]
        for cand in (t.replace("-", " "), t.replace(" ", "-"),
                     re.sub(r"\b(s|es)\b", "", t)):
            if cand in self.synonym_to_canonical:
                return self.synonym_to_canonical[cand]
        return None

    def domain_of(self, concept_key: str) -> Optional[str]:
        node = self.concepts.get(concept_key)
        return node.domain if node else None

    def cross_domain_edges(self) -> List[Relationship]:
        out = []
        for r in self.relationships:
            if self.domain_of(r.source) and self.domain_of(r.target) \
                    and self.domain_of(r.source) != self.domain_of(r.target):
                out.append(r)
        return out

    def build_nx_graph(self):
        if not _HAS_NX:
            raise RuntimeError("networkx is required for build_nx_graph()")
        G = nx.DiGraph()
        for k, node in self.concepts.items():
            G.add_node(k, concept_type=node.concept_type.value, domain=node.domain,
                       definition=node.definition)
        for r in self.relationships:
            G.add_edge(r.source, r.target, rel=r.rel_type.value,
                       confidence=r.confidence,
                       cross_domain=(self.domain_of(r.source) != self.domain_of(r.target)))
        return G


# ============================================================================
# 32 LATENT-MoE EXPERTS  (replaces BATTERY_EXPERT_LABELS) — split 12/20
# ============================================================================
LASER_SOLDERING_EXPERT_LABELS: List[str] = [
    # --- 12 laser (heat source) experts ---
    "Pulsed Laser Source", "Nd:YAG Laser", "Pulse Energy", "Pulse Width",
    "Repetition Rate", "Gaussian Beam Waist", "Laser Fluence",
    "Temporal Pulse Modulation", "Average & Effective Power",
    "Laser Absorption", "Pulse-on Gating", "Laser Heat Source Term",
    # --- 20 alloy-microstructure experts ---
    "SAC305 Alloy", "Latent Heat of Fusion", "Enthalpy-Temperature Method",
    "Mushy Zone & Phase Change", "Thermal Conductivity (T-dep)", "Density (T-dep)",
    "CTE & Poisson Ratio", "Young's Modulus (T-dep)", "Melt Pool Hydrodynamics",
    "Marangoni Convection", "Boussinesq Buoyancy", "Melt Viscosity",
    "beta-Sn Grain Structure", "Grain Refinement", "(Cu,Ni)6Sn5 IMC",
    "IMC Morphology & Thickness", "Cooling Rate", "Supercooling & Nucleation",
    "Solidification Mechanism", "Thermomechanical Response",
]
assert len(LASER_SOLDERING_EXPERT_LABELS) == 32, "expert count must stay 32 for the MoE"

# Domain assignment per expert (order must match LASER_SOLDERING_EXPERT_LABELS).
EXPERT_DOMAIN: List[str] = (
    [LASER] * 12 + [ALLOY] * 20
)
assert len(EXPERT_DOMAIN) == 32


# ============================================================================
# PROBLEM DEFINITIONS  (replaces BatteryProblem / BATTERY_PROBLEM_DEFINITIONS)
# ============================================================================
class LaserSolderingProblem(Enum):
    PULSE_PARAMETER_SELECTION = "pulse_parameter_selection"
    THERMAL_FIELD_CONTROL = "thermal_field_control"
    MELT_POOL_DYNAMICS = "melt_pool_dynamics"
    IMC_MORPHOLOGY_CONTROL = "imc_morphology_control"
    GRAIN_REFINEMENT = "grain_refinement"
    THERMOMECHANICAL_RELIABILITY = "thermomechanical_reliability"
    GENERAL = "general"
    MULTI_PROBLEM = "multi_problem"


@dataclass
class LaserSolderingProblemDefinition:
    problem_id: LaserSolderingProblem
    title: str
    scientific_description: str
    root_cause: str
    primary_domain: Optional[str]            # "laser" | "alloy_microstructure" | None
    key_concepts: List[str]
    key_relationships: List[Tuple[str, str, str]]
    solution_directions: List[str]
    relevant_descriptors: List[str]
    relevant_phenomena: List[str]
    relevant_properties: List[str]
    example_queries: List[str]
    visualization_focus: List[str]

    def get_ontology_concepts(self) -> Set[str]:
        concepts = set(self.key_concepts + self.relevant_descriptors +
                       self.relevant_phenomena + self.relevant_properties)
        for src, _, tgt in self.key_relationships:
            concepts.update([src, tgt])
        return concepts


LASER_SOLDERING_PROBLEM_DEFINITIONS: Dict[LaserSolderingProblem,
                                           LaserSolderingProblemDefinition] = {
    LaserSolderingProblem.PULSE_PARAMETER_SELECTION: LaserSolderingProblemDefinition(
        problem_id=LaserSolderingProblem.PULSE_PARAMETER_SELECTION,
        title="Pulse Parameter Selection for Laser Soldering",
        scientific_description="Selecting pulse energy, width and repetition rate to deliver the "
                              "target fluence and thermal history for SAC305 reflow.",
        root_cause="Trade-off between sufficient enthalpy to melt the ball and avoiding over-heating "
                   "/ coarse microstructure.",
        primary_domain=LASER,
        key_concepts=["pulse_energy", "pulse_width", "repetition_rate", "fluence",
                      "laser_heat_source_term"],
        key_relationships=[("pulse_energy", "DRIVES", "fluence"),
                           ("repetition_rate", "INFLUENCES", "average_power"),
                           ("temporal_modulation", "MODIFIES", "laser_heat_source_term")],
        solution_directions=["Sweep E in 0.5-8.0 mJ", "Bound tau_pulse < simulation window",
                            "Tune P_effective to ball geometry"],
        relevant_descriptors=["beam_waist", "normalization_coefficient"],
        relevant_phenomena=["laser_absorption"],
        relevant_properties=["average_power", "effective_power"],
        example_queries=["Which pulse energy gives the finest beta-Sn grains?",
                         "Trade-off between pulse width and IMC thickness"],
        visualization_focus=["pulse_sweep", "fluence_map"]),
    LaserSolderingProblem.THERMAL_FIELD_CONTROL: LaserSolderingProblemDefinition(
        problem_id=LaserSolderingProblem.THERMAL_FIELD_CONTROL,
        title="Thermal Field & Cooling-Rate Control",
        scientific_description="Managing the spatiotemporal temperature field, gradients and cooling "
                              "rate that link the laser source to the microstructure.",
        root_cause="Steep, transient gradients drive stress, Marangoni flow and solidification kinetics.",
        primary_domain=LASER,
        key_concepts=["laser_heat_source_term", "temperature_field", "thermal_gradient",
                      "cooling_rate", "heat_flux"],
        key_relationships=[("laser_heat_source_term", "DRIVES", "temperature_field"),
                           ("temperature_field", "DRIVES", "cooling_rate"),
                           ("heat_flux", "ABSORBS", "enthalpy")],
        solution_directions=["Resolve dt=1e-9 s transients", "Use enthalpy apparent-cp method",
                            "Interpolate fields with NAIF"],
        relevant_descriptors=["fem_simulation", "naif_framework"],
        relevant_phenomena=["thermal_stress", "marangoni_convection"],
        relevant_properties=["thermal_gradient"],
        example_queries=["How does pulse energy shape the cooling rate?",
                         "Predict the peak temperature field for E=7.5 mJ"],
        visualization_focus=["temperature_field", "cooling_rate_curve"]),
    LaserSolderingProblem.MELT_POOL_DYNAMICS: LaserSolderingProblemDefinition(
        problem_id=LaserSolderingProblem.MELT_POOL_DYNAMICS,
        title="Melt-Pool Hydrodynamics & Marangoni Flow",
        scientific_description="Modelling incompressible flow, buoyancy and thermocapillary convection "
                              "in the molten SAC305 pool.",
        root_cause="Thermocapillary (Marangoni) stresses dominate heat & mass transport in the melt.",
        primary_domain=ALLOY,
        key_concepts=["melt_pool", "navier_stokes_flow", "boussinesq_buoyancy",
                      "marangoni_convection", "melt_viscosity"],
        key_relationships=[("thermal_gradient", "DRIVES", "marangoni_convection"),
                           ("marangoni_convection", "INFLUENCES", "navier_stokes_flow"),
                           ("melt_viscosity", "CONSTRAINS", "navier_stokes_flow")],
        solution_directions=["Apply Boussinesq buoyancy", "Enforce Marangoni shear on free surface",
                            "Raise viscosity in solid to stop flow"],
        relevant_descriptors=["mushy_zone", "apparent_heat_capacity"],
        relevant_phenomena=["marangoni_convection", "boussinesq_buoyancy"],
        relevant_properties=["melt_viscosity"],
        example_queries=["How strong is Marangoni flow at E=7.5 mJ?",
                         "Effect of viscosity on melt-pool velocity"],
        visualization_focus=["velocity_field", "marangoni_streamlines"]),
    LaserSolderingProblem.IMC_MORPHOLOGY_CONTROL: LaserSolderingProblemDefinition(
        problem_id=LaserSolderingProblem.IMC_MORPHOLOGY_CONTROL,
        title="(Cu,Ni)6Sn5 IMC Morphology & Thickness Control",
        scientific_description="Tuning interfacial (Cu,Ni)6Sn5 morphology (granular vs needle) and "
                              "thickness via the laser thermal history.",
        root_cause="Cooling rate and peak temperature set IMC nucleation, growth and morphology.",
        primary_domain=ALLOY,
        key_concepts=["imc_morphology", "imc_thickness", "imc_volume_fraction",
                      "precipitate", "cooling_rate"],
        key_relationships=[("cooling_rate", "INFLUENCES", "imc_morphology"),
                           ("imc_morphology", "CORRELATES", "imc_thickness"),
                           ("imc_volume_fraction", "QUANTIFIES", "precipitate")],
        solution_directions=["Favour granular thin IMC via fast cooling",
                             "Limit peak temperature to curb IMC growth",
                             "Track volume fraction 2.13 vs 1.42"],
        relevant_descriptors=["solid_liquid_transition", "nucleation_rate"],
        relevant_phenomena=["supercooling"],
        relevant_properties=["imc_thickness", "imc_volume_fraction"],
        example_queries=["Why is laser IMC thinner than reflow IMC?",
                         "Pulse energy for minimum IMC thickness"],
        visualization_focus=["imc_morphology", "imc_thickness_bar"]),
    LaserSolderingProblem.GRAIN_REFINEMENT: LaserSolderingProblemDefinition(
        problem_id=LaserSolderingProblem.GRAIN_REFINEMENT,
        title="beta-Sn Grain Refinement",
        scientific_description="Achieving 20-30 striated beta-Sn grains (vs 2-5 large + twinning) "
                              "through rapid laser-driven solidification.",
        root_cause="High cooling rate raises supercooling, lowers critical nucleation radius, "
                   "boosts nucleation.",
        primary_domain=ALLOY,
        key_concepts=["beta_sn_grain", "grain_refinement", "grain_size", "cooling_rate",
                      "supercooling", "critical_nucleation_radius", "nucleation_rate"],
        key_relationships=[("cooling_rate", "CAUSES", "supercooling"),
                           ("supercooling", "CONSTRAINS", "critical_nucleation_radius"),
                           ("nucleation_rate", "RESULTS_IN", "grain_refinement"),
                           ("grain_refinement", "REFINES", "beta_sn_grain")],
        solution_directions=["Maximize cooling rate within thermal-stress limits",
                             "Promote heterogeneous nucleation",
                             "Target 20-30 striated grains"],
        relevant_descriptors=["solid_liquid_transition"],
        relevant_phenomena=["supercooling", "nucleation_rate"],
        relevant_properties=["grain_size"],
        example_queries=["How does cooling rate set beta-Sn grain count?",
                         "Pulse parameters for maximum grain refinement"],
        visualization_focus=["grain_count", "nucleation_map"]),
    LaserSolderingProblem.THERMOMECHANICAL_RELIABILITY: LaserSolderingProblemDefinition(
        problem_id=LaserSolderingProblem.THERMOMECHANICAL_RELIABILITY,
        title="Thermomechanical Reliability of SAC305 Microbumps",
        scientific_description="Linking microstructure to shear strength, hardness, elastic modulus "
                              "and creep resistance of the joint.",
        root_cause="Microstructure (grain size, IMC fraction) sets strength via Hall-Petch and "
                   "precipitation strengthening.",
        primary_domain=ALLOY,
        key_concepts=["shear_strength", "hardness", "elastic_modulus_indent", "creep_resistance",
                      "hall_petch_strengthening", "precipitation_strengthening", "joint_reliability"],
        key_relationships=[("hall_petch_strengthening", "STRENGTHENS", "shear_strength"),
                           ("precipitation_strengthening", "STRENGTHENS", "shear_strength"),
                           ("grain_refinement", "STRENGTHENS", "hardness"),
                           ("shear_strength", "PART_OF", "joint_reliability")],
        solution_directions=["Optimise grain size + IMC fraction jointly",
                             "Validate via shear + nanoindentation",
                             "Use NAIF to map pulse -> reliability"],
        relevant_descriptors=["grain_size", "imc_volume_fraction"],
        relevant_phenomena=["ductile_fracture", "thermal_stress"],
        relevant_properties=["shear_strength", "hardness", "creep_resistance"],
        example_queries=["Which microstructure maximises shear strength?",
                         "Pulse parameters for best creep resistance"],
        visualization_focus=["reliability_radar", "strengthening_breakdown"]),
    LaserSolderingProblem.GENERAL: LaserSolderingProblemDefinition(
        problem_id=LaserSolderingProblem.GENERAL,
        title="General Pulsed Laser Soldering Inquiry",
        scientific_description="General question about pulsed laser soldering of SAC305.",
        root_cause="N/A",
        primary_domain=None,
        key_concepts=["pulsed_laser_source", "joint_reliability"],
        key_relationships=[],
        solution_directions=[],
        relevant_descriptors=[],
        relevant_phenomena=[],
        relevant_properties=[],
        example_queries=["What factors control SAC305 joint reliability under laser soldering?"],
        visualization_focus=["general_overview"]),
    LaserSolderingProblem.MULTI_PROBLEM: LaserSolderingProblemDefinition(
        problem_id=LaserSolderingProblem.MULTI_PROBLEM,
        title="Multi-Problem Laser Soldering Inquiry",
        scientific_description="Inquiry spanning multiple core problems across both domains.",
        root_cause="N/A",
        primary_domain=None,
        key_concepts=[],
        key_relationships=[],
        solution_directions=[],
        relevant_descriptors=[],
        relevant_phenomena=[],
        relevant_properties=[],
        example_queries=[],
        visualization_focus=["multi_problem_comparison"]),
}


# ============================================================================
# QDWA 2-DOMAIN CONFIG  (replaces FOUR_CATEGORIES / 4-rotor strategy)
# ============================================================================
TWO_DOMAINS: List[str] = [LASER, ALLOY]
DOMAIN_DISPLAY: Dict[str, str] = {
    LASER: "Laser Heat Source",
    ALLOY: "Alloy Microstructure",
}
_LABEL2KEY: Dict[str, str] = {v: k for k, v in DOMAIN_DISPLAY.items()}

# Map each LaserSolderingProblem to its primary domain (None = both / general).
PROBLEM_TO_DOMAIN: Dict[LaserSolderingProblem, Optional[str]] = {
    LaserSolderingProblem.PULSE_PARAMETER_SELECTION: LASER,
    LaserSolderingProblem.THERMAL_FIELD_CONTROL: LASER,
    LaserSolderingProblem.MELT_POOL_DYNAMICS: ALLOY,
    LaserSolderingProblem.IMC_MORPHOLOGY_CONTROL: ALLOY,
    LaserSolderingProblem.GRAIN_REFINEMENT: ALLOY,
    LaserSolderingProblem.THERMOMECHANICAL_RELIABILITY: ALLOY,
    LaserSolderingProblem.GENERAL: None,
    LaserSolderingProblem.MULTI_PROBLEM: None,
}


# ============================================================================
# HIERARCHY LABEL BUILDER  (replaces _HIERARCHY_PARENTS — here parent=domain)
# ============================================================================
def _build_hierarchy_parents() -> Dict[str, Tuple[Optional[str], int]]:
    """Assign every ontology concept to its domain label (tier 1)."""
    onto = DomainOntology()
    parents: Dict[str, Tuple[Optional[str], int]] = {}
    for k, node in onto.concepts.items():
        parents[k] = (DOMAIN_DISPLAY.get(node.domain, "General"), 1)
    return parents


_HIERARCHY_PARENTS: Dict[str, Tuple[Optional[str], int]] = _build_hierarchy_parents()

# Seed concepts per domain for the QDWA category-anchor embeddings.
DOMAIN_SEEDS: Dict[str, List[str]] = {k: [] for k in TWO_DOMAINS}
for _c, (_parent, _tier) in _HIERARCHY_PARENTS.items():
    _key = _LABEL2KEY.get(_parent)
    if _key in DOMAIN_SEEDS:
        DOMAIN_SEEDS[_key].append(_c)
# Guarantee non-empty seeds (fall back to the domain's expert labels).
if not DOMAIN_SEEDS[LASER]:
    DOMAIN_SEEDS[LASER] = [c.replace(" ", "_").lower()
                           for c in LASER_SOLDERING_EXPERT_LABELS[:12]]
if not DOMAIN_SEEDS[ALLOY]:
    DOMAIN_SEEDS[ALLOY] = [c.replace(" ", "_").replace("(", "").replace(")", "").lower()
                           for c in LASER_SOLDERING_EXPERT_LABELS[12:]]


def get_hierarchy_label(concept_key: str, style: str = "arrow") -> str:
    leaf = concept_key.replace("_", " ").title()
    entry = _HIERARCHY_PARENTS.get(concept_key)
    if entry is None or entry[0] is None or style == "leaf":
        return leaf
    parent_label = entry[0]
    sep = " -> " if style == "arrow" else " [" if style == "bracket" else " · "
    if style == "bracket":
        return f"{parent_label}{sep}{leaf}]"
    return f"{parent_label}{sep}{leaf}"


# ============================================================================
# KEYWORDS & PATTERNS  (replaces BATTERY_KEYWORDS / *_MAPPING / *_PATTERNS)
# ============================================================================
LASER_SOLDERING_KEYWORDS: List[str] = [
    "pulsed laser", "nd:yag", "pulse energy", "pulse width", "repetition rate",
    "beam waist", "gaussian", "fluence", "fourier", "harmonic", "average power",
    "effective power", "laser absorption", "heat source", "sac305", "sn-ag-cu",
    "snagcu", "sn-3.0ag-0.5cu", "latent heat", "enthalpy", "apparent heat capacity",
    "mushy zone", "phase change", "melting point", "thermal conductivity", "density",
    "cte", "poisson", "young's modulus", "melt pool", "navier-stokes", "buoyancy",
    "marangoni", "thermocapillary", "viscosity", "beta-sn", "beta-tin", "grain",
    "grain refinement", "grain size", "hall-petch", "imc", "intermetallic",
    "(cu,ni)6sn5", "cu6sn5", "morphology", "imc thickness", "volume fraction",
    "precipitate", "orowan", "cooling rate", "supercooling", "undercooling",
    "nucleation", "solidification", "thermal stress", "elastoplastic",
    "ramberg-osgood", "von mises", "yield strength", "shear strength", "hardness",
    "vickers", "nanoindentation", "elastic modulus", "creep", "ductile fracture",
    "joint reliability", "fem", "elmer", "temperature field", "thermal gradient",
    "naif", "cross attention", "heat flux",
]

DESCRIPTOR_MAPPING: Dict[str, str] = {
    r'pulsed\s*laser|nd:yag|ndyag|pulse\s*energy|pulse\s*width|repetition\s*rate|'
    r'beam\s*waist|gaussian|fluence|fourier|harmonic|average\s*power|effective\s*power|'
    r'laser\s*absorption|heat\s*source\s*term': LASER,
    r'sac305|sn.?ag.?cu|sn-3\.0ag-0\.5cu|latent\s*heat|enthalpy|apparent\s*heat|'
    r'mushy\s*zone|phase\s*change|melting\s*point|thermal\s*conductivity|density|'
    r'\bcte\b|poisson|young.?s?\s*modulus|melt\s*pool|navier.?stokes|buoyancy|'
    r'marangoni|thermocapillary|viscosity|beta.?sn|beta.?tin|grain|hall.?petch|'
    r'imc|intermetallic|\(cu,?ni\)?6sn5|cu6sn5|morphology|volume\s*fraction|'
    r'precipitate|orowan|cooling\s*rate|supercooling|undercooling|nucleation|'
    r'solidification|thermal\s*stress|elastoplastic|ramberg.?osgood|von\s*mises|'
    r'yield\s*strength|shear\s*strength|hardness|vickers|nanoindentation|'
    r'elastic\s*modulus|creep|ductile\s*fracture|joint\s*reliability|fem|elmer|'
    r'temperature\s*field|thermal\s*gradient|naif|cross\s*attention|heat\s*flux': ALLOY,
}

QUANTITATIVE_PATTERNS: List[str] = [
    r'\bpulsed\s+laser\s+soldering\b',
    r'\bnd:?\s*yag\b',
    r'\bpulse\s+energy\b',
    r'\bpulse\s+(?:width|duration)\b',
    r'\brepetition\s+rate\b',
    r'\bbeam\s+waist\b',
    r'\bgaussian\s+(?:profile|beam|fluence)\b',
    r'\bfluenc(?:e|y)\b',
    r'\baverage\s+power\b',
    r'\beffective\s+power\b',
    r'\bsn-3\.0ag-0\.5cu\b',
    r'\bsac305\b',
    r'\blatent\s+heat\b',
    r'\bapparent\s+heat\s+capacity\b',
    r'\bmushy\s+zone\b',
    r'\bmarangoni\s+convection\b',
    r'\bboussinesq\b',
    r'\bbeta-sn\b',
    r'\bgrain\s+refinement\b',
    r'\bhall-petch\b',
    r'\b\(cu,?\s*ni\)6sn5\b',
    r'\bcu6sn5\b',
    r'\bimc\s+(?:morphology|thickness)\b',
    r'\bcooling\s+rate\b',
    r'\bsupercooling\b',
    r'\bnucleation\s+rate\b',
    r'\bramberg-osgood\b',
    r'\bvon\s+mises\b',
    r'\byield\s+strength\b',
    r'\bshear\s+strength\b',
    r'\b(vickers|nanoindentation)\s+hardness\b',
    r'\bcreep\s+(?:resistance|displacement)\b',
    r'\bjoint\s+reliability\b',
]


# ============================================================================
# EXTRACTION & CATEGORIZATION  (replaces the battery_* helpers)
# ============================================================================
def extract_laser_soldering_concepts_from_text(text: str) -> List[str]:
    found: List[str] = []
    low = text.lower()
    for pat in QUANTITATIVE_PATTERNS:
        for m in re.finditer(pat, low):
            found.append(m.group(0).strip())
    for kw in LASER_SOLDERING_KEYWORDS:
        if kw in low:
            found.append(kw)
    # dedupe preserving order
    seen, out = set(), []
    for c in found:
        c2 = c.strip()
        if c2 and c2 not in seen:
            seen.add(c2); out.append(c2)
    return out


def is_valid_laser_soldering_concept(concept: str) -> bool:
    c = concept.lower().strip()
    if not c:
        return False
    for pat in QUANTITATIVE_PATTERNS:
        if re.search(pat, c):
            return True
    return c in LASER_SOLDERING_KEYWORDS


def normalize_laser_soldering_concept(concept: str) -> str:
    c = concept.lower().strip()
    c = re.sub(r"[^a-z0-9]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def categorize_laser_soldering_concept(concept: str) -> str:
    c = concept.lower()
    for pattern, domain in DESCRIPTOR_MAPPING.items():
        if re.search(pattern, c):
            return domain
    return "general"


def get_concept_domain_color(concept: str, cmap_colors=None) -> str:
    if cmap_colors:
        return cmap_colors[hash(concept) % len(cmap_colors)]
    dom = categorize_laser_soldering_concept(concept)
    return DOMAIN_COLORS.get(dom, "#7f7f7f")


# ============================================================================
# GRAPH ASSEMBLY & SUMMARY
# ============================================================================
_ontology_singleton: Optional[DomainOntology] = None


def get_ontology() -> DomainOntology:
    global _ontology_singleton
    if _ontology_singleton is None:
        _ontology_singleton = DomainOntology()
    return _ontology_singleton


def build_concept_graph():
    return get_ontology().build_nx_graph()


def summary() -> Dict[str, object]:
    onto = get_ontology()
    n_laser = sum(1 for n in onto.concepts.values() if n.domain == LASER)
    n_alloy = sum(1 for n in onto.concepts.values() if n.domain == ALLOY)
    cross = onto.cross_domain_edges()
    type_counts: Dict[str, int] = {}
    for n in onto.concepts.values():
        type_counts[n.concept_type.value] = type_counts.get(n.concept_type.value, 0) + 1
    rel_counts: Dict[str, int] = {}
    for r in onto.relationships:
        rel_counts[r.rel_type.value] = rel_counts.get(r.rel_type.value, 0) + 1
    return {
        "total_concepts": len(onto.concepts),
        "laser_concepts": n_laser,
        "alloy_microstructure_concepts": n_alloy,
        "total_relationships": len(onto.relationships),
        "cross_domain_bridges": len(cross),
        "concept_type_counts": type_counts,
        "relationship_type_counts": rel_counts,
        "expert_labels": len(LASER_SOLDERING_EXPERT_LABELS),
        "problems": len(LASER_SOLDERING_PROBLEM_DEFINITIONS),
        "domains": TWO_DOMAINS,
    }


# ============================================================================
# INTEGRATION NOTES  (for the host drone app)
# ============================================================================
INTEGRATION_NOTES = """
HOW TO WIRE THIS MODULE INTO THE EXISTING DRONE APP
---------------------------------------------------
1. Replace the domain-content blocks in the drone script with this module
   (or `from pulsed_laser_snagcu_concept_graph import *`). The symbol map at the
   top of this file lists every original name and its new name.

2. QDWA: 4-rotor -> 2-domain.
   - Replace `FOUR_CATEGORIES` with `TWO_DOMAINS` everywhere (it is len-2 now).
   - Replace `CATEGORY_DISPLAY` with `DOMAIN_DISPLAY`.
   - In QDWAWeightAllocator / render_category_weight_gauge / the 4-metric Streamlit
     row (render_qdwa_tab etc.), loop over `TWO_DOMAINS` instead of FOUR_CATEGORIES;
     the weight vector becomes 2-D (w_laser, w_microstructure). The soft-membership
     math (Eq. 2-5) is unchanged in form, only the dimensionality drops 4 -> 2.
   - Update EXPERT_CATEGORY -> EXPERT_DOMAIN (12 laser + 20 alloy).

3. The generic infrastructure is domain-agnostic and needs NO content change:
   LatentMoEKGExtractor / SparseGraphSAGE / QDWAWeightAllocator / the pyvis &
   plotly renderers / GraphRAGAnswerGenerator / DynamicOntologyExpander /
   IncrementalGraphBuilder. They consume ConceptNode/Relationship/DomainOntology,
   whose shapes are unchanged.

4. ConceptNode gains one field: `domain` ("laser" | "alloy_microstructure").
   Any code that unpacks ConceptNode positionally must add this field; keyword
   construction (the norm) is unaffected.

5. LatentMoE expert count stays 32 (LASER_SOLDERING_EXPERT_LABELS) so the
   SparseGraphSAGE / LatentMoEKGExtractor dimensions are unchanged.
"""


if __name__ == "__main__":
    s = summary()
    print("=" * 64)
    print("PULSED LASER SOLDERING SAC305 — TWO-DOMAIN CONCEPT GRAPH")
    print("=" * 64)
    print(f"Domains                 : {s['domains']}")
    print(f"Total concepts          : {s['total_concepts']}")
    print(f"  laser                 : {s['laser_concepts']}")
    print(f"  alloy_microstructure  : {s['alloy_microstructure_concepts']}")
    print(f"Total relationships     : {s['total_relationships']}")
    print(f"Cross-domain bridges    : {s['cross_domain_bridges']}")
    print(f"Latent-MoE experts      : {s['expert_labels']} (12 laser + 20 alloy)")
    print(f"Problem definitions     : {s['problems']}")
    print("-" * 64)
    print("Concept-type counts:")
    for k, v in sorted(s["concept_type_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:18s}: {v}")
    print("Top relationship types:")
    for k, v in sorted(s["relationship_type_counts"].items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {k:18s}: {v}")
    print("-" * 64)
    onto = get_ontology()
    print("Cross-domain bridges (laser <-> alloy_microstructure):")
    for r in onto.cross_domain_edges():
        print(f"  [{onto.domain_of(r.source):>19s}] {r.source:24s} "
              f"--{r.rel_type.value:14s}--> [{onto.domain_of(r.target):>19s}] {r.target}")
    # sanity assertions
    assert s["laser_concepts"] + s["alloy_microstructure_concepts"] == s["total_concepts"], \
        "every concept must belong to exactly one of the two domains"
    assert s["expert_labels"] == 32
    print("\nOK: ontology built, two-domain invariant satisfied, 32 experts present.")
