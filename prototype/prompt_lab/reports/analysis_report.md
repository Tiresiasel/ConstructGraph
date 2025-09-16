## Model Output Consistency Report

Generated: 2025-08-19 16:32:25

### Chen和Miller - 1994 - Competitive attack, retaliation and performance An expectancy‐valence framework.pdf

- Baseline model: **gpt-5**
- Constructs:
  - gpt-5: count=16, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=9, jaccard_vs_baseline=0.39
- Relationships:
  - gpt-5: count=16, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=10, jaccard_vs_baseline=0.44
- Measurements:
  - gpt-5: count=10, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=7, jaccard_vs_baseline=0.06

- Metadata consistency
  - Title: gpt-5 exact and correct; gpt-5-mini garbled word order (“…FRAMEWORK PERFORMANCE: AN EXPECTANCY-VALENCE”). Divergence.
  - Authors: Same names; casing differs only.
  - Year/Journal/Research type: Consistent (1994; Strategic Management Journal; Quantitative).
  - DOI: Present in gpt-5 (10.1002/smj.4250150202); missing in gpt-5-mini.
  - Research context: Both capture U.S. airlines 1979–1986; gpt-5-mini adds sample counts (780 actions; 222 responses).

- Construct coverage
  - Overlap: attack visibility; response difficulty; attack centrality; number of retaliatory responses; response ratio; financial performance; potential benefit of the attack.
  - Missing in gpt-5-mini (vs gpt-5): response ease (inverse of difficulty); interaction product constructs (visibility×ease, centrality×ease, visibility×centrality, and three-way); performance sub-constructs split out (revenue/RPM, operating profit/RPM, profit margin, S&P rating).
  - Additional in gpt-5-mini: expectancy and valence (theoretical primitives), which duplicate the roles of response difficulty and centrality but were not separate constructs in gpt-5.
  - Overall: gpt-5 is more granular and hypothesis-aligned; gpt-5-mini is more conceptual and aggregates constructs.

- Relationships (subject → object; status/evidence/effect)
  - Overlap: H1 (visibility → responses, +, supported); H2 (difficulty → responses, −, supported); H3 (centrality → responses, +, supported); H5 (response ratio → financial performance, −, supported).
  - Divergences on H4 (interactions):
    - gpt-5 specifies all four: vis×ease (+, supported), cent×ease (+, supported), vis×cent (ns), vis×ease×cent (ns).
    - gpt-5-mini collapses to a single “products of …” relationship with “Mixed” evidence and omits the explicit insignificance of vis×cent and the three-way, and the positive significance of the two specific two-way terms.
  - gpt-5-mini lacks explicit boundary conditions and detailed quotes for several links; gpt-5 includes them.

- Measurements and naming
  - Overlap with naming differences: 
    - Attack visibility (“Attack visibility composite index” vs “Visibility composite (standardized average)”).
    - Response difficulty (“composite index” vs “composite (5-scale)”).
    - Centrality (“Centrality of attack measure” vs “Centrality (proportion of annual passengers affected)”).
    - Responses (“Response count per action” vs “Count of responding competitors”).
    - Response ratio consistent.
  - Present only in gpt-5: separate measures for revenue/RPM, operating profit/RPM, profit margin, S&P rating.
  - Present only in gpt-5-mini: a combined “performance factor” description with loadings; S&P not a standalone measurement entry.

- Severity and score
  - Severity: Medium (granularity gaps on interactions and measurements; core story consistent).
  - Consistency score: 64/100.

- Recommendations
  - Prompt: Instruct models to enumerate all interaction terms (each two-way and the three-way) as separate constructs and relationships with effect directions and significance.
  - Prompt: Require listing all performance indicators both individually and any composite/factor used.
  - Post-processing: Normalize synonyms (e.g., map “response ease” ↔ inverse of difficulty; merge measurement name variants).
  - Post-processing: Validate titles against DOI metadata and backfill missing DOI.


### Melkonyan 等 - 2018 - Collusion in Bertrand vs. Cournot Competition A Virtual Bargaining Approach.pdf

- Baseline model: **gpt-5**
- Constructs:
  - gpt-5: count=15, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=11, jaccard_vs_baseline=0.44
- Relationships:
  - gpt-5: count=7, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=3, jaccard_vs_baseline=0.00
- Measurements:
  - gpt-5: count=0, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=3, jaccard_vs_baseline=0.00

- 1) Metadata consistency
  - Title/DOI/year: Fully consistent across models (same title string; DOI; 2018).
  - Authors: Same three authors and order; only case differences in the heuristics view (not in JSON).
  - Journal: Same venue; gpt-5 uses “Management Science”; gpt-5-mini uses “MANAGEMENT SCIENCE” (case variance).
  - Research type/context: Both “Conceptual,” similar context; gpt-5 adds more detail (status-quo, VBE rationale).

- 2) Construct coverage
  - Overlaps: virtual bargaining equilibrium/VBE, feasible agreement, worst payoff, Cournot competition, Bertrand competition, collusive outcome, best response, Nash equilibrium (mini has null definition).
  - Missing in gpt-5-mini (vs gpt-5): maxmin expected utility, Nash bargaining solution (as a construct), joint profits, price, quantity, strategic complementarity, strategic substitutability.
  - Additional in gpt-5-mini: virtual bargaining (umbrella concept), minimum feasible worst payoff, agreement.
  - Consistency: Moderate. Mini omits core theoretical underpinnings (maxmin/Nash bargaining) and key comparative-statics concepts, weakening explanatory coherence.

- 3) Relationships (subject→object; status/evidence/effect)
  - Overlaps in substance: Both capture Proposition 1 (VBE=NE in Cournot) and Proposition 2 (in Bertrand, pN < pV and profits: VBE > NE).
  - Differences:
    - gpt-5 encodes 7 relationships, splitting compound statements into atomic links (e.g., pV vs pN and pV vs p*; joint profits VBE vs NE and VBE vs collusive; plus conceptual claims about feasibility set and Pareto undominance).
    - gpt-5-mini encodes only 3, labels them “Hypothesized/Quantitative” despite being theoretical propositions, and collapses pN < pV < p* into a single relation (thus missing the explicit VBE vs collusive and VBE profits vs collusive links).
    - Mini omits conceptual relationships (feasible-set contains NE; VBE Pareto-undominated by NE).
  - Net: Substantial structural and typing divergence despite overlapping quotes.

- 4) Measurements
  - gpt-5: None (appropriate for conceptual paper).
  - gpt-5-mini: Adds three “measurements” that are actually formal definitions (worst payoff function, feasibility condition, Nash-product selection). This is a taxonomy drift; not empirical measurements. No naming conflicts, but redundancy with constructs.

- 5) Divergence severity and score
  - Severity: High (relationship structure/typing mismatch; missing core constructs; spurious measurements).
  - Consistency score: 52/100.

- 6) Recommendations
  - Prompt:
    - Enforce controlled vocab and ontology: require atomic relationships; label theoretical results as “Proposition” with evidence_type “Qualitative.”
    - Require inclusion of foundational constructs when VBE is used (maxmin expected utility; Nash bargaining solution; strategic complements/substitutes; price/quantity).
    - Disallow null definitions for core constructs (e.g., Nash equilibrium).
    - Ask to split compound inequalities (pN < pV < p*) and profit comparisons into separate relations.
    - Clarify that “measurements” are empirical instruments; formal definitions should remain in constructs.
  - Post-processing:
    - Normalize casing and journal names.
    - Parse compound quotes into multiple relationships.
    - Map synonyms (virtual bargaining ↔ VBE) and merge duplicates.
    - Validate status/evidence fields against a ruleset for theoretical papers.


### Shi 等 - 2024 - Bricks without Straw Overcoming Resource Limitations to Architect Ecosystem Leadership.pdf

- Baseline model: **gpt-5**
- Constructs:
  - gpt-5: count=19, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=0, jaccard_vs_baseline=0.00
- Relationships:
  - gpt-5: count=9, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=0, jaccard_vs_baseline=0.00
- Measurements:
  - gpt-5: count=0, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=0, jaccard_vs_baseline=1.00

- Metadata consistency
  - Title: gpt-5 is clean; gpt-5-mini appends author/affiliation text to the title (contamination). Divergence.
  - Authors: Same three authors, same order in both models. Consistent.
  - Year/Journal/DOI: All match (2024; Academy of Management Journal; doi.org/10.5465/amj.2021.1440). Consistent.
  - Research type/context: Both “Qualitative.” Context differs in granularity (gpt-5-mini provides 9-year scope and data modalities; gpt-5 is briefer). Acceptable variation.

- Construct coverage
  - gpt-5: 19 constructs (e.g., foothold resource, identity movement, user community, democratization of innovation, emancipatory/pioneering/orchestrator visions, demand-/supply-side support and synergies, social proof, derivative ecosystem, second-/first-tier complementors, triadic/dyadic interactions, conservatory, mutualistic approach).
  - gpt-5-mini: 0 constructs.
  - Notable missing (gpt-5-mini vs gpt-5): All 19 above. No additional constructs present in either model.
  - Overall consistency: Very low (Jaccard 0.0).

- Relationships (subject -> object; status/evidence/effect)
  - gpt-5: 9 qualitative empirical results, all Positive effects (e.g., foothold resource -> identity movement; democratization of innovation -> identity movement; emancipatory vision -> user community; demand-side support -> supply-side legitimacy; social proof -> credibility; incubating fledgling firms -> conservatory; evoking emotions -> resonance; supply-side synergy -> attract prominent partners; demand-side synergy -> expand integrated offerings).
  - gpt-5-mini: 0 relationships; thus misses all 9 found by gpt-5.
  - Overlaps: None. Discrepancies: Complete omission by gpt-5-mini.

- Measurements
  - Both models: No measurements extracted. Consistent; no naming conflicts.

- Divergence severity and consistency score
  - Severity: High (complete omission of constructs and relationships by gpt-5-mini; title contamination).
  - Consistency score: 30/100.

- Recommendations
  - Prompting:
    - Allow extraction of core claims/mechanisms when no explicit Hypotheses/Propositions exist (accept anchors like “Core_Claim,” “Mechanism,” “Microprocess,” or section headers).
    - Require a minimum set: at least N constructs and M relationships when a process model is present; justify if none.
    - Instruct strict metadata parsing: title should exclude author/affiliation lines; capture affiliations separately.
  - Post-processing:
    - Normalize titles (trim lines after first sentence-case title; remove ALLCAPS artifacts).
    - Canonicalize author order against DOI/metadata lookup when possible.
    - Validate emptiness: if constructs/relationships are zero but abstract contains cue terms (e.g., “we introduce,” “mechanism”), trigger a re-extraction pass.
    - Standardize relationship fields (status/evidence/effect) and map synonyms (e.g., orchestrator vs orchestrator vision) to canonical terms.


### Strategic Management Journal - 2001 - Park - Guanxi and organizational dynamics  organizational networking in Chinese firms.pdf

- Baseline model: **gpt-5**
- Constructs:
  - gpt-5: count=14, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=12, jaccard_vs_baseline=0.37
- Relationships:
  - gpt-5: count=36, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=21, jaccard_vs_baseline=0.00
- Measurements:
  - gpt-5: count=12, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=9, jaccard_vs_baseline=0.00

Assessment of cross-model JSON consistency for: Strategic Management Journal (2001) Park & Luo — “Guanxi and organizational dynamics”

1) Metadata consistency
- Overlap: DOI, title, authors, year, journal, research_type, is_replication_study fully match; research context consistent in substance (sample, location, period, survey/regression).
- Differences: None material.

2) Construct coverage
- Overlap (same ideas, naming varies): guanxi (and its two dimensions), ownership structure, location, strategic orientation/market-oriented strategy, size/firm size, technological skills, managerial capabilities, firm age (length of operation[s]), sales growth, net profit growth.
- Missing in gpt-5-mini: control constructs included by gpt-5 (industry growth index, industry type).
- Naming conflicts:
  - guanxi_with_business_community vs “guanxi with the business community”
  - guanxi_with_government_authorities vs “guanxi with government authorities”
  - strategic orientation vs “market-oriented strategy”
  - size vs “firm size”
  - length of operation vs “length of operations”
- Overall: High semantic overlap; gpt-5-mini is less granular on controls.

3) Relationships comparison
- Overlap (direction/status):
  - H1–H3: Positive effects (nonstate ownership, less-open location, market orientation) on guanxi; both models agree. gpt-5 distinguishes horizontal vs vertical; gpt-5-mini sometimes collapses to generic “guanxi/guanxi utilization.”
  - H4–H6: For business-community guanxi, size/tech skills/managerial skills negatively related (i.e., small/poorer capabilities → more guanxi); gpt-5-mini encodes via subjects like “small organizations/poor skills → guanxi (positive)” which is directionally consistent.
  - H7: Age not significant; both agree.
  - H8: Guanxi → sales growth positive; → net profit growth insignificant; both agree and report for both guanxi dimensions (explicit in gpt-5; implicit/aggregated in gpt-5-mini).
- Differences:
  - Granularity: gpt-5 enumerates hypotheses and empirical results per guanxi dimension with significance notes; gpt-5-mini often uses generic “guanxi utilization,” fewer explicit control mentions, and subjects as categories (e.g., “nonstate-owned firms”) rather than constructs (“ownership structure”).
  - Missing explicit insignificance for vertical guanxi with organizational factors in some gpt-5-mini entries (not always per-dimension).

4) Measurements presence and naming
- Overlap: Scales for both guanxi dimensions, strategic orientation, tech/managerial capabilities, size, age, sales growth, net profit growth; Likert/categorical details broadly align; reliability notes similar.
- Missing in gpt-5-mini: measurements for ownership structure (dummy), location (open/nonopen dummy), industry growth index, industry type.
- Naming conflicts mirror constructs (snake_case vs spaced labels; “market-oriented strategy” vs “strategic orientation”).

5) Severity and score
- Severity: Medium. Core findings align; divergences are mainly naming, dimensional granularity, and omission of control-variable measurements.
- Consistency score: 68/100.

6) Recommendations
- Prompt:
  - Enforce canonical construct IDs and snake_case names; require mapping of synonyms (e.g., firm size→size).
  - Require splitting guanxi into explicit dimensions for all hypotheses/results; disallow generic “guanxi utilization.”
  - Specify that subjects should be constructs (ownership_structure) not categories (nonstate-owned firms); encode category via measurement/coding.
  - Mandate inclusion of all controls and their operationalizations (industry_growth_index, industry_type, ownership, location).
- Post-processing:
  - Alias normalization and string canonicalization (lowercase, underscores).
  - Dimension-expansion for any generic “guanxi” relations to both horizontal/vertical when text indicates.
  - Sign-harmonization when categorical subjects are used (translate “small orgs → guanxi +” to “size → guanxi −”).
  - Validate presence of expected H1–H8 per dimension and flag missing controls.


### Strategic Management Journal - 2013 - Lipparini - From core to periphery and back  A study on the deliberate shaping of.pdf

- Baseline model: **gpt-5**
- Constructs:
  - gpt-5: count=13, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=0, jaccard_vs_baseline=0.00
- Relationships:
  - gpt-5: count=5, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=0, jaccard_vs_baseline=0.00
- Measurements:
  - gpt-5: count=0, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=0, jaccard_vs_baseline=1.00

Comparison of JSON outputs for: Strategic Management Journal (2014), Lipparini et al. “From core to periphery and back”

1) Metadata consistency
- Matches across models: DOI (10.1002/smj.2110), title, authors, year (2014), journal, is_replication_study (false). Only casing differs.
- Divergence: research_type — gpt-5: Qualitative vs gpt-5-mini: Mixed-Methods.
- Context granularity: gpt-5 provides detailed research_context (industry, sample, period); gpt-5-mini is terse.

2) Construct coverage
- gpt-5: 13 constructs extracted (e.g., knowledge-enhancing practices; core firms; peripheral firms; knowledge flows; interfirm dyads; interfirm networks; interfirm knowledge exchange and creation; R&D; operations; upstream/downstream activities; bidirectional knowledge flows; collaborative knowledge; inter-supplier flows).
- gpt-5-mini: 0 constructs (explicitly filtered out due to “no hypotheses found”).
- Notable missing in gpt-5-mini: all 13 baseline constructs (listed above). No additional constructs introduced.
- Overall consistency: very low (Jaccard 0.0).

3) Relationships (subject -> object; status/evidence/effect)
- gpt-5: 5 qualitative empirical relationships aligned with four-phase model:
  - core firms -> peripheral firms (Phase 1, outbound flows)
  - peripheral firms -> core firms (Phase 2, inbound flows)
  - bidirectional knowledge flows -> collaborative knowledge (Phase 3, co-creation)
  - core firms -> knowledge flows between peripheral firms (Phase 4, inter-supplier)
  - knowledge-enhancing practices -> interfirm knowledge exchange and creation (enabling condition)
- gpt-5-mini: 0 relationships (scope restriction).
- Overlap: none. Discrepancy: gpt-5-mini misses all five, including status/evidence details (all qualitative empirical results).

4) Measurements
- Both: none reported. No naming conflicts.

5) Divergence severity and consistency score
- Severity: High.
- Consistency score: 28/100 (metadata mostly aligned; core content—constructs/relationships—absent in gpt-5-mini).

6) Recommendations
- Prompting:
  - Remove hypothesis-only gating; explicitly extract constructs/relationships from qualitative findings, phased models, and narrative empirical results.
  - Instruct to capture enabling practices and role-based entities (core vs peripheral firms) as constructs even without formal hypotheses.
  - Clarify research_type taxonomy and decision rules; default to Qualitative when evidence is interviews/case-process without systematic quant analysis.
- Post-processing:
  - Normalize casing and map research_type to controlled vocabulary; flag Mixed-Methods when both systematic quantitative and qualitative evidence exist.
  - Backfill constructs from relationships and vice versa; ensure phases produce subject->object edges.
  - Ensemble merge: prefer superset of constructs/relations across models; run rule to prevent empty outputs when rich qualitative results are present.


### Strategic Management Journal - 2014 - Kehoe - Lighting the way or stealing the shine  An examination of the duality in star.pdf

- Baseline model: **gpt-5**
- Constructs:
  - gpt-5: count=13, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=7, jaccard_vs_baseline=0.54
- Relationships:
  - gpt-5: count=12, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=12, jaccard_vs_baseline=1.00
- Measurements:
  - gpt-5: count=6, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=5, jaccard_vs_baseline=0.00

- Task 1: Metadata consistency
  - Overlaps: DOI, title text (including OCR spacing “WA Y”), authors (same two, different casing), publication year (2015), research_type (Quantitative), non-replication.
  - Differences:
    - Journal: gpt-5 = “StrategicManagementJournal” (concatenated full), gpt-5-mini = “Strat.Mgmt.J.” (abbrev.).
    - Authors casing: Proper case (gpt-5) vs all caps (gpt-5-mini).
    - Research context phrasing differs but consistent in substance.
  - Assessment: High consistency; differences are formatting/abbreviation.

- Task 2: Construct coverage
  - Overlap (core): presence_of_a_star; innovative_productivity; innovative_leadership_among_non_star_scientists; breadth_of_star_expertise; star_collaborative_strength; mutual_dependence; power_imbalance.
  - Missing in gpt-5-mini (present in gpt-5): expert_power; workflow_centrality; knowledge_spillovers; information_overload; structural_holes; vicarious_learning.
  - Additions: None unique to gpt-5-mini; gpt-5 provides broader theoretical scaffolding constructs.
  - Overall: Core constructs align; gpt-5-mini omits several theory-side constructs.

- Task 3: Relationships (subject→object; status/evidence/effect)
  - Overlaps: All eight hypothesis dyads (H1a–H4b) present in both with consistent directions and confirmations:
    - presence_of_a_star → innovative_productivity (H1a, +, supported)
    - presence_of_a_star → innovative_leadership (H1b, −, supported)
    - breadth_of_star_expertise → innovative_productivity (H2a, diminishing +, supported)
    - breadth_of_star_expertise → innovative_leadership (H2b, hypothesized U-shape; results: increasing at decreasing rate)
    - star_collaborative_strength → both outcomes (+, supported)
    - Moderation by star_collaborative_strength on breadth → both outcomes (linear +, squared −, supported)
  - Differences:
    - Granularity: gpt-5 supplies boundary_conditions, controls, and Panel_FE design; gpt-5-mini has fewer boundary details, labels design as “Regression”, and adds supporting quotes.
    - Nonlinear flags: gpt-5 marks “Other” or “U-shaped”; gpt-5-mini often leaves non_linear_type null despite describing quadratic patterns.

- Task 4: Measurements
  - Overlap in concepts; naming differs:
    - innovative_productivity: “Citation-weighted patent count” (gpt-5) vs “Patent-weighted productivity” (gpt-5-mini).
    - presence_of_a_star: “Star firm dummy …” vs “Star identification via inventor performance score.”
    - breadth_of_star_expertise: Herfindahl-based breadth (same formula, different label).
    - star_collaborative_strength: normalized co-invention frequency (same formula, different label).
    - innovative_leadership: index vs unique non-star inventor count (same construct).
  - Missing in gpt-5-mini: the variant “citation-weighted patent count excluding star co-invented patents.”

- Task 5: Divergence severity and score
  - Severity: Medium-Low (core content consistent; omissions in theoretical constructs and some measurement variants).
  - Consistency score: 85/100.

- Task 6: Recommendations
  - Prompt: Require explicit capture of theoretical constructs supporting hypotheses; mandate non_linear_type labeling when quadratic terms are reported; request boundary_conditions and controls lists.
  - Post-processing: Normalize journal titles and author casing; map synonymous measurement names to canonical labels; detect and align measurement variants (e.g., excluding-star patents); standardize design_type taxonomy (e.g., Panel_FE vs generic Regression).


### Strategic Management Journal - 2014 - Skilton - Competition network structure and product market entry.pdf

- Baseline model: **gpt-5**
- Constructs:
  - gpt-5: count=13, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=5, jaccard_vs_baseline=0.12
- Relationships:
  - gpt-5: count=6, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=6, jaccard_vs_baseline=0.00
- Measurements:
  - gpt-5: count=10, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=4, jaccard_vs_baseline=0.00

Paper: Strategic Management Journal (Skilton & Bernardes) – “Competition network structure and product market entry”

1) Metadata consistency
- Title/DOI/Year: Match across models (DOI 10.1002/smj.2318; 2015; identical title).
- Journal: gpt-5 correct (“Strategic Management Journal”); gpt-5-mini concatenated (“StrategicManagementJournal”).
- Authors: gpt-5 clean (“Paul F. Skilton”, “Ednilson Bernardes”); gpt-5-mini includes capitalization/footnote artifacts (“PAUL F . SKILTON1,2*”, “EDNILSON BERNARDES2,3”).
- Research type/context: Both “Quantitative”; gpt-5 provides rich context; gpt-5-mini a brief phrase.

2) Construct coverage
- Overlap (core): ego network size, ego network density, competitor diversity, competition network(s), DV (product market entry rate).
- Missing in gpt-5-mini (present in gpt-5): direct competition (tie definition), ego network (concept), multimarket contact, market centrality, learned market entry capability, type certificate holder (TCDS), rotorcraft share, multiengine share.
- Naming/alias divergence: “product_market_entry_rate” vs “rate of product market entry”; “competitor_diversity” vs “diversity of competitors in ego network”; “competition network” vs “competition networks”.
- Overall: gpt-5-mini under-covers controls and operational precursors.

3) Relationships (subject -> object; status/evidence/effect)
- Overlaps (substantive): 
  - H1: ego network size -> product market entry (Positive; supported).
  - H2: ego network density -> product market entry (Negative hypothesized; empirically Insignificant).
  - H3: competitor diversity -> product market entry (Positive; supported).
- Differences:
  - gpt-5-mini adds post hoc nonlinearity (Inverted U for H1); gpt-5 purposefully excludes post hoc.
  - gpt-5 lists boundary conditions and controls; gpt-5-mini omits.
  - Theory labels differ (gpt-5: structural holes, network architecture; gpt-5-mini: theory of brokerage, social network theory).
  - String mismatches (DV and predictor names) prevent automatic matching though semantics align.

4) Measurements
- Overlap: Both capture DV as STC count (2000–2009) and operationalize degree, density, diversity (CV of multimarket contact).
- Conflicts/gaps:
  - gpt-5 provides instruments (FAA STC, TCDS), direct competition operationalization, multimarket contact details, and control measures; gpt-5-mini lacks instruments and omits all control measurements.
  - Naming inconsistencies as above impede mapping.

5) Divergence severity and score
- Severity: Medium.
- Consistency score: 62/100.

6) Recommendations
- Enforce canonical term mapping and alias normalization (e.g., map “rate of product market entry” to “product_market_entry_rate”; “diversity of competitors in ego network” to “competitor_diversity”).
- Require extraction of control variables and their measurements.
- Standardize metadata cleaning (strip author footnotes/affiliations; fix journal spacing).
- Distinguish and optionally exclude post hoc findings or tag them explicitly.
- Mandate instruments/source fields for all measurements; add minimal completeness checks.
- Use a post-processor to harmonize singular/plural and snake_case vs plain text names.


### hernandez-et-al-2017-acquisitions-node-collapse-and-network-revolution.pdf

- Baseline model: **gpt-5**
- Constructs:
  - gpt-5: count=18, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=15, jaccard_vs_baseline=0.38
- Relationships:
  - gpt-5: count=23, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=14, jaccard_vs_baseline=0.00
- Measurements:
  - gpt-5: count=9, jaccard_vs_baseline=1.00
  - gpt-5-mini: count=9, jaccard_vs_baseline=0.00

Paper: Hernandez & Menon (2018) “Acquisitions, Node Collapse, and Network Revolution”

1) Metadata consistency
- Match: DOI, title, authors, journal, year (2018); non-replication flag.
- Divergence: research_type
  - gpt-5: Conceptual
  - gpt-5-mini: Quantitative
  - Note: The paper is best labeled Simulation/Computational modeling (agent-based) rather than purely conceptual or purely empirical.

2) Construct coverage
- Overlaps (substantively aligned): node collapse(s)/acquisition; performance; degree centrality; constraint/structural holes; modularity; synergies (internal/network); transmission (tr); distance weighting (d); firm size.
- Notable missing in gpt-5-mini (vs gpt-5): internal resources; network resources; transmission (as a construct label); shortest path length; number of firms remaining; standard deviation of firm size; tie additions.
- Additions in gpt-5-mini: Erdős–Rényi initial network; resource production; network externalities.
- Naming conflicts hampering alignment: “node collapses” vs “node collapse”; “transmission” vs “resource transmission (tr)”; “cost of being a large firm” vs “cost of size (c)”; “number of firms remaining” vs “network shrinkage”.
- Overall consistency: Moderate on content, weak on canonicalization (Jaccard ~0.38).

3) Relationships (subject→object; direction/status)
- Clear overlaps (content): 
  - Node collapse → acquirer performance (+); degree (+); constraint (−); modularity (−).
  - Node collapse → others’ degree (−); others’ constraint (+); others’ performance (+).
  - Network synergies moderating network shrinkage: slow under increasing returns; facilitate under decreasing returns.
  - Parameter effects: c increases → more firms remain (+), higher modularity (+), lower constraint (−); d and tr show nonlinear/mixed patterns.
- Important discrepancies:
  - gpt-5 has distinct, atomic links (including tie-additions comparisons); gpt-5-mini sometimes conflates multiple outcomes in one relation (e.g., “number of node collapses / inequality / constraint / modularity”).
  - Object naming mismatches (“number of firms remaining” vs “rate of network shrinkage”) prevent exact matches; gpt-5-mini omits tie-additions relationships.
  - Result counts differ (gpt-5: 23 vs gpt-5-mini: 14); heuristic Jaccard ~0 due to label variance despite conceptual overlap.

4) Measurements presence and naming
- Overlaps: performance equation; degree; constraint; modularity; d, tr parameters.
- gpt-5-only: shortest path length; number of firms remaining; standard deviation of firm size; firm size (as measurement).
- gpt-5-mini-only: Erdős–Rényi initialization; resource production; explicit “cost of size (c)” parameter.
- Naming conflicts mirror constructs (e.g., “transmission” vs “resource transmission”; modularity naming variants).

5) Severity and score
- Severity: High (relationship canonicalization and several construct/measurement mismatches).
- Consistency score: 58/100.

6) Recommendations
- Enforce a controlled vocabulary with canonical IDs and alias lists (e.g., node_collapse, number_of_firms_remaining).
- Require atomic relationships (one outcome per relation) and role qualifiers (acquirer vs others; ego vs global).
- Standardize parameter/metric namespaces: parameter.c, parameter.d, parameter.tr; outcome.modularity; outcome.num_firms.
- Normalize singular/plural and synonyms during post-processing (lemmatize; alias mapping).
- Add guidance to classify research_type as “Simulation/Computational modeling” when agent-based results are central.
- Validate outputs with schema-based matching to split composite relations and map semantically equivalent objects.

