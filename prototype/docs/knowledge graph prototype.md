# Technical Blueprint for an Academic Knowledge Graph Prototype

## 1. Project Goal & Core Philosophy

### 1.1. Project Goal

This document provides a comprehensive and executable technical blueprint for building a dynamic **Academic Knowledge Graph Prototype**. The core objective is to validate a data model and processing pipeline capable of accurately capturing, aligning, and querying complex academic knowledge in the social sciences, including constructs, theories, and their interrelationships across various research methodologies.

**Key Focus Areas:**

1. **Versatile Data Model Design**: Establish a graph database schema that can express complex relationships from quantitative, qualitative, and review articles.
2. **Intelligent Data Processing Pipeline**: Design an automated pipeline with a sophisticated entity resolution workflow to transform diverse paper texts into a unified, coherent graph.
3. **Core Functionality Validation**: Prove the model's effectiveness through complex query examples that span different research types.

### 1.2. Core Design Philosophy: A Dual-Database Architecture

For the prototype phase, we select a dual-database architecture as our cornerstone:

- **Labeled Property Graph (LPG) Database (Neo4j)**: This serves as the primary store for our structured knowledge network. Its intuitive model (nodes, relationships, properties) is ideal for representing and querying the complex, interconnected nature of academic knowledge. The core modeling pattern of **Relationship Reification** (treating relationships as `RelationshipInstance` nodes) will be implemented here.
- **Vector Database**: This serves as a specialized index for semantic understanding. It will store vector embeddings of construct definitions, enabling efficient similarity searches. This is the engine that powers our automated entity resolution workflow.

This dual approach combines the strengths of both technologies: the logical, explicit connections of a graph database with the semantic, implicit understanding of a vector database.

---

## 2. Core Graph Database Schema Design (Revised)

This is the backbone of the entire prototype. A well-designed schema is the foundation for all subsequent functionalities.

### 2.1. Node Definitions

Nodes represent the core entities within the knowledge graph.

| Node Label | Description | Key Properties |
|------------|-------------|----------------|
| `Paper` | Represents the metadata of an academic publication. | `uuid`, `doi`, `title`, `authors`, `publication_year`, `journal`, `research_type`, `research_context`, `is_replication_study` (Boolean) |
| `Author` | An author of a paper. | `uuid`, `full_name`, `orcid` |
| `Theory` | The theoretical foundation used to explain relationships between constructs. | `uuid`, `name`, `description` |
| `CanonicalConstruct` | **Normalized Construct**: Represents a disambiguated and standardized abstract academic concept. | `uuid`, `preferred_name`, `description`, `status` (`Verified`, `Provisional`) |
| `Term` | **Term**: The specific vocabulary or phrase used in a particular paper to refer to a construct. | `uuid`, `text` |
| `Definition` | The specific textual definition of a `Term` within a particular `Paper` (unique per paper × term × text). | `uuid`, `text`, `context_snippet`, `term_text`, `paper_uid` |
| `Measurement` | The method/scale used to operationalize a construct in a particular `Paper` (unique per paper × construct × name). | `uuid`, `name`, `description`, `scale_items` (JSON), `instrument`, `scoring_procedure`, `formula`, `reliability`, `validity`, `context_adaptations`, `construct_term`, `paper_uid` |
| `RelationshipInstance` | **Relationship Instance**: Reifies a specific relationship claim made in a paper into a node. **This is the core of the design.** | `uuid`, `description`, `context_snippet`, `status` (Enum), `evidence_type` (Enum), `effect_direction` (Enum), `non_linear_type` (Enum), `statistical_details` (JSON), `is_meta_analysis` (Boolean), `qualitative_finding` (String), `supporting_quote` (String), `is_validated_causality` (Boolean), `boundary_conditions` (String), `replication_outcome` (Enum) |

### 2.2. Edge (Relationship) Definitions

Relationships represent the connections between nodes.

| Relationship Type | Start Node | End Node | Description |
|------------------|------------|----------|-------------|
| `AUTHORED_BY` | `Paper` | `Author` | Connects a paper to its authors. |
| `CITES` | `Paper` | `Paper` | Indicates that one paper cites another. |
| `USES_TERM` | `Paper` | `Term` | A paper uses a specific term. |
| `HAS_DEFINITION` | `Term` | `Definition` | A term has a definition within a specific paper. |
| `IS_REPRESENTATION_OF` | `Term` | `CanonicalConstruct` | A specific term is an instance of a normalized construct. |
| `HAS_DIMENSION` | `CanonicalConstruct` | `CanonicalConstruct` | Indicates that one construct is a sub-dimension of another. |
| `ESTABLISHES` | `Paper` | `RelationshipInstance` | A paper establishes or proposes a specific relationship claim. |
| `HAS_SUBJECT` | `RelationshipInstance` | `CanonicalConstruct` | Points to the subject construct of a relationship claim (A in A → B). |
| `HAS_OBJECT` | `RelationshipInstance` | `CanonicalConstruct` | Points to the object construct of a relationship claim (B in A → B). |
| `HAS_MODERATOR` | `RelationshipInstance` | `CanonicalConstruct` | Points to a moderating variable in the relationship claim. |
| `HAS_MEDIATOR` | `RelationshipInstance` | `CanonicalConstruct` | Points to a mediating variable in the relationship claim. |
| `APPLIES_THEORY` | `RelationshipInstance` | `Theory` | A relationship claim applies a theory as its foundation. |
| `USES_MEASUREMENT` | `CanonicalConstruct` | `Measurement` | A construct is measured using a specific method. |
| `DEFINED_IN` | `Definition` | `Paper` | A definition originates from a specific paper. |
| `MEASURED_IN` | `Measurement` | `Paper` | A measurement method is used or proposed in a specific paper. |
| `IS_SIMILAR_TO` | `CanonicalConstruct` | `CanonicalConstruct` | **NEW**: Indicates semantic similarity between constructs with metadata. |

---

## 3. Data Processing Pipeline (Prototype Version)

This pipeline describes the complete process of transforming a paper into the knowledge graph, incorporating the advanced two-stage entity resolution workflow.

### 3.1. Step 1: Structured Information Extraction via LLM

This step uses an LLM to perform the initial, broad extraction of all knowledge elements from the raw text.

#### 3.1.1. LLM Extraction Prompt

This prompt is designed to extract all potential knowledge components from a paper.

```text
# ROLE AND GOAL
You are an extremely meticulous, detail-oriented research analyst and knowledge engineer. Your sole task is to act as an information extraction engine, strictly following the provided JSON Schema to convert unstructured text from the provided academic paper into structured knowledge.

# CORE DIRECTIVES
1. **Fidelity to Source**: All information you extract must originate directly from the provided paper text. You are strictly forbidden from making inferences, guessing, or adding external knowledge. If information is not present, use `null` or an empty array `[]`.
2. **Schema is Supreme**: Your output must be, and can only be, a single, valid JSON object that strictly conforms to the provided JSON Schema. Do not add any explanations, comments, or Markdown formatting outside the JSON object.
3. **Comprehensive Coverage**: You must make your best effort to comprehensively scan the entire paper to ensure all required information is extracted without omission.

# MULTI-STEP CHAIN-OF-THOUGHT PROCESS

Execute the following tasks in strict sequence:

### **Phase 1: Initial Scan & Global Classification**
1. Read the entire paper to understand its overall structure and purpose.
2. Extract the top-level metadata to populate the `paper_metadata` section.
3. **Crucially, classify the `research_type`** based on its methodology. Is it `Quantitative`, `Qualitative`, `Conceptual`, `Review`, `Meta-Analysis`, or `Mixed-Methods`?
4. **Identify Replication Study**: Determine if the paper is explicitly a `Replication Study` and set the `is_replication_study` boolean accordingly.

### **Phase 2: Relationship-First Identification and Construct Harvesting**
This is the most critical phase. Your primary goal is to find all relationship claims first, as this guarantees the capture of all relevant constructs.
1. **Systematic Search for Claims**: Meticulously scan the paper, paying special attention to the **Abstract, Hypothesis Development, Propositions, and Results sections**. Your goal is to identify every sentence or passage that proposes or tests a relationship between concepts.
2. **Harvest Constructs from Claims**: For **every single relationship claim** you identify, immediately extract all the construct `term`s mentioned within that claim (e.g., the subject, object, moderators, mediators).

   **CRITICAL - What to Extract vs What NOT to Extract:**
   
   **EXTRACT (Theoretical Constructs):**
   - ✅ **Core theoretical concepts** that appear in propositions and hypotheses
   - ✅ **Latent variables** that represent underlying theoretical phenomena
   - ✅ **Abstract concepts** that need to be operationalized
   - ✅ **Theoretical mediators and moderators** that explain relationships
   - ✅ **ONLY constructs that have relationships with other constructs**
   
   **CRITICAL RULE**: Extract ONLY constructs that participate in relationships (as subject, object, mediator, or moderator). Skip isolated constructs that don't relate to anything else.
   
   **EXTRACT (Related Measurements):**
   - ✅ **Measurement methods** for constructs mentioned in hypotheses/propositions
   - ✅ **Operationalization details** for theoretical constructs
   - ✅ **Scales and instruments** used to measure the constructs
   
   **EXTRACT (Related Measurements):**
   - ✅ **Measurement methods** for constructs mentioned in hypotheses/propositions
   - ✅ **Operationalization details** for theoretical constructs
   - ✅ **Scales and instruments** used to measure the constructs
   
   **EXTRACT (Theoretically Relevant Control Variables):**
   - ✅ **Control variables** explicitly mentioned in hypotheses/propositions
   - ✅ **Control variables** with clear theoretical justification
   - ✅ **Control variables** that authors explain why they matter theoretically
   
   **DO NOT EXTRACT (Unrelated Variables):**
   - ❌ **Measurement variables** NOT related to hypothesis/proposition constructs
   - ❌ **Statistical parameters** (e.g., "p-value", "beta coefficient")
   - ❌ **Purely methodological control variables** without theoretical explanation
   - ❌ **Methodological artifacts** (e.g., "sample size", "survey items")
   - ❌ **Mathematical parameters** used in models but not as theoretical constructs
   
   **Rule**: Extract constructs from hypotheses/propositions AND their corresponding measurements, but avoid unrelated operational variables

   **IMPORTANT - Construct Name Extraction Rules:**
   - Extract ONLY the full construct name, NOT abbreviations in parentheses
   - **Examples of CORRECT extraction:**
     - ✅ "organizational commitment" (NOT "organizational commitment (OC)")
     - ✅ "network constraint" (NOT "network constraint (NC)")
     - ✅ "structural holes" (NOT "structural holes (SH)")
   - **Examples of INCORRECT extraction:**
     - ❌ "organizational commitment (OC)"
     - ❌ "network constraint (NC)"
     - ❌ "structural holes (SH)")
   - **Rule**: Remove all parenthetical abbreviations and keep only the complete construct name
3. **Compile a Master List**: Create a master list of all construct `term`s harvested from all relationship claims. This list is the definitive source of all important constructs in the paper.

   **IMPORTANT**: Only include constructs that actually participate in relationships. If a construct is mentioned but doesn't relate to anything else, exclude it to avoid reader confusion.

### **Phase 3: Construct De-duplication, Dimension Identification & Definition Search**
1. **De-duplicate**: Create a unique set of construct `term`s from the master list compiled in Phase 2.
2. **Identify Dimensions**: Re-examine the paper, especially the literature review and theory sections. If the paper describes a construct (e.g., "Organizational Justice") as being composed of multiple dimensions (e.g., "Distributive Justice," "Procedural Justice"), you must capture these hierarchical links and populate the `construct_dimensions` array.
3. **Targeted Definition Search**: For **each unique term**, now go back through the paper and search for its most explicit, verbatim `definition` and `context_snippet`.

   **IMPORTANT - Validate Construct vs Measurement:**
   - **For Constructs**: Verify the term appears in propositions, hypotheses, or theoretical discussions
   - **For Measurements**: Extract measurement methods for constructs mentioned in hypotheses/propositions
   - **Key Question**: Is this measurement directly related to a construct in the hypothesis/proposition?
   - **Focus on relevance**: Extract measurements that operationalize the theoretical constructs
   - **Avoid unrelated variables**: Skip measurements that don't correspond to hypothesis/proposition constructs

   **CRITICAL - Detailed Definition & Measurement Extraction:**
   - **Definitions must be comprehensive**: Extract the FULL definition, not just a brief mention
   - **Context must be rich**: Include enough context so someone unfamiliar with the paper can understand the concept
   - **Examples**: If the paper provides examples, include them in the definition
   - **Mathematical notation**: If the construct involves mathematical formulas or parameters, extract them completely
   - **Operational details**: Include how the construct is operationalized or measured
   - **Boundary conditions**: Extract any limitations or conditions mentioned for the construct
4. **Populate Constructs Array**: Populate the `constructs` array in the final JSON with every unique term and its found definition. If a term involved in a relationship has no explicit definition, it **must** still be included in this array (its `definition` field can be `null`).

   **VALIDATION STEP**: Before finalizing, verify that every construct in the array participates in at least one relationship. Remove any constructs that don't have relationships to avoid reader confusion.

### **Phase 4: In-depth, Methodology-Aware Relationship Characterization**
Now, return to the relationship claims identified in Phase 2 and characterize them in detail.
1. **Identify General Properties**:
   - Clearly identify the `subject_term` and `object_term`.
   - Determine the `status`: is it a `Hypothesized` claim or an `Empirical_Result`?
   - Find any `supporting_theories`, `moderators`, and `mediators`.
   - **Extract Boundary Conditions**: Extract any `boundary_conditions` the authors set for the hypothesis or finding (e.g., "only in high-tech industries").
   - **Extract Replication Outcome**: If the paper is a replication study, determine and extract the `replication_outcome` for key relationships (`Successful`, `Failed`, or `Mixed`).
   - **Extract Theoretically Relevant Control Variables**: Only extract control variables that:
     - Are explicitly mentioned in hypotheses/propositions, OR
     - Have clear theoretical justification explained by the authors, OR
     - Are discussed as theoretically important for understanding the main relationship

2. **Branch Extraction by Evidence Type**:
   - **If the evidence is Quantitative**:
     - Set `evidence_type` to `Quantitative`.
     - Extract `effect_direction`, `non_linear_type`, `statistical_details`, `is_meta_analysis`, and `is_validated_causality`.
     - Leave qualitative fields as `null`.
   - **If the evidence is Qualitative**:
     - Set `evidence_type` to `Qualitative`.
     - Extract the `qualitative_finding` and `supporting_quote`.
     - Leave all quantitative fields as `null`.

### **Phase 5: Measurement Extraction**
For constructs that are operationalized in the methodology section, extract their specific `measurement` methods.

**CRITICAL - Comprehensive Measurement Extraction:**
- **Extract complete measurement procedures**: Include step-by-step operationalization details
- **Include measurement scales**: Extract the specific scales, questionnaires, or instruments used
- **Mathematical formulas**: If measurements involve formulas or calculations, extract them completely
- **Validation details**: Include reliability, validity, and other psychometric properties mentioned
- **Examples of items**: If the paper provides sample questionnaire items, include them
- **Scoring procedures**: Explain how responses are converted to construct scores
- **Context-specific details**: Include any industry-specific or context-specific measurement adaptations

### **Phase 6: Citation Parsing**
1. Locate the References section.
2. Parse each reference to extract its key information.
3. Populate the `citations` array.

# OUTPUT FORMAT
The final output MUST be a single, valid JSON object that strictly conforms to the following JSON Schema.

<JSON_SCHEMA>
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Academic Paper Extraction for Prototype",
  "description": "Schema for structured information extracted from a single academic paper.",
  "type": "object",
  "properties": {
    "paper_metadata": {
      "type": "object",
      "properties": {
        "doi": { "type": "string" },
        "title": { "type": "string" },
        "authors": { "type": "array", "items": { "type": "string" } },
        "publication_year": { "type": "integer" },
        "journal": { "type": "string" },
        "research_type": { "type": "string", "enum": ["Quantitative", "Qualitative", "Conceptual", "Review", "Meta-Analysis", "Mixed-Methods"] },
        "research_context": { "type": "string" },
        "is_replication_study": { "type": "boolean" }
      },
      "required": ["title", "authors", "publication_year", "research_type"]
    },
    "constructs": { "type": "array", "items": { "type": "object", "properties": { "term": { "type": "string" }, "definition": { "type": ["string", "null"] }, "context_snippet": { "type": ["string", "null"] } }, "required": ["term"] } },
    "construct_dimensions": {
      "type": "array",
      "description": "Hierarchical links between constructs and their dimensions.",
      "items": { "type": "object", "properties": { "parent_construct": { "type": "string" }, "dimension_construct": { "type": "string" } }, "required": ["parent_construct", "dimension_construct"] }
    },
    "relationships": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "subject_term": { "type": "string" },
          "object_term": { "type": "string" },
          "status": { "type": "string", "enum": ["Hypothesized", "Empirical_Result"] },
          "evidence_type": { "type": "string", "enum": ["Quantitative", "Qualitative"] },
          "effect_direction": { "type": "string", "enum": ["Positive", "Negative", "Non-linear", "Insignificant", "Mixed","U shape", "Inverted U shape", "S shape", "Other"] },
          "non_linear_type": { "type": "string", "enum": ["U-shaped", "Inverted_U-shaped", "S-shaped", "Other"] },
          "is_validated_causality": { "type": "boolean" },
          "is_meta_analysis": { "type": "boolean" },
          "statistical_details": { "type": "object", "properties": { "p_value": { "type": "number" }, "beta_coefficient": { "type": "number" } } },
          "qualitative_finding": { "type": "string" },
          "supporting_quote": { "type": "string" },
          "boundary_conditions": { "type": "string", "description": "Specific conditions under which the relationship holds." },
          "replication_outcome": { "type": "string", "enum": ["Successful", "Failed", "Mixed"], "description": "The outcome if the paper is a replication study." },
          "context_snippet": { "type": "string" },
          "supporting_theories": { "type": "array", "items": { "type": "string" } },
          "moderators": { "type": "array", "items": { "type": "string" } },
          "mediators": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["subject_term", "object_term", "status"]
      }
    },
    "measurements": { "type": "array", "items": { "type": "object", "properties": { "construct_term": { "type": "string" }, "name": { "type": "string" }, "details": { "type": "string" } }, "required": ["construct_term", "name"] } },
    "citations": { "type": "array", "items": { "type": "object", "properties": { "raw_text": { "type": "string" }, "authors": { "type": "array", "items": { "type": "string" } }, "year": { "type": "integer" }, "title": { "type": "string" } }, "required": ["raw_text"] } }
  },
  "required": ["paper_metadata", "constructs"]
}
</JSON_SCHEMA>

<PAPER_TEXT>
[Insert the full text of the paper to be processed here]
</PAPER_TEXT>
```

#### 3.1.1 (v2) Prompt – Layered Core+Secondary Scope and Strict Rules (Supersedes Above)

The following prompt specification supersedes the previous version and is the single source of truth for extraction behavior.

```text
# ROLE AND GOAL
You are an extremely meticulous, detail-oriented research analyst and knowledge engineer. Your sole task is to act as an information extraction engine, strictly following the provided JSON Schema to convert unstructured text from the provided academic paper into structured knowledge.

# CORE DIRECTIVES
1. Fidelity to Source – Only extract what is explicitly stated in the paper text. If information is not present, use null or [].
2. Schema is Supreme – Output must be a single valid JSON object conforming strictly to the JSON Schema. No commentary outside JSON.
3. Comprehensive Coverage – Scan the full paper. However, obey the scope policy below to control inclusion.
4. Titles and Journal Names – For paper_metadata.title, extract verbatim as printed (keep prefixes/suffixes, punctuation, diacritics, casing). For paper_metadata.journal, extract the full official journal name exactly as shown (no abbreviations). 
5. DOI Formatting – If the DOI appears as a URL (e.g., https://doi.org/10.xxxx/abc), return everything after the URL scheme (remove http:// or https://), e.g., doi.org/10.xxxx/abc. If a bare DOI string appears (10.xxxx/abc), return it as-is.

# SCOPE POLICY (Layered: Core + Secondary)
- Core (mandatory): ONLY extract constructs, relationships, and measurements that are explicitly tied to Hypotheses/Propositions, including Hypothesis Development sections and Results that explicitly confirm/refute the same.
- Secondary (allowed, minimal):
  - Measurements for core constructs, even if described in Methods/Measures, provided they can be explicitly mapped to a core construct.
  - Results that explicitly confirm or refute a specific hypothesis/proposition and can be linked to its identifier.
- Out of scope: Exploratory/ancillary relationships, general literature review claims, background theories not used for the core hypotheses/propositions.

# PROCESS
## Phase 1: Initial Scan & Global Classification
1. Read the entire paper.
2. Populate paper_metadata. Classify research_type.
3. Identify replication status.

## Phase 2: Relationship-First Identification and Construct Harvesting (Core-only)
1. Systematically search the Hypotheses, Hypothesis Development, Propositions, and Results sections. Identify ONLY claims that belong to a Hypothesis or Proposition (including their development) or explicitly confirm/refute them in Results.
2. For every core relationship claim, harvest all construct terms (subject, object, moderators, mediators) and capture identifiers (e.g., H1, H1a, P2) where available.

Name Hygiene (Global):
- Always output the full term/name and drop any parenthetical abbreviations across constructs/terms/measurements.
  - Use "distance" not "distance (d)"; "return on assets" not "return on assets (ROA)"; "research and development intensity" not "R&D intensity".

## Phase 3: Construct De-duplication, Dimensions & Definition Search
1. De-duplicate the core term set.
2. Identify dimensions when explicitly described.
3. Extract comprehensive definitions and context_snippet for each unique term where available.

## Phase 4: Relationship Characterization (Methodology-Aware)
1. Populate relationship fields (status, evidence_type, effect_direction, non_linear_type, etc.) and include origin_source, hypothesis_id, section_header, line_index_range.
2. STRICT rules:
   - Moderators: Only extract as moderator when explicitly labeled as moderation by the authors (e.g., "X moderates A→B"). Do not infer from generic interactions.
   - Non-linear Effects: Only set a non-linear type when explicitly stated (e.g., "U-shaped", "Inverted U-shaped", "S-shaped"). Do NOT infer solely from polynomial terms or curvature tests.

## Phase 5: Measurement Extraction (Secondary allowed)
1. Extract measurement methods ONLY for constructs in the core scope.
2. Explicitly link measurements to construct_term and, where applicable, hypothesis_id.

## Phase 6: (Removed) Citation Parsing
Do not extract citations in this version.

# OUTPUT FORMAT (JSON Schema)
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Academic Paper Extraction v2",
  "type": "object",
  "properties": {
    "paper_metadata": {
      "type": "object",
      "properties": {
        "doi": {"type": "string"},
        "title": {"type": "string"},
        "authors": {"type": "array", "items": {"type": "string"}},
        "publication_year": {"type": "integer"},
        "journal": {"type": "string"},
        "research_type": {"type": "string", "enum": ["Quantitative","Qualitative","Conceptual","Review","Meta-Analysis","Mixed-Methods"]},
        "research_context": {"type": "string"},
        "is_replication_study": {"type": "boolean"}
      },
      "required": ["title","authors","publication_year","research_type"]
    },
    "constructs": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "term": {"type": "string"},
          "definition": {"type": ["string","null"]},
          "context_snippet": {"type": ["string","null"]}
        },
        "required": ["term"]
      }
    },
    "construct_dimensions": {
      "type": "array",
      "items": {"type": "object", "properties": {"parent_construct": {"type": "string"}, "dimension_construct": {"type": "string"}}, "required": ["parent_construct","dimension_construct"]}
    },
    "relationships": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "subject_term": {"type": "string"},
          "object_term": {"type": "string"},
          "status": {"type": "string", "enum": ["Hypothesized","Empirical_Result"]},
          "evidence_type": {"type": "string", "enum": ["Quantitative","Qualitative"]},
          "effect_direction": {"type": "string", "enum": ["Positive","Negative","Non-linear","Insignificant","Mixed","U shape","Inverted U shape","S shape","Other"]},
          "non_linear_type": {"type": "string", "enum": ["U-shaped","Inverted_U-shaped","S-shaped","Other"]},
          "is_validated_causality": {"type": "boolean"},
          "is_meta_analysis": {"type": "boolean"},
          "statistical_details": {"type": "object", "properties": {"p_value": {"type": "number"}, "beta_coefficient": {"type": "number"}}},
          "qualitative_finding": {"type": "string"},
          "supporting_quote": {"type": "string"},
          "boundary_conditions": {"type": "string"},
          "replication_outcome": {"type": "string", "enum": ["Successful","Failed","Mixed"]},
          "context_snippet": {"type": "string"},
          "origin_source": {"type": "string", "enum": ["Hypothesis","Proposition","Hypothesis_Development","Results-Confirming"]},
          "hypothesis_id": {"type": ["string","null"]},
          "section_header": {"type": ["string","null"]},
          "line_index_range": {"type": ["string","null"]},
          "supporting_theories": {"type": "array", "items": {"type": "string"}},
          "moderators": {"type": "array", "items": {"type": "string"}},
          "mediators": {"type": "array", "items": {"type": "string"}},
          "controls": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["subject_term","object_term","status"]
      }
    },
    "measurements": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "construct_term": {"type": "string"},
          "name": {"type": "string"},
          "details": {"type": "string"},
          "instrument": {"type": ["string","null"]},
          "scale_items": {"type": ["array","string","null"], "items": {"type": "string"}},
          "scoring_procedure": {"type": ["string","null"]},
          "formula": {"type": ["string","null"]},
          "reliability": {"type": ["string","null"]},
          "validity": {"type": ["string","null"]},
          "context_adaptations": {"type": ["string","null"]},
          "origin_source": {"type": "string", "enum": ["Methods-Operationalization","Hypothesis_Development"]},
          "hypothesis_id": {"type": ["string","null"]}
        },
        "required": ["construct_term","name"]
      }
    },
    "core_theories": {
      "type": "array",
      "description": "Theories explicitly used to ground core hypotheses/propositions.",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "canonical_id": {"type": ["string","null"]},
          "role": {"type": "string", "enum": ["primary","secondary"]},
          "linked_hypotheses": {"type": "array", "items": {"type": "string"}},
          "grounding_phrase": {"type": ["string","null"]},
          "evidence": {"type": ["string","null"]},
          "confidence": {"type": ["number","null"]}
        },
        "required": ["name"]
      }
    }
  },
  "required": ["paper_metadata","constructs"]
}
```

#### 3.1.2. LLM Adjudication Prompt

This is a second, distinct prompt used in the entity resolution workflow (Step 3.2). It is a focused, comparative task.

```text
# ROLE AND GOAL
You are a highly discerning academic expert with deep ontological reasoning skills. Your task is to determine if two construct definitions, extracted from different academic papers, refer to the exact same underlying concept.

# CORE DIRECTIVES
1. **Analyze Semantics Deeply**: Do not just perform a surface-level keyword match. Analyze the core meaning, scope, and nuance of each definition.
2. **Binary Decision**: Your final answer must be a single JSON object with a single key, "is_same_construct", and a boolean value (`true` or `false`).
3. **No Extraneous Text**: Do not provide any explanation or reasoning outside of the JSON object.

# INPUTS

<DEFINITION_A>
[Insert the definition text of the new construct here]
</DEFINITION_A>

<DEFINITION_B>
[Insert the definition text of the candidate construct from the database here]
</DEFINITION_B>

# OUTPUT FORMAT
Your entire output must be only the following JSON structure:
{
  "is_same_construct": true
}
```

### 3.2. Step 2: Mapping, Entity Resolution, and Storage

After receiving the initial JSON from the LLM, a script will execute the following two-stage workflow to intelligently write the data to the graph.

**Detailed Logic (Python Pseudocode):**

```python
import uuid
# Assume 'driver' is an established Neo4j driver instance
# Assume 'vector_db' is a client for a local vector database (e.g., ChromaDB, FAISS)
# Assume 'llm_adjudicate' is a function that calls the LLM with the adjudication prompt

def process_paper_json_to_graph(jsonData):
    """Main function to orchestrate the entire ingestion and resolution workflow."""
    with driver.session() as session:
        # First, ingest all the raw data from the paper into the graph
        session.write_transaction(ingest_raw_paper_data, jsonData)
        
        # Second, run the entity resolution process for the newly added terms
        session.write_transaction(resolve_new_constructs, jsonData)

def ingest_raw_paper_data(tx, jsonData):
    """Ingests all data from the JSON, creating Term nodes but only provisional CanonicalConstructs."""
    
    # --- 1. Create Paper, Author, and other primary nodes ---
    paper_meta = jsonData.get('paper_metadata', {})
    if not paper_meta.get('doi'):
        paper_meta['doi'] = paper_meta.get('title', 'Unknown Title')

    query_paper = """
    MERGE (p:Paper {doi: $meta.doi})
    ON CREATE SET p = $meta, p.uuid = randomUUID(), p.created_at = datetime()
    ON MATCH SET p += $meta
    """
    tx.run(query_paper, meta=paper_meta)

    for author_name in paper_meta.get('authors', []):
        query_author = """
        MATCH (p:Paper {doi: $doi})
        MERGE (a:Author {full_name: $name})
        ON CREATE SET a.uuid = randomUUID()
        MERGE (p)-[:AUTHORED_BY]->(a)
        """
        tx.run(query_author, doi=paper_meta['doi'], name=author_name)

    # --- 2. Process constructs, creating a provisional CanonicalConstruct for each Term ---
    for construct in jsonData.get('constructs', []):
        # Create a TEMPORARY, one-to-one CanonicalConstruct for this Term.
        # This will be resolved in the next step.
        query_provisional_cc = """
        MATCH (p:Paper {doi: $doi})
        MERGE (t:Term {text: $term_text})
        CREATE (cc:CanonicalConstruct {
            uuid: randomUUID(),
            preferred_name: $term_text,
            status: 'Provisional'
        })
        CREATE (p)-[:USES_TERM]->(t)
        CREATE (t)-[:IS_REPRESENTATION_OF]->(cc)
        """
        tx.run(query_provisional_cc, doi=paper_meta['doi'], term_text=construct['term'])
        
        # Create or re-use Definition (unique per paper × term × text)
        if construct.get('definition'):
            query_definition = """
            MATCH (t:Term {text: $term_text})
            MATCH (p:Paper {doi: $doi})
            MERGE (d:Definition {text: $definition, term_text: $term_text, paper_uid: p.paper_uid})
            ON CREATE SET d.uuid = randomUUID(), d.context_snippet = $snippet
            MERGE (t)-[:HAS_DEFINITION]->(d)
            MERGE (d)-[:DEFINED_IN]->(p)
            """
            tx.run(query_definition, term_text=construct['term'], doi=paper_meta['doi'],
                   definition=construct['definition'], snippet=construct.get('context_snippet'))

    # --- 3. Ingest RelationshipInstances, linking them to the PROVISIONAL CanonicalConstructs ---
    for rel in jsonData.get('relationships', []):
        ri_properties = {k: v for k, v in rel.items() if k not in ['subject_term', 'object_term', 'supporting_theories', 'moderators', 'mediators']}
        ri_properties['uuid'] = 'ri_' + str(uuid.uuid4())

        query_ri = """
        MATCH (p:Paper {doi: $doi})
        MATCH (subject_term:Term {text: $subj_term})-[:IS_REPRESENTATION_OF]->(subject_cc:CanonicalConstruct)
        MATCH (object_term:Term {text: $obj_term})-[:IS_REPRESENTATION_OF]->(object_cc:CanonicalConstruct)
        WHERE subject_cc.status = 'Provisional' AND object_cc.status = 'Provisional'
        
        CREATE (ri:RelationshipInstance)
        SET ri = $props
        
        CREATE (p)-[:ESTABLISHES]->(ri)
        CREATE (ri)-[:HAS_SUBJECT]->(subject_cc)
        CREATE (ri)-[:HAS_OBJECT]->(object_cc)
        
        RETURN ri.uuid AS ri_uuid
        """
        result = tx.run(query_ri, doi=paper_meta['doi'], subj_term=rel['subject_term'], obj_term=rel['object_term'], props=ri_properties)
        ri_uuid = result.single()['ri_uuid'] if result.single() else None
        
        if ri_uuid:
            for theory_name in rel.get('supporting_theories', []):
                tx.run("MATCH (ri:RelationshipInstance {uuid: $ri_uuid}) MERGE (th:Theory {name: $theory_name}) MERGE (ri)-[:APPLIES_THEORY]->(th)", ri_uuid=ri_uuid, theory_name=theory_name)
            for mod_name in rel.get('moderators', []):
                 tx.run("MATCH (ri:RelationshipInstance {uuid: $ri_uuid}) MATCH (mod_term:Term {text: $mod_name})-[:IS_REPRESENTATION_OF]->(mod_cc:CanonicalConstruct) MERGE (ri)-[:HAS_MODERATOR]->(mod_cc)", ri_uuid=ri_uuid, mod_name=mod_name)
            # (Mediator logic is identical)

def resolve_new_constructs_enhanced(tx, jsonData):
    """
    Enhanced two-stage entity resolution workflow that preserves constructs and creates similarity relationships.
    This approach maintains all original information while establishing semantic connections.
    """
    
    SIMILARITY_THRESHOLD = 0.85 # Example threshold
    
    for construct in jsonData.get('constructs', []):
        new_term_text = construct['term']
        new_definition = construct.get('definition')
        
        if not new_definition:
            continue

        # --- Stage 1: Candidate Retrieval via Vector Search ---
        # new_embedding = generate_embedding(new_definition)
        # candidate_results = vector_db.search(new_embedding, top_k=5)
        candidate_results = [] # Placeholder
        
        found_similar = False
        for candidate in candidate_results:
            if candidate.similarity < SIMILARITY_THRESHOLD:
                break
            
            candidate_definition = candidate.metadata['definition']
            
            # --- Stage 2: Candidate Validation via LLM Adjudication ---
            # is_same = llm_adjudicate(new_definition, candidate_definition) # Returns True/False
            is_same = False # Placeholder
            
            if is_same:
                target_cc_name = candidate.metadata['term']
                
                # Instead of deleting, create similarity relationships
                query_create_similarity = """
                MATCH (new_cc:CanonicalConstruct)<-[:IS_REPRESENTATION_OF]-(t:Term {text: $new_term})
                MATCH (target_cc:CanonicalConstruct {preferred_name: $target_name})
                
                // Create bidirectional similarity relationships
                MERGE (new_cc)-[:IS_SIMILAR_TO {
                    similarity_score: $similarity_score,
                    llm_confidence: $llm_confidence,
                    relationship_type: 'synonym',
                    created_at: datetime()
                }]->(target_cc)
                
                MERGE (target_cc)-[:IS_SIMILAR_TO {
                    similarity_score: $similarity_score,
                    llm_confidence: $llm_confidence,
                    relationship_type: 'synonym',
                    created_at: datetime()
                }]->(new_cc)
                
                // Mark the target as canonical (primary)
                SET target_cc.canonical_status = 'primary'
                SET new_cc.canonical_status = 'variant'
                
                // Update status to indicate similarity resolution
                SET new_cc.status = 'Similarity_Resolved'
                SET target_cc.status = 'Verified'
                """
                
                tx.run(query_create_similarity, 
                       new_term=new_term_text, 
                       target_name=target_cc_name,
                       similarity_score=candidate.similarity,
                       llm_confidence=0.95)
                
                found_similar = True
                break
        
        if not found_similar:
            # If no similar construct found, finalize as independent
            query_finalize = """
            MATCH (cc:CanonicalConstruct)<-[:IS_REPRESENTATION_OF]-(t:Term {text: $term_text})
            WHERE cc.status = 'Provisional'
            SET cc.status = 'Verified',
                cc.canonical_status = 'primary'
            """
            tx.run(query_finalize, term_text=new_term_text)
```

#### 3.2.x Theory Extraction and Entity Resolution (v2)

Scope and Inclusion
- Only include theories that explicitly ground core hypotheses/propositions (Hypothesis, Hypothesis_Development, Proposition, Results-Confirming) and can be linked to at least one hypothesis_id or the corresponding section.
- Exclude background theories listed as prior literature if not used to support the core claims.

Extraction and Anchoring
- From relationship instances within the core scope, collect supporting_theories (string names) and require at least one textual anchor (context_snippet or section_header phrase such as “drawing on/based on/grounded in”).
- Aggregate to form a paper-level core_theories list with fields: name, canonical_id (optional), role (primary/secondary by coverage), linked_hypotheses, grounding_phrase, evidence, confidence.

Embeddings & Canonicalization
- Generate embeddings for theory names (and, when available, concise descriptions/evidence snippets) to enable candidate retrieval.
- Perform vector similarity search to find candidate canonical Theory nodes above a threshold (e.g., 0.85 cosine).
- Validate with an LLM adjudicator (binary is_same) before merging. If confirmed, link to the canonical Theory node; otherwise create a new Theory node and mark status/metadata.

Naming Policy
- Preserve full names (no abbreviations in parentheses). Example: “resource-based view” not “RBV (resource-based view)”.
- Maintain a canonical_id naming convention (snake_case) for merged theories (e.g., resource_based_view) and keep original name variants as Term-like aliases if needed for provenance.

Graph Links
- For each RelationshipInstance in the core scope, add APPLIES_THEORY edges to the canonical Theory nodes it explicitly uses. Theories should not be attached to non-core relationships.

Post-Processing Quality Rules
- Enforce exact moderation/nonlinearity strictness at relationship level (no inferred moderators/nonlinear forms without explicit statements).
- Title/journal/DOI normalization as specified (verbatim title, full journal name, DOI without scheme) in ingestion.

---

## 4. Querying the Knowledge Graph Prototype

This enhanced schema allows for even more nuanced queries.

### Example 1: Quantitative Significance Query

**Question:** "Find all empirical studies that use 'Agency Theory' to explain a **statistically significant, negative relationship** between 'CEO Duality' and 'Firm Performance'."

**Cypher Query:**

```cypher
MATCH (theory:Theory {name: 'Agency Theory'})
MATCH (paper:Paper {research_type: 'Empirical'})
MATCH (subject:CanonicalConstruct {preferred_name: 'CEO Duality'})
MATCH (object:CanonicalConstruct {preferred_name: 'Firm Performance'})

MATCH (paper)-[:ESTABLISHES]->(ri:RelationshipInstance)-[:APPLIES_THEORY]->(theory)
MATCH (ri)-[:HAS_SUBJECT]->(subject)
MATCH (ri)-[:HAS_OBJECT]->(object)

WHERE ri.status = 'Empirical_Result'
  AND ri.effect_direction = 'Negative'
  AND ri.statistical_details.p_value < 0.05

RETURN paper.title AS paperTitle, paper.authors AS authors, ri.statistical_details AS stats
```

### Example 2: Qualitative Findings Query

**Question:** "Find all qualitative propositions about how 'Organizational Culture' shapes 'Innovation'."

**Cypher Query:**

```cypher
MATCH (subject:CanonicalConstruct {preferred_name: 'Organizational Culture'})
MATCH (object:CanonicalConstruct {preferred_name: 'Innovation'})

MATCH (paper)-[:ESTABLISHES]->(ri:RelationshipInstance)
WHERE (ri)-[:HAS_SUBJECT]->(subject) 
  AND (ri)-[:HAS_OBJECT]->(object) 
  AND ri.status = 'Empirical_Result'
  AND ri.evidence_type = 'Qualitative'

RETURN paper.title AS paperTitle,
       ri.qualitative_finding AS proposition,
       ri.supporting_quote AS keyEvidence
ORDER BY paper.publication_year DESC
```

### Example 3: Querying for Construct Dimensions and Boundary Conditions

**Question:** "What are the dimensions of 'Organizational Justice'? Also, find all relationships involving 'Procedural Justice' that are explicitly limited to the 'public sector' context."

**Cypher Query:**

```cypher
// Part 1: Find the dimensions
MATCH (parent:CanonicalConstruct {preferred_name: 'Organizational Justice'})-[:HAS_DIMENSION]->(dimension:CanonicalConstruct)
WITH parent, collect(dimension.preferred_name) AS dimensions

// Part 2: Find relationships with boundary conditions
MATCH (pj:CanonicalConstruct {preferred_name: 'Procedural Justice'})
MATCH (paper)-[:ESTABLISHES]->(ri:RelationshipInstance)
WHERE ((ri)-[:HAS_SUBJECT]->(pj) OR (ri)-[:HAS_OBJECT]->(pj))
  AND ri.boundary_conditions CONTAINS 'public sector'

RETURN parent.preferred_name AS parentConstruct,
       dimensions,
       paper.title AS relevantPaper,
       ri.description AS relationshipDescription
```

### Example 4: Querying for Similar Constructs (NEW)

**Question:** "Find all constructs that are semantically similar to 'Organizational Commitment' and show their similarity scores."

**Cypher Query:**

```cypher
MATCH (main:CanonicalConstruct {preferred_name: 'Organizational Commitment'})
MATCH (main)-[r:IS_SIMILAR_TO]->(similar:CanonicalConstruct)
WHERE r.similarity_score > 0.8
RETURN similar.preferred_name AS similarConstruct,
       similar.canonical_status AS status,
       r.similarity_score AS similarity,
       r.llm_confidence AS confidence,
       r.relationship_type AS type
ORDER BY r.similarity_score DESC
```

### Example 5: Finding Canonical vs. Variant Constructs (NEW)

**Question:** "Show all primary (canonical) constructs and their variant forms with similarity relationships."

**Cypher Query:**

```cypher
MATCH (primary:CanonicalConstruct {canonical_status: 'primary'})
OPTIONAL MATCH (primary)-[r:IS_SIMILAR_TO]->(variant:CanonicalConstruct {canonical_status: 'variant'})
RETURN primary.preferred_name AS primaryConstruct,
       collect({
           variant: variant.preferred_name,
           similarity: r.similarity_score,
           confidence: r.llm_confidence
       }) AS variants
ORDER BY primary.preferred_name
```

---

## 5. Prototype Tech Stack Recommendations

To maintain simplicity, the following lightweight tech stack is recommended for the prototype phase:

- **Graph Database**: **Neo4j Desktop**. It provides an excellent local development environment.
- **Vector Database**: A local, self-hosted vector database like **ChromaDB**, **FAISS**, or **Qdrant**. They are easy to set up for a prototype and integrate well with Python.
- **Data Processing Script**: **Python**. Use the official `neo4j` library, a vector DB client (e.g., `chromadb`), and an LLM client (e.g., `openai`).
- **LLM API**: **OpenAI GPT-4o** or **Anthropic Claude 3 Opus** series. Choose the model that performs best at following complex instructions and formatting JSON output.

---

---

## 6. Migration Plan (One-time Data Cleanup)

To align existing data with the revised uniqueness rules (per-paper unique Definition/Measurement), run the following steps once:

1. Add composite uniqueness constraints (recommended where supported):

```cypher
// Optional: node key constraints (Neo4j 4.x+ enterprise)
CREATE CONSTRAINT definition_per_paper IF NOT EXISTS
FOR (d:Definition) REQUIRE (d.text, d.term_text, d.paper_uid) IS NODE KEY;

CREATE CONSTRAINT measurement_per_paper IF NOT EXISTS
FOR (m:Measurement) REQUIRE (m.name, m.construct_term, m.paper_uid) IS NODE KEY;
```

2. Backfill missing keys on existing nodes:

```cypher
// Definitions
MATCH (t:Term)-[:HAS_DEFINITION]->(d:Definition)-[:DEFINED_IN]->(p:Paper)
SET d.term_text = coalesce(d.term_text, t.text), d.paper_uid = coalesce(d.paper_uid, p.paper_uid);

// Measurements
MATCH (cc:CanonicalConstruct)-[:USES_MEASUREMENT]->(m:Measurement)-[:MEASURED_IN]->(p:Paper)
SET m.construct_term = coalesce(m.construct_term, cc.preferred_name), m.paper_uid = coalesce(m.paper_uid, p.paper_uid);
```

3. Deduplicate per composite keys:

```cypher
// Definitions
MATCH (d:Definition)
WITH d.paper_uid AS pid, d.term_text AS term, d.text AS txt, collect(d) AS defs
WHERE pid IS NOT NULL AND term IS NOT NULL AND txt IS NOT NULL AND size(defs) > 1
FOREACH (x IN tail(defs) | DETACH DELETE x);

// Measurements
MATCH (m:Measurement)
WITH m.paper_uid AS pid, m.construct_term AS ct, toLower(m.name) AS n, collect(m) AS mm
WHERE pid IS NOT NULL AND ct IS NOT NULL AND n IS NOT NULL AND size(mm) > 1
FOREACH (x IN tail(mm) | DETACH DELETE x);
```

---

## 7. Frontend Data Contract Update

- Each construct record now includes `best_description` computed server-side from its per-paper definitions; the UI should display it as a summary even when the current paper filter yields no definition.
- Definitions and measurements are returned de-duplicated per `(paper_uid, ...)` and grouped by paper in the panel.

---

## 8. Conclusion and Next Steps

This enhanced technical blueprint provides a robust foundation for a prototype capable of handling the diverse methodologies within social science literature. By implementing the two-stage entity resolution workflow, the resulting knowledge graph will not only store information but will actively work to unify and align concepts, offering a far more coherent and intelligent view of the academic landscape.

This prototype will serve as a solid foundation for more complex future applications.