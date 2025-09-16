"""High-level data fetch APIs used by visualizations and tools.

These functions encapsulate Cypher queries and result normalization so that
front-end scripts can be thin.
"""

from __future__ import annotations

from typing import Any, Dict, List
from py2neo import Graph


def serialize_neo4j_data(obj: Any) -> Any:
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: serialize_neo4j_data(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_neo4j_data(x) for x in obj]
    if hasattr(obj, '__dict__'):
        return serialize_neo4j_data(obj.__dict__)
    return obj


def fetch_constructs(graph: Graph) -> List[Dict[str, Any]]:
    query = """
        MATCH (cc:CanonicalConstruct)
        OPTIONAL MATCH (cc)<-[:IS_REPRESENTATION_OF]-(t:Term)-[:HAS_DEFINITION]->(def:Definition)-[:DEFINED_IN]->(paper:Paper)
        OPTIONAL MATCH (cc)-[:USES_MEASUREMENT]->(meas:Measurement)-[:MEASURED_IN]->(meas_paper:Paper)
        OPTIONAL MATCH (cc)-[:HAS_DIMENSION]->(dim:CanonicalConstruct)
        OPTIONAL MATCH (parent:CanonicalConstruct)-[:HAS_DIMENSION]->(cc)
        OPTIONAL MATCH (cc)-[:IS_SIMILAR_TO]->(similar:CanonicalConstruct)
        OPTIONAL MATCH (cc)<-[:IS_SIMILAR_TO]-(similar_to:CanonicalConstruct)
        RETURN cc.preferred_name as name,
               cc.description as description,
               cc.status as status,
               cc.canonical_status as canonical_status,
               coalesce(cc.active, true) as active,
               collect(DISTINCT {
                   definition: def.text,
                   context_snippet: def.context_snippet,
                   paper_uid: paper.paper_uid,
                   paper_title: paper.title,
                   paper_authors: paper.authors,
                   paper_year: paper.publication_year
               }) as definitions,
               [m in collect(DISTINCT {
                   name: meas.name,
                   description: meas.description,
                   paper_uid: meas_paper.paper_uid,
                   paper_title: meas_paper.title,
                   paper_authors: meas_paper.authors,
                   paper_year: meas_paper.publication_year,
                   active: meas.active
               }) WHERE coalesce(m.active, true)] as measurements,
               collect(DISTINCT dim.preferred_name) as dimensions,
               collect(DISTINCT parent.preferred_name) as parent_constructs,
               collect(DISTINCT {
                   name: similar.preferred_name,
                   status: similar.canonical_status
               }) as similar_constructs,
               collect(DISTINCT {
                   name: similar_to.preferred_name,
                   status: similar_to.canonical_status
               }) as similar_to_constructs,
               [pid IN (collect(DISTINCT paper.paper_uid) + collect(DISTINCT meas_paper.paper_uid)) WHERE pid IS NOT NULL | pid] as paper_ids
        ORDER BY cc.preferred_name
    """
    return graph.run(query).data()


def fetch_relationships(graph: Graph) -> List[Dict[str, Any]]:
    query = """
        MATCH (source:CanonicalConstruct)<-[:HAS_SUBJECT]-(ri:RelationshipInstance)-[:HAS_OBJECT]->(target:CanonicalConstruct)
        MATCH (p:Paper)-[:ESTABLISHES]->(ri)
        OPTIONAL MATCH (ri)-[:APPLIES_THEORY]->(theory:Theory)
        OPTIONAL MATCH (ri)-[:HAS_MODERATOR]->(moderator:CanonicalConstruct)
        OPTIONAL MATCH (ri)-[:HAS_MEDIATOR]->(mediator:CanonicalConstruct)
        WITH source, target, ri, p,
             collect(DISTINCT theory.name) as theories,
             collect(DISTINCT moderator.preferred_name) as moderators,
             collect(DISTINCT mediator.preferred_name) as mediators
        WITH source, target,
             collect(DISTINCT {
                 uuid: ri.uuid,
                 description: ri.description,
                 context_snippet: ri.context_snippet,
                 status: ri.status,
                 evidence_type: ri.evidence_type,
                 effect_direction: ri.effect_direction,
                 non_linear_type: ri.non_linear_type,
                 is_validated_causality: ri.is_validated_causality,
                 is_meta_analysis: ri.is_meta_analysis,
                 statistical_details: ri.statistical_details,
                 qualitative_finding: ri.qualitative_finding,
                 supporting_quote: ri.supporting_quote,
                 boundary_conditions: ri.boundary_conditions,
                 replication_outcome: ri.replication_outcome,
                 theories: theories,
                 moderators: moderators,
                 mediators: mediators,
                 active: coalesce(ri.active, true),
                 paper_uid: p.paper_uid,
                 paper_title: p.title,
                 paper_authors: p.authors,
                 paper_year: p.publication_year
             }) AS relationship_instances
        WHERE source.preferred_name <> target.preferred_name
        RETURN DISTINCT source.preferred_name as source_construct,
               target.preferred_name as target_construct,
               coalesce(source.active, true) as source_active,
               coalesce(target.active, true) as target_active,
               relationship_instances[0].status as status,
               relationship_instances[0].evidence_type as evidence_type,
               relationship_instances[0].effect_direction as effect_direction,
               relationship_instances[0].is_validated_causality as is_validated_causality,
               relationship_instances,
               [pid IN collect(DISTINCT [ri IN relationship_instances WHERE ri.paper_uid IS NOT NULL | ri.paper_uid]) WHERE pid IS NOT NULL | pid] as paper_ids
        ORDER BY source.preferred_name, target.preferred_name
    """
    return graph.run(query).data()


def fetch_papers(graph: Graph) -> List[Dict[str, Any]]:
    """Return minimal paper records used by the visualization sidebar/filters.

    Fields:
    - id: paper_uid
    - title: paper title
    - authors: list of authors
    - year: publication_year
    """
    query = """
        MATCH (p:Paper)
        RETURN p.paper_uid AS id,
               p.title AS title,
               p.authors AS authors,
               p.publication_year AS year
        ORDER BY coalesce(p.publication_year, 9999) ASC, p.title
    """
    return graph.run(query).data()

