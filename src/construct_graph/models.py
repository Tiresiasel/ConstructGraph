"""Lightweight typed data models for clarity in function signatures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Paper:
    id: str
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    research_type: Optional[str] = None
    research_context: Optional[str] = None
    is_replication_study: Optional[bool] = None


@dataclass
class RelationshipInstance:
    uuid: str
    status: Optional[str] = None
    evidence_type: Optional[str] = None
    effect_direction: Optional[str] = None
    non_linear_type: Optional[str] = None
    is_validated_causality: Optional[bool] = None
    is_meta_analysis: Optional[bool] = None
    statistical_details: Optional[str] = None
    qualitative_finding: Optional[str] = None
    supporting_quote: Optional[str] = None
    boundary_conditions: Optional[str] = None
    replication_outcome: Optional[str] = None
    theories: List[str] = field(default_factory=list)
    moderators: List[str] = field(default_factory=list)
    mediators: List[str] = field(default_factory=list)
    paper_uid: Optional[str] = None
    paper_title: Optional[str] = None
    paper_authors: Optional[str] = None
    paper_year: Optional[int] = None


@dataclass
class Relationship:
    source_construct: str
    target_construct: str
    status: Optional[str] = None
    evidence_type: Optional[str] = None
    effect_direction: Optional[str] = None
    is_validated_causality: Optional[bool] = None
    relationship_instances: List[RelationshipInstance] = field(default_factory=list)
    paper_ids: List[str] = field(default_factory=list)


@dataclass
class Construct:
    name: str
    description: Optional[str] = None
    status: Optional[str] = None
    canonical_status: Optional[str] = None
    definitions: list = field(default_factory=list)
    measurements: list = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)
    parent_constructs: List[str] = field(default_factory=list)
    similar_constructs: list = field(default_factory=list)
    similar_to_constructs: list = field(default_factory=list)
    paper_ids: List[str] = field(default_factory=list)
    best_description: Optional[str] = None


