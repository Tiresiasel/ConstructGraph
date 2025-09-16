from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple
import threading
import time
import hashlib
from pathlib import Path
from flask import send_from_directory

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
from py2neo import Graph

from construct_graph.config import CONFIG
from construct_graph.data.fetchers import fetch_constructs, fetch_relationships
from construct_graph.layout import compute_layouts
from construct_graph.db.neo4j import get_graph as get_graph_client

# Reuse backend ops from build_graph
from build_graph import (
    setup_vector_database,
    update_construct_description,
    _apply_soft_merge,
    rollback_merge,
    create_construct,
    update_relationship_instance,
    soft_delete_relationship_instance,
    restore_relationship_instance,
    rollback_relationship_operation,
    create_relationship_instance,
    process_pdf_concurrent,
)


def create_app() -> Flask:
    # Configure template folder to render HTML at runtime (no host dist file needed)
    template_root = Path(__file__).resolve().parent.parent / 'construct_graph' / 'render' / 'templates'
    static_root = Path(__file__).resolve().parent.parent / 'construct_graph' / 'render' / 'static'
    app = Flask(__name__, template_folder=str(template_root), static_folder=str(static_root), static_url_path='/static')
    CORS(app)

    # Neo4j
    graph: Graph = get_graph_client()

    # Optional vector DB (used for constructs update auto-merge)
    vector_db = setup_vector_database()
    
    # Start polling thread when app is created
    start_polling()

    def _paginate(items: List[Dict[str, Any]], page: int, limit: int) -> Dict[str, Any]:
        total = len(items)
        if limit <= 0:
            return {"total": total, "items": items}
        start = max((page - 1) * limit, 0)
        end = start + limit
        return {"total": total, "items": items[start:end]}

    def _get_pagination_args() -> Tuple[int, int]:
        try:
            page = int(request.args.get('page', '1'))
        except Exception:
            page = 1
        try:
            limit = int(request.args.get('limit', '50'))
        except Exception:
            limit = 50
        return max(page, 1), max(limit, 0)

    # --- Runtime-rendered index page from Jinja template ---
    @app.get('/')
    def index():
        # Compute deterministic layout positions once at render-time so the
        # front-end can toggle between centrality and embedding layouts.
        try:
            constructs = fetch_constructs(graph)
            relationships = fetch_relationships(graph)
            embed_pos, central_pos = compute_layouts(constructs, relationships)
        except Exception:
            # Fall back gracefully if layout computation fails
            embed_pos, central_pos = {}, {}

        # The template fetches live data from /api/*; we only pass layout maps
        # so that layout toggles work immediately without additional API calls.
        return render_template(
            'constructs_network.html.j2',
            constructs=[],
            relationships=[],
            papers=[],
            embed_pos=embed_pos,
            central_pos=central_pos,
        )

    @app.get('/api/health')
    def health():
        return jsonify({"status": "ok"})
    
    @app.post('/api/cleanup/duplicates')
    def cleanup_duplicates():
        """Clean up duplicate papers and constructs"""
        try:
            # Clean up duplicate papers
            paper_duplicates = graph.run("""
                MATCH (p:Paper)
                WITH p.paper_uid as paper_uid, collect(p) as papers
                WHERE size(papers) > 1
                RETURN paper_uid, papers
                ORDER BY paper_uid
            """).data()
            
            cleaned_papers = []
            for dup in paper_duplicates:
                papers = dup['papers']
                # Keep the first one, delete the rest
                keep = papers[0]
                delete_list = papers[1:]
                
                for delete_paper in delete_list:
                    # Delete duplicate paper and its relationships
                    graph.run("""
                        MATCH (p:Paper {paper_uid: $delete_id})
                        OPTIONAL MATCH (p)-[r]-()
                        DELETE r, p
                    """, delete_id=delete_paper['paper_uid'])
                
                cleaned_papers.append({
                    'paper_uid': dup['paper_uid'],
                    'kept': keep['paper_uid'],
                    'deleted': len(delete_list)
                })
            
            return jsonify({
                "message": "Cleanup completed",
                "cleaned_papers": cleaned_papers
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.post('/api/cleanup/reset-ingestion')
    def reset_ingestion():
        """Reset ingestion status to reprocess PDFs"""
        try:
            # Delete all IngestedFile records
            result = graph.run("""
                MATCH (f:IngestedFile)
                DELETE f
                RETURN count(f) as deleted
            """).data()
            
            deleted_count = result[0]['deleted'] if result else 0
            
            return jsonify({
                "message": "Ingestion status reset",
                "deleted_files": deleted_count,
                "note": "PDFs will be reprocessed on next poll cycle"
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.get('/api/test-reload')
    def test_reload():
        return jsonify({"message": "实时监控测试成功！", "timestamp": "2025-08-26 21:45:00"})

    @app.get('/api/constructs')
    def api_constructs():
        q = (request.args.get('q') or '').strip().lower()
        page, limit = _get_pagination_args()
        active_param = request.args.get('active')
        connected_only_param = request.args.get('connected_only')
        data = fetch_constructs(graph)
        # Default view: active only unless explicitly overridden
        if active_param is None or active_param.lower() == 'true':
            data = [d for d in data if d.get('active', True)]
        elif active_param.lower() == 'false':
            pass  # include all (both active and inactive)
        # Filter out isolated constructs by default unless explicitly disabled
        if connected_only_param is None or connected_only_param.lower() == 'true':
            try:
                rels = fetch_relationships(graph)
                connected_names: set[str] = set()
                for r in rels or []:
                    s = (r.get('source_construct') or '').strip()
                    t = (r.get('target_construct') or '').strip()
                    if s:
                        connected_names.add(s.lower())
                    if t:
                        connected_names.add(t.lower())
                    for ri in (r.get('relationship_instances') or []):
                        for m in (ri.get('moderators') or []):
                            if isinstance(m, str) and m.strip():
                                connected_names.add(m.strip().lower())
                        for m in (ri.get('mediators') or []):
                            if isinstance(m, str) and m.strip():
                                connected_names.add(m.strip().lower())
                if connected_names:
                    data = [d for d in data if isinstance(d.get('name'), str) and d['name'].strip().lower() in connected_names]
            except Exception:
                # If anything fails, fall back to unfiltered to avoid 500s
                pass
        # Optional text filter
        if q:
            data = [d for d in data if isinstance(d.get('name'), str) and q in d['name'].lower()]
        return jsonify(_paginate(data, page, limit))

    @app.get('/api/relationships')
    def api_relationships():
        page, limit = _get_pagination_args()
        status = request.args.get('status')
        evidence_type = request.args.get('evidence_type')
        data = fetch_relationships(graph)
        if status:
            data = [d for d in data if (d.get('status') == status)]
        if evidence_type:
            data = [d for d in data if (d.get('evidence_type') == evidence_type)]
        return jsonify(_paginate(data, page, limit))

    @app.get('/api/papers')
    def api_papers():
        # Basic list with optional year filter + pagination
        rows = graph.run(
            """
            MATCH (p:Paper)
            RETURN p.paper_uid as id, p.title as title, p.authors as authors,
                   p.publication_year as year, p.journal as journal,
                   p.research_type as research_type, p.research_context as research_context
            ORDER BY coalesce(p.publication_year, 0) DESC, p.title ASC
            """
        ).data()
        year = request.args.get('year')
        if year:
            try:
                y = int(year)
                rows = [r for r in rows if r.get('year') == y]
            except Exception:
                pass
        page, limit = _get_pagination_args()
        return jsonify(_paginate(rows, page, limit))

    @app.patch('/api/constructs/<name>/description')
    def api_update_construct_description(name: str):
        body: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
        desc = body.get('description')
        if desc is None:
            return jsonify({"error": "description is required"}), 400
        auto_merge = bool(body.get('auto_merge', False))
        sim_th = float(body.get('similarity_threshold', 0.80))
        merge_th = float(body.get('merge_confidence_threshold', 0.95))
        editor = body.get('editor') or 'api'
        reason = body.get('reason')
        changes = update_construct_description(
            graph, name, desc,
            editor=editor, reason=reason,
            auto_merge=auto_merge,
            similarity_threshold=sim_th,
            merge_confidence_threshold=merge_th,
            vector_db=vector_db,
        )
        return jsonify(changes)

    @app.post('/api/constructs/merge')
    def api_merge_constructs():
        body = request.get_json(force=True, silent=True) or {}
        keep = body.get('keep')
        drop = body.get('drop')
        if not keep or not drop:
            return jsonify({"error": "keep and drop are required"}), 400
        editor = body.get('editor') or 'api'
        reason = body.get('reason') or 'api merge'
        similarity = float(body.get('similarity', 1.0))
        confidence = float(body.get('confidence', 1.0))
        op_id = _apply_soft_merge(graph, keep, drop, similarity=similarity, confidence=confidence, initiator=editor, reason=reason)
        return jsonify({"operation_id": op_id})

    @app.post('/api/constructs/rollback-merge')
    def api_rollback_merge():
        body = request.get_json(force=True, silent=True) or {}
        op_id = body.get('operation_id')
        if not op_id:
            return jsonify({"error": "operation_id is required"}), 400
        report = rollback_merge(graph, op_id)
        return jsonify(report)

    @app.post('/api/constructs')
    def api_create_construct():
        body = request.get_json(force=True, silent=True) or {}
        term = body.get('term')
        definition = body.get('definition')
        paper_uid = body.get('paper_uid')
        context_snippet = body.get('context_snippet')
        measurements = body.get('measurements') or []
        auto_merge = bool(body.get('auto_merge', True))
        sim_th = float(body.get('similarity_threshold', 0.80))
        merge_th = float(body.get('merge_confidence_threshold', 0.95))
        res = create_construct(
            graph,
            term=term,
            definition=definition,
            context_snippet=context_snippet,
            paper_uid=paper_uid,
            measurements=measurements,
            vector_db=vector_db,
            auto_merge=auto_merge,
            similarity_threshold=sim_th,
            merge_confidence_threshold=merge_th,
        )
        return jsonify(res)

    @app.delete('/api/constructs/<name>')
    def api_soft_delete_construct(name: str):
        # Soft delete by setting active=false
        tx = graph.begin()
        try:
            res = tx.run(
                """
                MATCH (cc:CanonicalConstruct {preferred_name: $n})
                SET cc.active = false
                RETURN cc.preferred_name as name, cc.active as active
                """,
                n=name.strip().lower(),
            ).data()
            tx.commit()
            if not res:
                return jsonify({"error": "Construct not found"}), 404
            return jsonify(res[0])
        except Exception as e:
            tx.rollback()
            return jsonify({"error": str(e)}), 500

    @app.patch('/api/relationships/<ri_uuid>')
    def api_update_relationship(ri_uuid: str):
        body = request.get_json(force=True, silent=True) or {}
        props = body.get('props') or {}
        role_changes = body.get('role_changes') or {}
        editor = body.get('editor') or 'api'
        reason = body.get('reason')
        auto_create = bool(body.get('auto_create_construct', False))
        changes = update_relationship_instance(
            graph, ri_uuid,
            props=props, role_changes=role_changes,
            editor=editor, reason=reason,
            auto_create_construct=auto_create,
        )
        return jsonify(changes)

    @app.post('/api/relationships')
    def api_create_relationship():
        body: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
        subject = body.get('subject')
        object_ = body.get('object')
        status = body.get('status')
        evidence_type = body.get('evidence_type')
        effect_direction = body.get('effect_direction')
        non_linear_type = body.get('non_linear_type')
        is_validated_causality = body.get('is_validated_causality')
        is_meta_analysis = body.get('is_meta_analysis')
        theories = body.get('theories') or []
        moderators = body.get('moderators') or []
        mediators = body.get('mediators') or []
        controls = body.get('controls') or []
        context_snippet = body.get('context_snippet')
        description = body.get('description')
        paper_uid = body.get('paper_uid')
        auto_create_construct = bool(body.get('auto_create_construct', True))
        res = create_relationship_instance(
            graph,
            subject=subject,
            object_=object_,
            status=status,
            evidence_type=evidence_type,
            effect_direction=effect_direction,
            non_linear_type=non_linear_type,
            is_validated_causality=is_validated_causality,
            is_meta_analysis=is_meta_analysis,
            theories=theories,
            moderators=moderators,
            mediators=mediators,
            controls=controls,
            context_snippet=context_snippet,
            description=description,
            paper_uid=paper_uid,
            auto_create_construct=auto_create_construct,
        )
        return jsonify(res)

    @app.post('/api/relationships/<ri_uuid>/soft-delete')
    def api_soft_delete_relationship(ri_uuid: str):
        body = request.get_json(force=True, silent=True) or {}
        editor = body.get('editor') or 'api'
        reason = body.get('reason')
        res = soft_delete_relationship_instance(graph, ri_uuid, editor=editor, reason=reason)
        return jsonify(res)

    @app.post('/api/relationships/<ri_uuid>/restore')
    def api_restore_relationship(ri_uuid: str):
        body = request.get_json(force=True, silent=True) or {}
        editor = body.get('editor') or 'api'
        reason = body.get('reason')
        res = restore_relationship_instance(graph, ri_uuid, editor=editor, reason=reason)
        return jsonify(res)

    @app.post('/api/relationships/rollback-operation')
    def api_rollback_relationship_operation():
        body = request.get_json(force=True, silent=True) or {}
        op_id = body.get('operation_id')
        if not op_id:
            return jsonify({"error": "operation_id is required"}), 400
        res = rollback_relationship_operation(graph, op_id)
        return jsonify(res)

    @app.delete('/api/relationships/<ri_uuid>')
    def api_delete_relationship_soft(ri_uuid: str):
        # Convenience soft-delete using DELETE method
        res = soft_delete_relationship_instance(graph, ri_uuid, editor='api', reason='delete endpoint')
        return jsonify(res)

    # --- Dimensions ---
    @app.post('/api/constructs/<name>/dimensions')
    def api_add_dimension(name: str):
        body: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
        child = (body.get('child') or '').strip().lower()
        if not child:
            return jsonify({"error": "child is required"}), 400
        tx = graph.begin()
        try:
            tx.run(
                """
                MERGE (parent:CanonicalConstruct {preferred_name: $parent})
                ON CREATE SET parent.uuid=coalesce(parent.uuid, randomUUID()), parent.status=coalesce(parent.status,'Provisional'), parent.active=true
                MERGE (child:CanonicalConstruct {preferred_name: $child})
                ON CREATE SET child.uuid=coalesce(child.uuid, randomUUID()), child.status=coalesce(child.status,'Provisional'), child.active=true
                MERGE (parent)-[:HAS_DIMENSION]->(child)
                """,
                parent=name.strip().lower(), child=child,
            )
            tx.commit()
            return jsonify({"parent": name.strip().lower(), "child": child})
        except Exception as e:
            tx.rollback()
            return jsonify({"error": str(e)}), 500

    @app.delete('/api/constructs/<name>/dimensions/<child>')
    def api_remove_dimension(name: str, child: str):
        tx = graph.begin()
        try:
            res = tx.run(
                """
                MATCH (parent:CanonicalConstruct {preferred_name: $parent})-[r:HAS_DIMENSION]->(child:CanonicalConstruct {preferred_name: $child})
                DELETE r
                RETURN 1 as removed
                """,
                parent=name.strip().lower(), child=child.strip().lower(),
            ).data()
            tx.commit()
            return jsonify({"removed": bool(res)})
        except Exception as e:
            tx.rollback()
            return jsonify({"error": str(e)}), 500

    # --- Similarity links ---
    @app.post('/api/constructs/<a>/similar-to/<b>')
    def api_add_similarity(a: str, b: str):
        body: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
        rel_type = body.get('relationship_type') or 'synonym'
        sim = float(body.get('similarity_score') or 0.0)
        conf = float(body.get('llm_confidence') or 0.0)
        tx = graph.begin()
        try:
            tx.run(
                """
                MERGE (a:CanonicalConstruct {preferred_name: $a})
                ON CREATE SET a.uuid=coalesce(a.uuid, randomUUID()), a.status=coalesce(a.status,'Provisional'), a.active=true
                MERGE (b:CanonicalConstruct {preferred_name: $b})
                ON CREATE SET b.uuid=coalesce(b.uuid, randomUUID()), b.status=coalesce(b.status,'Provisional'), b.active=true
                MERGE (a)-[r1:IS_SIMILAR_TO]->(b)
                ON CREATE SET r1.relationship_type=$rt, r1.similarity_score=$sim, r1.llm_confidence=$conf
                SET r1.relationship_type=$rt, r1.similarity_score=$sim, r1.llm_confidence=$conf
                MERGE (b)-[r2:IS_SIMILAR_TO]->(a)
                ON CREATE SET r2.relationship_type=$rt, r2.similarity_score=$sim, r2.llm_confidence=$conf
                SET r2.relationship_type=$rt, r2.similarity_score=$sim, r2.llm_confidence=$conf
                """,
                a=a.strip().lower(), b=b.strip().lower(), rt=rel_type, sim=sim, conf=conf,
            )
            tx.commit()
            return jsonify({"a": a.strip().lower(), "b": b.strip().lower(), "relationship_type": rel_type, "similarity_score": sim, "llm_confidence": conf})
        except Exception as e:
            tx.rollback()
            return jsonify({"error": str(e)}), 500

    @app.delete('/api/constructs/<a>/similar-to/<b>')
    def api_remove_similarity(a: str, b: str):
        tx = graph.begin()
        try:
            tx.run(
                """
                MATCH (a:CanonicalConstruct {preferred_name: $a})-[r1:IS_SIMILAR_TO]->(b:CanonicalConstruct {preferred_name: $b})
                DELETE r1
                WITH a,b
                MATCH (b)-[r2:IS_SIMILAR_TO]->(a)
                DELETE r2
                """,
                a=a.strip().lower(), b=b.strip().lower(),
            )
            tx.commit()
            return jsonify({"removed": True})
        except Exception as e:
            tx.rollback()
            return jsonify({"error": str(e)}), 500

    # --- Measurements ---
    @app.post('/api/measurements')
    def api_create_measurement():
        body: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
        name = body.get('name')
        construct_term = (body.get('construct_term') or '').strip().lower()
        paper_uid = body.get('paper_uid')
        if not name or not construct_term:
            return jsonify({"error": "name and construct_term are required"}), 400
        tx = graph.begin()
        try:
            res = tx.run(
                """
                MERGE (cc:CanonicalConstruct {preferred_name: $ct})
                ON CREATE SET cc.uuid=coalesce(cc.uuid, randomUUID()), cc.status=coalesce(cc.status,'Provisional'), cc.active=true
                OPTIONAL MATCH (p:Paper {paper_uid: $paper})
                MERGE (m:Measurement {name: $name, construct_term: $ct, paper_uid: $paper})
                ON CREATE SET m.uuid=coalesce(m.uuid, randomUUID()), m.active=true
                SET m.description=$description,
                    m.instrument=$instrument,
                    m.scale_items=$scale_items,
                    m.scoring_procedure=$scoring_procedure,
                    m.formula=$formula,
                    m.reliability=$reliability,
                    m.validity=$validity,
                    m.context_adaptations=$context_adaptations
                MERGE (cc)-[:USES_MEASUREMENT]->(m)
                FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END | MERGE (m)-[:MEASURED_IN]->(p))
                RETURN m.uuid as uuid
                """,
                name=name,
                ct=construct_term,
                paper=paper_uid,
                description=body.get('description'),
                instrument=body.get('instrument'),
                scale_items=body.get('scale_items'),
                scoring_procedure=body.get('scoring_procedure'),
                formula=body.get('formula'),
                reliability=body.get('reliability'),
                validity=body.get('validity'),
                context_adaptations=body.get('context_adaptations'),
            ).data()
            tx.commit()
            return jsonify({"uuid": res[0]["uuid"] if res else None})
        except Exception as e:
            tx.rollback()
            return jsonify({"error": str(e)}), 500

    @app.patch('/api/measurements/<uuid>')
    def api_patch_measurement(uuid: str):
        body: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
        fields = {k: v for k, v in body.items() if k in {
            'name','description','instrument','scale_items','scoring_procedure','formula','reliability','validity','context_adaptations'
        }}
        if not fields:
            return jsonify({"error": "no updatable fields provided"}), 400
        sets = ", ".join([f"m.{k} = ${k}" for k in fields.keys()])
        tx = graph.begin()
        try:
            res = tx.run(
                f"""
                MATCH (m:Measurement {{uuid: $uuid}})
                SET {sets}
                RETURN m.uuid as uuid
                """,
                uuid=uuid, **fields
            ).data()
            tx.commit()
            if not res:
                return jsonify({"error": "Measurement not found"}), 404
            return jsonify({"uuid": uuid})
        except Exception as e:
            tx.rollback()
            return jsonify({"error": str(e)}), 500

    @app.delete('/api/measurements/<uuid>')
    def api_delete_measurement(uuid: str):
        tx = graph.begin()
        try:
            res = tx.run(
                """
                MATCH (m:Measurement {uuid: $uuid})
                SET m.active = false
                RETURN m.uuid as uuid, m.active as active
                """,
                uuid=uuid,
            ).data()
            tx.commit()
            if not res:
                return jsonify({"error": "Measurement not found"}), 404
            return jsonify(res[0])
        except Exception as e:
            tx.rollback()
            return jsonify({"error": str(e)}), 500

    # --- Definitions ---
    @app.post('/api/definitions')
    def api_create_definition():
        body: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
        text = body.get('text')
        term_text = (body.get('term_text') or '').strip().lower()
        paper_uid = body.get('paper_uid')
        if not term_text or not paper_uid:
            return jsonify({"error": "term_text and paper_uid are required"}), 400
        tx = graph.begin()
        try:
            res = tx.run(
                """
                MERGE (t:Term {text: $term})
                ON CREATE SET t.uuid=coalesce(t.uuid, randomUUID())
                MERGE (p:Paper {paper_uid: $paper})
                MERGE (d:Definition {text: coalesce($text,''), term_text: $term, paper_uid: $paper})
                ON CREATE SET d.uuid=coalesce(d.uuid, randomUUID()), d.active=true
                SET d.text=$text, d.context_snippet=$context_snippet
                MERGE (t)-[:HAS_DEFINITION]->(d)
                MERGE (d)-[:DEFINED_IN]->(p)
                RETURN d.uuid as uuid
                """,
                term=term_text, paper=paper_uid, text=text, context_snippet=body.get('context_snippet'),
            ).data()
            tx.commit()
            return jsonify({"uuid": res[0]["uuid"] if res else None})
        except Exception as e:
            tx.rollback()
            return jsonify({"error": str(e)}), 500

    @app.patch('/api/definitions/<uuid>')
    def api_patch_definition(uuid: str):
        body: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
        fields = {k: v for k, v in body.items() if k in {'text','context_snippet'}}
        if not fields:
            return jsonify({"error": "no updatable fields provided"}), 400
        sets = ", ".join([f"d.{k} = ${k}" for k in fields.keys()])
        tx = graph.begin()
        try:
            res = tx.run(
                f"""
                MATCH (d:Definition {{uuid: $uuid}})
                SET {sets}
                RETURN d.uuid as uuid
                """,
                uuid=uuid, **fields
            ).data()
            tx.commit()
            if not res:
                return jsonify({"error": "Definition not found"}), 404
            return jsonify({"uuid": uuid})
        except Exception as e:
            tx.rollback()
            return jsonify({"error": str(e)}), 500

    @app.delete('/api/definitions/<uuid>')
    def api_delete_definition(uuid: str):
        tx = graph.begin()
        try:
            res = tx.run(
                """
                MATCH (d:Definition {uuid: $uuid})
                SET d.active = false
                RETURN d.uuid as uuid, d.active as active
                """,
                uuid=uuid,
            ).data()
            tx.commit()
            if not res:
                return jsonify({"error": "Definition not found"}), 404
            return jsonify(res[0])
        except Exception as e:
            tx.rollback()
            return jsonify({"error": str(e)}), 500

    # --- Papers/Authors/Theories ---
    @app.patch('/api/papers/<paper_uid>')
    def api_patch_paper(paper_uid: str):
        body: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
        allowed = {'title','authors','publication_year','journal','research_type','research_context'}
        fields = {k: v for k, v in body.items() if k in allowed}
        if not fields:
            return jsonify({"error": "no updatable fields provided"}), 400
        sets = ", ".join([f"p.{k} = ${k}" for k in fields.keys()])
        tx = graph.begin()
        try:
            res = tx.run(
                f"""
                MATCH (p:Paper {{paper_uid: $id}})
                SET {sets}
                RETURN p.paper_uid as id
                """,
                id=paper_uid, **fields
            ).data()
            tx.commit()
            if not res:
                return jsonify({"error": "Paper not found"}), 404
            return jsonify({"paper_uid": paper_uid})
        except Exception as e:
            tx.rollback()
            return jsonify({"error": str(e)}), 500

    @app.patch('/api/authors/<path:full_name>')
    def api_patch_author(full_name: str):
        body: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
        new_name = body.get('full_name')
        if not new_name:
            return jsonify({"error": "full_name is required"}), 400
        tx = graph.begin()
        try:
            res = tx.run(
                """
                MATCH (a:Author {full_name: $old})
                SET a.full_name = $new
                RETURN a.full_name as full_name
                """,
                old=full_name, new=new_name,
            ).data()
            tx.commit()
            if not res:
                return jsonify({"error": "Author not found"}), 404
            return jsonify(res[0])
        except Exception as e:
            tx.rollback()
            return jsonify({"error": str(e)}), 500

    @app.patch('/api/theories/<canonical_id>')
    def api_patch_theory(canonical_id: str):
        body: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
        name = body.get('name')
        if not name:
            return jsonify({"error": "name is required"}), 400
        tx = graph.begin()
        try:
            res = tx.run(
                """
                MATCH (t:Theory {canonical_id: $id})
                SET t.name = $name
                RETURN t.canonical_id as canonical_id, t.name as name
                """,
                id=canonical_id, name=name,
            ).data()
            tx.commit()
            if not res:
                return jsonify({"error": "Theory not found"}), 404
            return jsonify(res[0])
        except Exception as e:
            tx.rollback()
            return jsonify({"error": str(e)}), 500

    # --- Operations audit ---
    @app.get('/api/operations')
    def api_list_operations():
        page, limit = _get_pagination_args()
        typ = (request.args.get('type') or '').strip()
        data: List[Dict[str, Any]] = []
        if not typ or typ == 'merge':
            data += graph.run(
                """
                MATCH (op:MergeOperation)
                OPTIONAL MATCH (op)-[:KEEP]->(keep:CanonicalConstruct)
                OPTIONAL MATCH (op)-[:DROP]->(drop:CanonicalConstruct)
                RETURN 'merge' as type, op.id as id, op.created_at as created_at, op.initiator as initiator,
                       op.reason as reason, op.similarity_score as similarity_score, op.llm_confidence as llm_confidence,
                       op.status as status, keep.preferred_name as keep, drop.preferred_name as drop
                ORDER BY op.created_at DESC
                """
            ).data()
        if not typ or typ == 'relationship':
            data += graph.run(
                """
                MATCH (op:RelationshipOperation)
                OPTIONAL MATCH (op)-[:TARGET]->(ri:RelationshipInstance)
                RETURN 'relationship' as type, op.id as id, op.created_at as created_at, op.editor as initiator,
                       op.reason as reason, op.status as status, op.type as op_type, ri.uuid as ri_uuid
                ORDER BY op.created_at DESC
                """
            ).data()
        # Sort mixed list by created_at desc if both types requested
        data.sort(key=lambda x: x.get('created_at') or '', reverse=True)
        return jsonify(_paginate(data, page, limit))

    return app


# --- Background poller for /app/data/input (bind-mounted) ---
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _poll_input_dir():
    poll_enabled = (os.getenv('POLL_ENABLED', 'true').lower() in ('1','true','yes'))
    if not poll_enabled:
        return
    interval = float(os.getenv('POLL_INTERVAL', '5'))
    input_dir = Path(os.getenv('INPUT_DIR', CONFIG.input_dir)).resolve()
    
    print(f"Starting input directory poller for: {input_dir}")
    print(f"Poll interval: {interval} seconds")
    
    # Get fresh connections for each poll cycle
    graph = get_graph_client()
    vector_db = setup_vector_database()
    
    while True:
        try:
            if not input_dir.exists():
                print(f"Input directory does not exist: {input_dir}")
                time.sleep(interval)
                continue
            # Scan PDFs
            pdfs = list(input_dir.rglob('*.pdf'))
            print(f"Found {len(pdfs)} PDF files in {input_dir}")

            # Before each cycle, mark stale in-progress files as failed so they can be retried
            try:
                stale_minutes = int(os.getenv('POLL_STALE_MINUTES', '10'))
                updated = graph.run(
                    """
                    MATCH (f:IngestedFile)
                    WHERE f.status = 'in_progress' AND f.started_at < datetime() - duration({minutes: $m})
                    SET f.status = 'failed', f.failed_at = datetime(),
                        f.last_error = coalesce(f.last_error, 'stale in_progress marked as failed')
                    RETURN count(f) AS updated
                    """,
                    m=stale_minutes,
                ).evaluate()
                if updated:
                    print(f"Marked {updated} stale in-progress files as failed (>{stale_minutes}m)")
            except Exception as e:
                print(f"Failed to mark stale in-progress files: {e}")
            # Collect PDFs that need processing
            pdfs_to_process = []
            for pdf in pdfs:
                try:
                    sha = _sha256_file(pdf)
                    print(f"Checking PDF: {pdf.name} (SHA: {sha[:8]}...)")
                    # decide by status of ingestion record
                    rec = graph.run(
                        """
                        MATCH (f:IngestedFile {sha256: $sha})
                        RETURN f.status AS status
                        """,
                        sha=sha,
                    ).data()
                    if rec:
                        status = (rec[0].get('status') or '').lower()
                        if status == 'succeeded':
                            print(f"PDF already ingested: {pdf.name}")
                            continue
                        if status == 'in_progress':
                            print(f"PDF currently in progress, skipping this cycle: {pdf.name}")
                            continue
                        if status == 'failed':
                            print(f"Retrying previously failed PDF: {pdf.name}")
                    
                    # Add to processing queue
                    pdfs_to_process.append((pdf, sha))
                    print(f"Added to processing queue: {pdf.name}")
                except Exception as e:
                    print(f"Failed to check PDF {pdf}: {e}")
            
            # Process PDFs concurrently (max 4 at a time)
            if pdfs_to_process:
                max_concurrent = int(os.getenv('MAX_CONCURRENT_PDFS', '4'))
                print(f"Starting concurrent processing of {len(pdfs_to_process)} PDFs (max {max_concurrent} at a time)")
                
                # Process in batches of max_concurrent
                for i in range(0, len(pdfs_to_process), max_concurrent):
                    batch = pdfs_to_process[i:i + max_concurrent]
                    print(f"Processing batch {i//max_concurrent + 1}: {[p[0].name for p in batch]}")
                    
                    # Start all PDFs in this batch concurrently
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                        # Submit all PDFs in the batch
                        future_to_pdf = {}
                        for pdf, sha in batch:
                            # Mark as in_progress before processing
                            try:
                                graph.run(
                                    """
                                    MERGE (f:IngestedFile {sha256: $sha})
                                    SET f.filename = $fn,
                                        f.status = 'in_progress',
                                        f.started_at = datetime(),
                                        f.last_error = NULL
                                    """,
                                    sha=sha, fn=str(pdf.name)
                                )
                                print(f"Starting concurrent processing for {pdf.name}...")
                                
                                # Submit to thread pool
                                future = executor.submit(process_pdf_concurrent, pdf, vector_db, graph)
                                future_to_pdf[future] = (pdf, sha)
                            except Exception as e:
                                print(f"Failed to start processing for {pdf.name}: {e}")
                                # Mark as failed
                                try:
                                    graph.run(
                                        """
                                        MERGE (f:IngestedFile {sha256: $sha})
                                        SET f.filename = $fn,
                                            f.status = 'failed',
                                            f.failed_at = datetime(),
                                            f.last_error = $err
                                        """,
                                        sha=sha, fn=str(pdf.name), err=str(e)
                                    )
                                except Exception as e2:
                                    print(f"Failed to mark {pdf.name} as failed: {e2}")
                        
                        # Wait for all PDFs in this batch to complete
                        for future in concurrent.futures.as_completed(future_to_pdf):
                            pdf, sha = future_to_pdf[future]
                            try:
                                ok = future.result()
                                print(f"process_pdf_concurrent returned for {pdf.name}: {ok}")
                                
                                if ok:
                                    print(f"Recording successful processing for {pdf.name}...")
                                    graph.run(
                                        """
                                        MATCH (f:IngestedFile {sha256: $sha})
                                        SET f.status = 'succeeded',
                                            f.processed_at = datetime(),
                                            f.last_error = NULL,
                                            f.failed_at = NULL
                                        """,
                                        sha=sha,
                                    )
                                    print(f"Successfully processed and recorded: {pdf.name}")
                                else:
                                    print(f"Failed to process: {pdf.name}")
                                    graph.run(
                                        """
                                        MATCH (f:IngestedFile {sha256: $sha})
                                        SET f.status = 'failed', f.failed_at = datetime(),
                                            f.last_error = coalesce(f.last_error, 'process_pdf_concurrent returned False')
                                        """,
                                        sha=sha,
                                    )
                            except Exception as e:
                                print(f"Exception during PDF processing for {pdf.name}: {e}")
                                import traceback
                                traceback.print_exc()
                                try:
                                    graph.run(
                                        """
                                        MERGE (f:IngestedFile {sha256: $sha})
                                        SET f.filename = $fn,
                                            f.status = 'failed',
                                            f.failed_at = datetime(),
                                            f.last_error = $err
                                        """,
                                        sha=sha, fn=str(pdf.name), err=str(e)
                                    )
                                except Exception as e2:
                                    print(f"Failed updating ingestion status for {pdf.name}: {e2}")
                    
                    print(f"Completed batch {i//max_concurrent + 1}")
            else:
                print("No PDFs need processing in this cycle")
            time.sleep(interval)
        except Exception as e:
            print(f"Poller loop error: {e}")
            time.sleep(interval)


def start_polling():
    """Start the background polling thread if not already running"""
    # Check if poller thread is already running
    for thread in threading.enumerate():
        if thread.name == 'input_poller' and thread.is_alive():
            return  # Already running
    
    # Start poller thread
    t = threading.Thread(target=_poll_input_dir, name='input_poller', daemon=True)
    t.start()
    print(f"Started input polling thread: {t.name} (PID: {t.ident})")


def create_app_with_polling():
    """Create Flask app and start polling thread"""
    app = create_app()
    with app.app_context():
        start_polling()
    return app


if __name__ == '__main__':
    app = create_app_with_polling()  # Create app and start polling
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', '5050'))
    app.run(host=host, port=port, debug=True)


