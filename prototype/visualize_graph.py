# visualize_graph_constructs.py
import json
from py2neo import Graph
from datetime import datetime
import numpy as np
import os
import qdrant_client

# --- 1. CONFIGURATION ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # <-- IMPORTANT: Change to your Neo4j password
OUTPUT_HTML_FILE = "constructs_network.html"

# Qdrant vector database configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "construct_definitions"

# --- 2. DATA SERIALIZATION ---
def serialize_neo4j_data(obj):
    """
    Recursively serialize Neo4j data to make it JSON-compatible.
    Converts DateTime objects to ISO format strings.
    """
    if hasattr(obj, 'isoformat'):  # Handle datetime objects
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_neo4j_data(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_neo4j_data(item) for item in obj]
    elif hasattr(obj, '__dict__'):  # Handle other Neo4j objects
        return serialize_neo4j_data(obj.__dict__)
    else:
        return obj

# --- 3. DATA FETCHING ---
def fetch_constructs_network_data(graph):
    """Fetch constructs network data using the blueprint-compliant schema"""
    try:
        # Query for CanonicalConstructs with their definitions and measurements (Enhanced Blueprint schema with similarity)
        constructs_query = """
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
               collect(DISTINCT {
                   definition: def.text,
                   context_snippet: def.context_snippet,
                   paper_uid: paper.paper_uid,
                   paper_title: paper.title,
                   paper_authors: paper.authors,
                   paper_year: paper.publication_year
               }) as definitions,
               collect(DISTINCT {
                   name: meas.name,
                   description: meas.description,
                   paper_uid: meas_paper.paper_uid,
                   paper_title: meas_paper.title,
                   paper_authors: meas_paper.authors,
                   paper_year: meas_paper.publication_year
               }) as measurements,
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
               ([pid IN collect(DISTINCT paper.paper_uid) WHERE pid IS NOT NULL | pid] +
                [pid2 IN collect(DISTINCT meas_paper.paper_uid) WHERE pid2 IS NOT NULL | pid2]) as paper_ids
        ORDER BY cc.preferred_name
        """
        
        # Query for RelationshipInstances (Enhanced Blueprint schema)
        relationships_query = """
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
                 paper_uid: p.paper_uid,
                 paper_title: p.title,
                 paper_authors: p.authors,
                 paper_year: p.publication_year
             }) AS relationship_instances
        WHERE source.preferred_name <> target.preferred_name
        RETURN DISTINCT source.preferred_name as source_construct,
               target.preferred_name as target_construct,
               relationship_instances[0].status as status,
               relationship_instances[0].evidence_type as evidence_type,
               relationship_instances[0].effect_direction as effect_direction,
               relationship_instances[0].is_validated_causality as is_validated_causality,
               relationship_instances,
               [ri IN relationship_instances WHERE ri.paper_uid IS NOT NULL | ri.paper_uid] as paper_ids
        ORDER BY source.preferred_name, target.preferred_name
        """

        # Papers list for left panel (Enhanced Blueprint schema)
        papers_query = """
        MATCH (p:Paper)
        RETURN DISTINCT p.paper_uid as id, 
               p.title as title, 
               p.authors as authors,
               p.publication_year as year,
               p.journal as journal,
               p.research_type as research_type,
               p.research_context as research_context,
               p.is_replication_study as is_replication_study
        ORDER BY p.publication_year DESC, p.title ASC
        """
        
        print("Fetching constructs...")
        constructs_result = graph.run(constructs_query).data()
        print(f"Found {len(constructs_result)} constructs")
        
        # Merge constructs with the same name to avoid duplicates
        print("Merging duplicate constructs...")
        constructs_by_name = {}
        for construct in constructs_result:
            name = construct['name']
            if name not in constructs_by_name:
                constructs_by_name[name] = construct
            else:
                # Merge data from duplicate construct
                existing = constructs_by_name[name]
                # Merge definitions
                if construct.get('definitions'):
                    existing['definitions'].extend(construct['definitions'])
                # Merge measurements  
                if construct.get('measurements'):
                    existing['measurements'].extend(construct['measurements'])
                # Merge dimensions
                if construct.get('dimensions'):
                    existing['dimensions'].extend(construct['dimensions'])
                # Merge parent constructs
                if construct.get('parent_constructs'):
                    existing['parent_constructs'].extend(construct['parent_constructs'])
                # Merge similar constructs
                if construct.get('similar_constructs'):
                    existing['similar_constructs'].extend(construct['similar_constructs'])
                # Merge similar_to constructs
                if construct.get('similar_to_constructs'):
                    existing['similar_to_constructs'].extend(construct['similar_to_constructs'])
                # Merge paper IDs
                if construct.get('paper_ids'):
                    existing['paper_ids'].extend(construct['paper_ids'])
        
        # Remove duplicates from merged arrays (preserving objects) and compute best_description
        for name, construct in constructs_by_name.items():
            # definitions: dedup by (paper_uid, definition)
            defs = construct.get('definitions') or []
            defs_map = {}
            for d in defs:
                if isinstance(d, dict):
                    key = (d.get('paper_uid'), (d.get('definition') or '').strip())
                    if key not in defs_map:
                        defs_map[key] = d
            construct['definitions'] = list(defs_map.values())

            # measurements: dedup by (paper_uid, name)
            meas_list = construct.get('measurements') or []
            meas_map = {}
            for m in meas_list:
                if isinstance(m, dict):
                    key = (m.get('paper_uid'), (m.get('name') or '').strip().lower())
                    if key not in meas_map:
                        meas_map[key] = m
            construct['measurements'] = list(meas_map.values())

            # set-like arrays
            if construct.get('dimensions'):
                construct['dimensions'] = list(sorted(set(construct['dimensions'])))
            if construct.get('parent_constructs'):
                construct['parent_constructs'] = list(sorted(set(construct['parent_constructs'])))
            if construct.get('similar_constructs'):
                sim_map = {}
                for s in construct.get('similar_constructs') or []:
                    if isinstance(s, dict):
                        sim_map[s.get('name')] = s
                construct['similar_constructs'] = list(sim_map.values())
            if construct.get('similar_to_constructs'):
                simto_map = {}
                for s in construct.get('similar_to_constructs') or []:
                    if isinstance(s, dict):
                        simto_map[s.get('name')] = s
                construct['similar_to_constructs'] = list(simto_map.values())
            if construct.get('paper_ids'):
                construct['paper_ids'] = list(sorted(set(construct['paper_ids'])))

            # compute best_description for summary panel
            best_desc = None
            if construct['definitions']:
                texts = [d.get('definition') for d in construct['definitions'] if d and (d.get('definition') or '').strip()]
                if texts:
                    best_desc = max(texts, key=lambda x: len(x or ''))
            if not best_desc:
                best_desc = construct.get('description')
            construct['best_description'] = best_desc
        
        constructs_result = list(constructs_by_name.values())
        print(f"After merging: {len(constructs_result)} unique constructs")
        
        print("Fetching relationships...")
        relationships_result = graph.run(relationships_query).data()
        print(f"Found {len(relationships_result)} relationships")

        # Build reified relationship nodes dataset (RI nodes) for toggled mode
        reified_nodes = []
        reified_edges = []
        for rel in relationships_result:
            for ri in rel.get('relationship_instances') or []:
                ri_id = ri.get('uuid')
                if not ri_id:
                    continue
                # RI node label: compact summary
                symbols = []
                if (ri.get('evidence_type') or '').lower() == 'quantitative':
                    dirl = (ri.get('effect_direction') or '').lower()
                    if 'inverted' in (ri.get('non_linear_type') or '').lower(): symbols.append('∩')
                    elif 'u' in (ri.get('non_linear_type') or '').lower(): symbols.append('∪')
                    elif 's' in (ri.get('non_linear_type') or '').lower(): symbols.append('S')
                    elif dirl == 'positive': symbols.append('+')
                    elif dirl == 'negative': symbols.append('−')
                    else: symbols.append('·')
                elif (ri.get('evidence_type') or '').lower() == 'qualitative':
                    symbols.append('Q')
                label = ''.join(symbols) or '·'

                # Create RI node
                reified_nodes.append({
                    'id': ri_id,
                    'label': label,
                    'shape': 'diamond',
                    'color': { 'background': '#e5e7eb', 'border': '#9ca3af', 'highlight': { 'background': '#f3f4f6', 'border': '#9ca3af' } },
                    'font': { 'color': '#374151', 'size': 12, 'face': 'Times New Roman', 'bold': True },
                    'size': 16,
                    'shadow': { 'enabled': True, 'color': 'rgba(0,0,0,0.25)', 'size': 6, 'x': 3, 'y': 3 }
                })

                # Connect subject -> RI -> object
                reified_edges.append({ 'from': rel['source_construct'], 'to': ri_id, 'arrows': { 'to': { 'enabled': True, 'scaleFactor': 0.6 } }, 'color': { 'color': '#9ca3af' }, 'width': 1.5 })
                reified_edges.append({ 'from': ri_id, 'to': rel['target_construct'], 'arrows': { 'to': { 'enabled': True, 'scaleFactor': 0.6 } }, 'color': { 'color': '#9ca3af' }, 'width': 1.5 })

                # Moderators/Mediators -> RI
                for m in (ri.get('moderators') or []):
                    reified_edges.append({ 'from': m, 'to': ri_id, 'dashes': True, 'color': { 'color': '#9ca3af' } })
                for m in (ri.get('mediators') or []):
                    reified_edges.append({ 'from': m, 'to': ri_id, 'color': { 'color': '#9ca3af' }, 'width': 2 })

        print("Fetching papers...")
        papers_result = graph.run(papers_query).data()
        print(f"Found {len(papers_result)} papers")
        
        # If no relationships found, create some dummy connections for visualization
        if not relationships_result:
            print("No relationships found, creating dummy connections for visualization...")
            # Create a simple network by connecting constructs in sequence
            dummy_relationships = []
            for i in range(len(constructs_result) - 1):
                dummy_relationships.append({
                    'source_construct': constructs_result[i]['name'],
                    'target_construct': constructs_result[i + 1]['name'],
                    'type': 'CONNECTED_TO',
                    'status': 'dummy',
                    'effect_direction': 'neutral',
                    'papers': [],
                    'statistics': []
                })
            relationships_result = dummy_relationships
        
        # Serialize the data
        constructs_result = serialize_neo4j_data(constructs_result)
        relationships_result = serialize_neo4j_data(relationships_result)
        papers_result = serialize_neo4j_data(papers_result)
        
        return constructs_result, relationships_result, papers_result
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return [], []

# --- 4. HTML & JAVASCRIPT GENERATION ---
def get_embeddings_from_qdrant(construct_names):
    """Get embeddings from Qdrant vector database for given construct names."""
    try:
        client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Get all points from the collection
        points = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=10000,  # Adjust based on your data size
            with_payload=True,
            with_vectors=True
        )[0]  # scroll returns (points, next_page_offset)
        
        # Create a mapping from construct name to embedding
        name_to_embedding = {}
        for point in points:
            if point.payload:
                # Try different possible field names for construct name
                construct_name = None
                for field in ['construct_name', 'name', 'term', 'preferred_name']:
                    if field in point.payload:
                        construct_name = point.payload[field]
                        break
                
                if construct_name and construct_name in construct_names:
                    name_to_embedding[construct_name] = point.vector
        
        # Debug info removed for cleaner output
        
        print(f"Found {len(name_to_embedding)} embeddings in Qdrant for {len(construct_names)} constructs")
        
        # Return embeddings in the same order as construct_names
        embeddings = []
        missing_names = []
        for name in construct_names:
            if name in name_to_embedding:
                embeddings.append(name_to_embedding[name])
            else:
                missing_names.append(name)
                # Use random vector for missing embeddings to avoid clustering at origin
                # This ensures better distribution in t-SNE
                random_vector = np.random.normal(0, 0.1, 384).tolist()
                embeddings.append(random_vector)
        
        if missing_names:
            print(f"Warning: Missing embeddings for constructs: {missing_names[:5]}...")
            print(f"Using random vectors for {len(missing_names)} constructs to improve layout distribution")
        
        return np.array(embeddings, dtype=float)
        
    except Exception as e:
        print(f"Error getting embeddings from Qdrant: {e}")
        # Return zero vectors as fallback
        return np.zeros((len(construct_names), 384), dtype=float)


def _compute_layouts(constructs_data, relationships_data):
    """Compute two layout modes:
    - embeddingPositions: 2D positions from sentence-transformer embeddings (PCA to 2D)
    - centralityPositions: radial layout from degree centrality (connections in relationships_data)
    Returns dicts keyed by construct name: { name: {x, y} }
    """
    names = [c.get('name') for c in constructs_data]
    texts = []
    for c in constructs_data:
        # Prefer rich text for embedding: description + first definition
        desc = c.get('description') or ''
        defs = c.get('definitions') or []
        def0 = (defs[0].get('definition') if defs and isinstance(defs[0], dict) else '') or ''
        texts.append((c.get('name') or '') + '\n' + desc + '\n' + def0)

    # Get embeddings from Qdrant vector database (already computed by build_graph.py)
    print(f"Getting {len(texts)} embeddings from Qdrant vector database...")
    try:
        vectors = get_embeddings_from_qdrant(names)
        print(f"Successfully retrieved {len(vectors)} embeddings with dimension {vectors.shape[1]}")
    except Exception as e:
        raise RuntimeError(f"Failed to get embeddings from Qdrant: {e}")

    # Use t-SNE for better semantic clustering visualization
    try:
        from sklearn.manifold import TSNE
        
        print(f"Computing t-SNE for {len(vectors)} nodes...")
        # Normalize vectors first for better t-SNE performance
        vectors_normalized = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
        
        # t-SNE with optimized parameters for semantic visualization
        # Adjust perplexity for better handling of mixed real/random vectors
        effective_perplexity = min(15, len(vectors) - 1)  # Lower perplexity for better separation
        
        # Pre-process vectors to improve t-SNE stability
        # Remove any vectors that are all zeros or have very low variance
        vector_variance = np.var(vectors_normalized, axis=1)
        valid_indices = vector_variance > 1e-6  # Keep vectors with meaningful variance
        
        if np.sum(valid_indices) < len(vectors) * 0.5:
            print(f"Warning: {len(vectors) - np.sum(valid_indices)} vectors have very low variance, this may affect t-SNE quality")
        
        tsne = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            random_state=42,
            metric='cosine',  # Use cosine similarity for semantic vectors
            init='pca',  # Initialize with PCA for better convergence
            learning_rate='auto',
            n_iter=1000,  # Increase iterations for better convergence
            early_exaggeration=12.0  # Increase separation between clusters
        )
        
        coords2d = tsne.fit_transform(vectors_normalized)
        print("t-SNE completed successfully")
        
        # Post-process coordinates to ensure they are within reasonable bounds
        coords2d = np.clip(coords2d, -1000, 1000)  # Clip extreme values
        
    except Exception as e:
        print(f"t-SNE failed: {e}, falling back to PCA")
        # Fallback to PCA if t-SNE fails
        try:
            X = vectors - vectors.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            coords2d = U[:, :2] * S[:2]
        except Exception:
            print("PCA also failed, using fallback grid layout")
            # Ultimate fallback: create a grid layout to avoid overlapping
            n = len(texts)
            grid_size = int(np.ceil(np.sqrt(n)))
            coords2d = np.zeros((n, 2))
            for i in range(n):
                row = i // grid_size
                col = i % grid_size
                coords2d[i, 0] = (col - grid_size/2) * 100
                coords2d[i, 1] = (row - grid_size/2) * 100
    
    # Normalize to a nice canvas radius
    coords2d = coords2d / (np.max(np.linalg.norm(coords2d, axis=1)) + 1e-9) * 600

    embedding_positions = {names[i]: {"x": float(coords2d[i, 0]), "y": float(coords2d[i, 1])} for i in range(len(names))}

    # Debug: Show basic semantic similarity info
    if len(names) > 1:
        try:
            # Compute cosine similarity matrix
            vectors_normalized = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
            similarity_matrix = np.dot(vectors_normalized, vectors_normalized.T)
            
            # Show top 3 most similar pairs (excluding self-similarity)
            similarities = []
            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    sim = similarity_matrix[i, j]
                    similarities.append((sim, names[i], names[j]))
            
            if similarities:
                similarities.sort(reverse=True)
                print(f"\nTop similar node pairs:")
                for i, (sim, name1, name2) in enumerate(similarities[:3]):
                    print(f"  {name1} <-> {name2}: {sim:.4f}")
        except Exception as e:
            print(f"Error computing similarity matrix: {e}")
    else:
        print(f"\nFound {len(names)} node - similarity calculation not applicable")

    # Degree centrality from relationships
    degree = {n: 0 for n in names}
    for r in relationships_data:
        s = r.get('source_construct'); t = r.get('target_construct')
        if s in degree: degree[s] += 1
        if t in degree: degree[t] += 1

    # Radial layout: sort by degree, place high degree near center
    sorted_names = sorted(names, key=lambda n: -degree.get(n, 0))
    # Partition into rings by quantiles
    counts = [degree.get(n, 0) for n in sorted_names]
    if counts:
        q1 = np.percentile(counts, 25)
        q2 = np.percentile(counts, 50)
        q3 = np.percentile(counts, 75)
    else:
        q1 = q2 = q3 = 0

    rings = {"inner": [], "mid": [], "outer": [], "edge": []}
    for n in sorted_names:
        d = degree.get(n, 0)
        if d >= q3:
            rings["inner"].append(n)
        elif d >= q2:
            rings["mid"].append(n)
        elif d >= q1:
            rings["outer"].append(n)
        else:
            rings["edge"].append(n)

    # Obsidian-like centrality layout: important nodes (high degree) near center,
    # others spread radially within a disk (not on a single circle)
    # Use a golden-angle spiral to distribute angles deterministically.
    max_degree = max(degree.values()) if degree else 0
    r_min, r_max = 80.0, 700.0
    golden_angle = np.pi * (3 - np.sqrt(5))  # ~2.399963
    centrality_positions = {}
    # Order nodes by decreasing degree (stable for determinism)
    ordered = sorted(names, key=lambda n: (-degree.get(n, 0), n))
    for i, n in enumerate(ordered):
        d = degree.get(n, 0)
        if max_degree > 0:
            # 0 for max degree, 1 for min degree
            norm = 1.0 - (d / max_degree)
        else:
            norm = 1.0
        # Ease exponent so differences near the center are emphasized
        eased = norm ** 0.85
        radius = r_min + eased * (r_max - r_min)
        angle = (i * golden_angle) % (2 * np.pi)
        centrality_positions[n] = {"x": float(radius * np.cos(angle)), "y": float(radius * np.sin(angle))}

    # Ensure all nodes have a position (fallback)
    for n in names:
        if n not in centrality_positions:
            centrality_positions[n] = {"x": 0.0, "y": 0.0}

    return embedding_positions, centrality_positions


def create_constructs_network_page(constructs_data, relationships_data, papers_data):
    """Create an enhanced network visualization page with 3-panel layout and paper filtering."""
    
    # Enhanced color scheme that complements the background
    node_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
    ]
    
    # Obsidian-like subtle greys for edges
    edge_colors = {
        'positive': '#9ca3af',
        'negative': '#9ca3af',
        'causal': '#9ca3af',
        'correlational': '#9ca3af',
        'default': '#9ca3af'
    }
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>构型关系网络图 - Enhanced</title>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        
        <!-- MathJax for mathematical formula rendering -->
        <script>
            MathJax = {{
                tex: {{
                    inlineMath: [],
                    displayMath: [['$$', '$$']],
                    processEscapes: true,
                    processEnvironments: true
                }},
                svg: {{
                    fontCache: 'global'
                }},
                startup: {{
                    pageReady: () => {{
                        MathJax.startup.defaultPageReady();
                        // Re-render math after dynamic content is loaded
                        if (window.renderMathAfterLoad) {{
                            window.renderMathAfterLoad();
                        }}
                    }}
                }}
            }};
        </script>
        <!-- Removed polyfill.io to avoid ERR_NAME_NOT_RESOLVED in some environments; modern browsers don't need it -->
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Times New Roman', Times, serif !important;
                background: #1e1e1e;
                min-height: 100vh;
                color: #c9d1d9;
            }}
            
            * {{
                font-family: 'Times New Roman', Times, serif !important;
            }}
            
            .global-toolbar {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.8rem 1.5rem;
                background: rgba(60, 60, 60, 0.5);
                backdrop-filter: blur(10px);
                border-bottom: 1px solid #3c3c3c;
                gap: 1rem;
                flex-wrap: wrap;
                position: relative;
                z-index: 200000;
            }}
            
            .toolbar-section {{
                display: flex;
                align-items: center;
                gap: 0.8rem;
            }}
            

            
            .search-input {{
                padding: 10px 16px;
                border-radius: 20px;
                border: 1px solid #3c3c3c;
                background: #2b2b2b;
                color: #e5e7eb;
                min-width: 400px;
                font-size: 0.95rem;
                position: relative;
            }}
            
            .toolbar-section {{
                position: relative;
            }}
            
            .search-dropdown {{
                position: absolute;
                top: calc(100% + 4px);
                left: 0;
                width: 400px;
                background: rgba(43, 43, 43, 0.95);
                border-radius: 8px;
                border: 1px solid rgba(156, 163, 175, 0.3);
                max-height: 300px;
                overflow-y: auto;
                z-index: 10000;
                display: none;
                backdrop-filter: blur(12px);
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            }}
            
            .search-result {{
                padding: 12px 16px;
                cursor: pointer;
                border-bottom: 1px solid rgba(255,255,255,0.1);
                color: rgba(255,255,255,0.9);
                transition: all 0.2s ease;
            }}
            
            .search-result:last-child {{
                border-bottom: none;
            }}
            
            .search-result:hover,
            .search-result.selected {{
                background: rgba(156, 163, 175, 0.2);
                color: #fff;
            }}
            
            .search-result-title {{
                font-weight: 500;
                font-size: 0.9rem;
            }}
            
            .search-result-meta {{
                font-size: 0.8rem;
                color: rgba(255,255,255,0.6);
                margin-top: 2px;
            }}
            
            .timeline-controls {{
                display: flex;
                align-items: center;
                gap: 0.8rem;
                background: rgba(60,60,60,0.5);
                padding: 8px 12px;
                border-radius: 20px;
            }}
            
            .year-slider {{
                width: 200px;
                accent-color: #9ca3af;
            }}
            
            .view-presets {{
                display: flex;
                gap: 4px;
                background: rgba(255,255,255,0.1);
                padding: 4px;
                border-radius: 20px;
            }}
            
            .preset-btn, .layout-btn {{
                background: none;
                border: none;
                color: rgba(255,255,255,0.7);
                padding: 6px 12px;
                border-radius: 16px;
                cursor: pointer;
                transition: all 0.2s;
                font-size: 0.9rem;
            }}
            
            .preset-btn:hover, .layout-btn:hover {{
                color: white;
                background: rgba(255,255,255,0.1);
            }}
            
            .preset-btn.active, .layout-btn.active {{
                color: white;
                background: rgba(156, 163, 175, 0.8);
            }}
            
            .filters-section {{
                display: flex;
                gap: 0.5rem;
                position: relative;
                z-index: 200000;
            }}
            
            .filter-dropdown {{
                background: rgba(255,255,255,0.1);
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 8px;
                position: relative;
                z-index: 210000;
            }}
            
            .filter-dropdown summary {{
                padding: 6px 10px;
                color: white;
                cursor: pointer;
                list-style: none;
                font-size: 0.9rem;
            }}
            
            .filter-dropdown summary::-webkit-details-marker {{
                display: none;
            }}
            
            .filter-content {{
                position: fixed;
                top: auto;
                left: auto;
                background: rgba(0,0,0,0.95);
                border: 1px solid rgba(255,255,255,0.4);
                border-radius: 8px;
                padding: 12px;
                min-width: 180px;
                z-index: 220000 !important;
                box-shadow: 0 8px 32px rgba(0,0,0,0.8);
                backdrop-filter: blur(16px);
                pointer-events: auto;
            }}
            
            .filter-content label {{
                display: block;
                color: white;
                padding: 4px 0;
                font-size: 0.85rem;
                cursor: pointer;
            }}
            
            .filter-content input[type="checkbox"] {{
                margin-right: 6px;
                accent-color: #9ca3af;
            }}
            
            .hud-overlay {{
                position: absolute;
                z-index: 100;
                pointer-events: none;
            }}
            
            .hud-overlay.top-left {{
                top: 10px;
                left: 10px;
            }}
            
            .hud-overlay.top-right {{
                top: 10px;
                right: 10px;
            }}
            
            .hud-overlay.bottom-right {{
                bottom: 10px;
                right: 10px;
            }}
            
            .selection-summary, .legend {{
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 10px 12px;
                border-radius: 8px;
                font-size: 0.85rem;
                line-height: 1.4;
                border: 1px solid rgba(255,255,255,0.2);
                backdrop-filter: blur(8px);
            }}
            
            .selection-summary div, .legend div {{
                margin: 2px 0;
            }}
            
            #toast-container {{
                display: flex;
                flex-direction: column;
                gap: 8px;
            }}
            
            .toast {{
                background: rgba(156, 163, 175, 0.9);
                color: white;
                padding: 8px 12px;
                border-radius: 10px;
                font-size: 0.8rem;
                animation: slideIn 0.3s ease;
                pointer-events: auto;
                cursor: pointer;
            }}
            
            .toast.success {{
                background: rgba(134, 239, 172, 0.9);
            }}
            
            .toast.warning {{
                background: rgba(253, 224, 71, 0.9);
            }}
            
            .toast.error {{
                background: rgba(252, 165, 165, 0.9);
            }}
            
            @keyframes slideIn {{
                from {{ transform: translateX(100%); opacity: 0; }}
                to {{ transform: translateX(0); opacity: 1; }}
            }}
            
            .content {{
                display: grid;
                /* 20% / 60% / 20% with minimums to ensure readability */
                grid-template-columns: minmax(280px, 1fr) minmax(720px, 3fr) minmax(320px, 1fr);
                gap: 16px;
                padding: 16px;
                height: calc(100vh - 64px);
                position: relative;
                z-index: 1;
            }}
            
            .stats {{
                display: flex;
                justify-content: center;
                gap: 3rem;
                margin-bottom: 2rem;
            }}
            
            .stat-item {{
                background: rgba(255, 255, 255, 0.1);
                padding: 1rem 2rem;
                border-radius: 15px;
                text-align: center;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
            
            .stat-number {{
                font-size: 2rem;
                font-weight: bold;
                color: #fff;
                display: block;
            }}
            
            .stat-label {{
                color: rgba(255, 255, 255, 0.8);
                font-size: 0.9rem;
                margin-top: 0.5rem;
            }}
            
            .sidebar, .details {{
                background: rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.15);
                backdrop-filter: blur(14px);
                overflow: hidden;
                display: flex;
                flex-direction: column;
            }}
            .sidebar-header {{
                padding: 12px 14px;
                color: #fff;
                font-weight: 600;
                border-bottom: 1px solid rgba(255,255,255,0.15);
            }}
            .paper-tools {{
                display: flex;
                gap: 8px;
                padding: 10px 12px;
            }}
            .paper-tools input {{
                flex: 1;
                padding: 8px 10px;
                border-radius: 8px;
                border: 1px solid rgba(255,255,255,0.25);
                background: rgba(0,0,0,0.2);
                color: #fff;
            }}
            .paper-action-buttons {{
                display: flex;
                gap: 8px;
                padding: 0 12px 8px 12px;
            }}
            .paper-action-btn {{
                flex: 1;
                padding: 6px 12px;
                border: 1px solid rgba(156, 163, 175, 0.4);
                background: rgba(156, 163, 175, 0.1);
                color: rgba(255, 255, 255, 0.9);
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.85rem;
                font-weight: 500;
                transition: all 0.2s ease;
                backdrop-filter: blur(4px);
            }}
            .paper-action-btn:hover {{
                background: rgba(156, 163, 175, 0.2);
                border-color: rgba(156, 163, 175, 0.6);
                color: #fff;
            }}
            .paper-action-btn:active {{
                background: rgba(156, 163, 175, 0.3);
                transform: translateY(1px);
            }}
            .paper-list {{
                overflow-y: auto;
                padding: 8px 12px 12px 12px;
            }}
            .paper-item {{
                display: flex;
                align-items: flex-start;
                gap: 8px;
                padding: 8px;
                border-radius: 8px;
                color: rgba(255,255,255,0.95);
            }}
            .paper-item:hover {{
                background: rgba(255,255,255,0.08);
            }}
            .paper-item input[type="checkbox"] {{
                appearance: none;
                -webkit-appearance: none;
                width: 15px;
                height: 15px;
                border: 2px solid #6b7280;
                border-radius: 4px;
                background-color: transparent;
                margin-top: 2.3px;
                margin-right: 8px;
                cursor: pointer;
                position: relative;
                transition: all 0.2s ease;
                flex-shrink: 0;
            }}
            .paper-item input[type="checkbox"]:checked {{
                background-color: #6b7280;
                border-color: #6b7280;
            }}
            .paper-item input[type="checkbox"]:checked::after {{
                content: '✓';
                position: absolute;
                top: -0.5px;
                left: 1.8px;
                color: white;
                font-size: 10px;
                font-weight: bold;
                line-height: 14px;
            }}
            #network-container {{
                width: 100%;
                height: 100%;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                position: relative;
                overflow: hidden;
                z-index: 1;
            }}
            .details {{
                color: #fff;
                padding: 12px 16px;
                overflow-y: auto;
                overflow-x: hidden;
                position: relative;
                z-index: 1;
                font-family: 'Times New Roman', Times, serif !important;
                word-wrap: break-word;
                word-break: break-word;
                max-width: 100%;
                box-sizing: border-box;
            }}
            .detail-section {{
                margin-bottom: 20px;
                border-bottom: 1px solid rgba(255,255,255,0.15);
                padding-bottom: 16px;
            }}
            .detail-section:last-child {{
                border-bottom: none;
                margin-bottom: 0;
            }}
            .detail-section.no-border {{
                border-bottom: none;
            }}
            
            .tooltip {{
                position: absolute;
                background: rgba(0, 0, 0, 0.95);
                color: white;
                padding: 1rem;
                border-radius: 10px;
                font-size: 0.9rem;
                max-width: 400px;
                z-index: 500;
                pointer-events: none;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
                opacity: 0;
                transition: opacity 0.3s ease;
                font-family: 'Times New Roman', Times, serif;
            }}
            
            .tooltip.show {{
                opacity: 1;
            }}
            
            .tooltip-section {{
                margin-bottom: 1rem;
                padding-bottom: 0.8rem;
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            }}
            
            .tooltip-section:last-child {{
                border-bottom: none;
                margin-bottom: 0;
            }}
            
            .tooltip-title {{
                font-weight: bold;
                color: #9ca3af;
                margin-bottom: 0.5rem;
                font-size: 1rem;
            }}
            
            .tooltip-content {{
                line-height: 1.4;
                color: rgba(255, 255, 255, 0.9);
            }}
            
            .paper-info {{
                background: rgba(255, 255, 255, 0.1);
                padding: 0.5rem;
                margin: 0.3rem 0;
                border-radius: 5px;
                border-left: 3px solid #9ca3af;
            }}
            
            .stat-info {{
                background: rgba(255, 255, 255, 0.1);
                padding: 0.5rem;
                margin: 0.3rem 0;
                border-radius: 5px;
                border-left: 3px solid #9ca3af;
            }}
            
            /* Allow MathJax display math to be properly centered like in academic papers */
            mjx-container[display="true"] {{
                display: block !important;
                margin: 1em auto !important;
                text-align: center !important;
            }}
            .MathJax_Display {{
                display: block !important;
                margin: 1em auto !important;
                text-align: center !important;
            }}
            
            /* Style for mathematical expressions in paragraphs */
            .math-block {{
                display: block;
                text-align: center;
                margin: 1em 0;
            }}
            
            /* Moderator-specific styles */
            .moderator-node {{
                border: 2px solid #6b7280 !important;
                box-shadow: 0 0 10px rgba(107, 114, 128, 0.5) !important;
            }}
            
            .moderator-edge {{
                stroke-dasharray: 8, 4 !important;
                stroke-width: 2 !important;
                stroke: #6b7280 !important;
            }}
            
            .moderator-triangle {{
                background: rgba(107, 114, 128, 0.1) !important;
                border: 1px solid rgba(107, 114, 128, 0.3) !important;
            }}
        </style>
    </head>
    <body>
        <div class="global-toolbar">
            <div class="toolbar-section">
                <input id="global-search" placeholder="搜索构型 (Enter跳转, ↑↓切换)" class="search-input" />
                <div id="search-results" class="search-dropdown"></div>
            </div>
            
            <div class="toolbar-section">
                <div class="timeline-controls">
                    <input id="year-range" type="range" min="1900" max="2100" step="1" value="2100" class="year-slider" />
                    <span id="year-label">年份: 全部</span>
                </div>
            </div>

            <div class="toolbar-section">
                <div class="view-presets">
                    <button class="preset-btn active" data-preset="overview">总览</button>
                    <button class="preset-btn" data-preset="causal">因果</button>
                    <button class="preset-btn" data-preset="correlation">相关</button>
                    <button class="preset-btn" data-preset="dense">密集</button>
                </div>
                <div class="view-presets" style="margin-left:8px">
                                    <button class="layout-btn" id="layout-centrality">中心度布局</button>
                <button class="layout-btn" id="layout-embedding">语义布局</button>
                </div>

            </div>
            
            <div class="toolbar-section">
                <div class="filters-section">
                    <details class="filter-dropdown">
                        <summary>关系过滤 ▼</summary>
                        <div class="filter-content">
                            <label><input type="radio" name="rel-filter" id="filter-rel-all" checked> 全部</label>
                            <label><input type="radio" name="rel-filter" id="filter-rel-pos"> 正向</label>
                            <label><input type="radio" name="rel-filter" id="filter-rel-neg"> 负向</label>
                            <label><input type="radio" name="rel-filter" id="filter-rel-insig"> 非显著</label>
                            <label><input type="radio" name="rel-filter" id="filter-rel-s"> S 型</label>
                            <label><input type="radio" name="rel-filter" id="filter-rel-u"> U 型</label>
                            <label><input type="radio" name="rel-filter" id="filter-rel-invu"> 倒 U 型</label>
                        </div>
                    </details>
                    <!-- 证据强度筛选暂时移除 -->
                </div>
            </div>
        </div>

        <div class="content">
            <div class="sidebar">
                <div class="sidebar-header">论文列表</div>
                <div class="paper-tools">
                    <input id="paper-search" placeholder="搜索论文标题/作者" />
                </div>
                <div class="paper-action-buttons">
                    <button id="select-all" class="paper-action-btn">全选</button>
                    <button id="clear-all" class="paper-action-btn">清空</button>
                </div>
                <div class="paper-list" id="paper-list"></div>
            </div>
            <div id="network-container">
                <!-- HUD Overlays -->
                <div class="hud-overlay top-left">
                    <div class="selection-summary" id="selection-summary">
                        <div><b>当前视图</b></div>
                        <div>论文: <span id="papers-count">0</span></div>
                        <div>构型: <span id="nodes-count">0</span></div>
                        <div>关系: <span id="edges-count">0</span></div>
                    </div>
                </div>
                
                <div class="hud-overlay top-right">
                    <div id="toast-container"></div>
                </div>
                
                <div class="hud-overlay bottom-right"></div>
            </div>
            <div class="details" id="details-panel">
                <div style="opacity:0.8">点击中间的节点或连线查看详细信息</div>
            </div>
        </div>

        <div id="tooltip" class="tooltip"></div>
        
        <script>
            // Data from Python
            const constructsData = {json.dumps(constructs_data, ensure_ascii=False)};
            const relationshipsData = {json.dumps(relationships_data, ensure_ascii=False)};
            const papersData = {json.dumps(papers_data, ensure_ascii=False)};
            
            // --- Math helpers ---
            // Process text with proper math formula formatting for academic display
            function normalizeMathInline(text) {{
                if (text == null) return '';
                try {{
                    let s = String(text);
                    
                    // Step 1: Detect and format display math blocks ($$...$$ or \[...\])
                    // Split text around math blocks and format them properly
                    s = s.replace(/(\$\$[^$]+\$\$|\\\[[^\]]+\\\])/g, function(match, mathBlock, offset) {{
                        // Check if the text before the math block ends with a colon
                        const beforeMath = s.substring(0, offset).trim();
                        const needsColon = beforeMath.length > 0 && !beforeMath.endsWith(':') && !beforeMath.endsWith('：');
                        
                        // Add colon if needed, then wrap math block
                        const colonPrefix = needsColon ? ':' : '';
                        return `${{colonPrefix}}<div class="math-block">${{mathBlock}}</div>`;
                    }});
                    
                    // Step 2: PDF artifact cleanup (no MathJax injection)
                    s = s.replace(/\/SL([a-zA-Z]+)/g, function(_, sym) {{
                        const map = {{alpha:'α', beta:'β', gamma:'γ', delta:'δ', lambda:'λ', theta:'θ'}};
                        return map[sym.toLowerCase()] || sym; // prefer unicode over LaTeX to avoid math mode
                    }});
                    s = s.replace(/\/lparenori/g, '(')
                         .replace(/\/rparenori/g, ')')
                         .replace(/\/commaori/g, ',')
                         .replace(/\/lbracketori/g, '[')
                         .replace(/\/rbracketori/g, ']');
                    
                    // Step 3: abbreviations normalization only (avoid JS \b here to prevent Python escape issues)
                    s = s.replace(/i\.\s*e\./gi, 'i.e.');
                    s = s.replace(/e\.\s*g\./gi, 'e.g.');
                    s = s.replace(/etc\./gi, 'etc.');
                    s = s.replace(/e\.\s*g\.\s*,/g, 'e.g.,').replace(/i\.\s*e\.\s*,/g, 'i.e.,');
                    
                    // Step 4: camelCase split only; do not touch punctuation spacing
                    s = s.replace(/([a-z])([A-Z])/g, '$1 $2');
                    
                    // Step 5: sanitize stray math inline markers to prevent unwanted math mode
                    s = s.split('\\(').join('(').split('\\)').join(')');
                    
                    return s;
                }} catch (e) {{
                    return String(text);
                }}
            }}
            function htmlWithMathSafe(text) {{
                return normalizeMathInline(text);
            }}

            // Debounced MathJax typeset for dynamic content
            function typesetDebounced(el) {{
                if (!window.MathJax || !el) return;
                if (window.__mjxDebounce) clearTimeout(window.__mjxDebounce);
                window.__mjxDebounce = setTimeout(() => {{
                    try {{ window.MathJax.typesetPromise([el]); }} catch (e) {{}}
                }}, 80);
            }}
            const allYears = papersData.map(p => p.year).filter(Boolean);
            const minYear = allYears.length ? Math.min(...allYears) : 1900;
            const maxYear = allYears.length ? Math.max(...allYears) : new Date().getFullYear();
            
            // Title case formatting function
            function formatTitle(title) {{
                if (!title) return '无标题';
                return title.split(' ').map(word => {{
                    if (word.length === 0) return word;
                    // Keep special characters and numbers as is, only capitalize first letter of words
                    const firstChar = word.charAt(0);
                    const rest = word.slice(1);
                    if (/[A-Za-z]/.test(firstChar)) {{
                        return firstChar.toUpperCase() + rest.toLowerCase();
                    }}
                    return word;
                }}).join(' ');
            }}
            
            console.log('数据加载完成:', constructsData.length, '个构型,', relationshipsData.length, '个关系');

            // Layout data injected from Python
            const embeddingPositions = __EMBED_POS__;
            const centralityPositions = __CENTRAL_POS__;
            let layoutMode = 'centrality'; // 'centrality' | 'embedding'
            
            // Tooltip management
            let tooltip = null;
            let tooltipTimeout = null;
            
            function createTooltip() {{
                tooltip = document.getElementById('tooltip');
                if (!tooltip) {{
                    tooltip = document.createElement('div');
                    tooltip.id = 'tooltip';
                    tooltip.className = 'tooltip';
                    document.body.appendChild(tooltip);
                }}
            }}
            
            function showNodeTooltip(node, event) {{
                if (tooltipTimeout) clearTimeout(tooltipTimeout);
                
                tooltipTimeout = setTimeout(() => {{
                    // Use node.id (we render empty internal labels and draw external labels)
                    const construct = constructsData.find(c => c.name === node.id);
                    if (!construct) return;
                    
                    let content = `<div class="tooltip-title">${{construct.name}}</div>`;
                    
                    // Definitions section
                    if (construct.definitions && construct.definitions.length > 0) {{
                        content += `<div class="tooltip-section">
                            <div class="tooltip-title">定义来源</div>`;
                        construct.definitions.forEach(def => {{
                            if (def.definition && def.paper_title) {{
                                content += `<div class="paper-info">
                                    <strong>定义:</strong> ${{htmlWithMathSafe(def.definition)}}<br>
                                    <strong>来源:</strong> ${{def.paper_title}} ${{(def.paper_authors || []).join(', ')}} (${{def.paper_year || 'N/A'}})
                                </div>`;
                            }}
                        }});
                        content += '</div>';
                    }}
                    
                    // Dimensions section
                    if (construct.dimensions && construct.dimensions.length > 0) {{
                        content += `<div class="tooltip-section">
                            <div class="tooltip-title">维度</div>
                            <div class="tooltip-content">${{construct.dimensions.join(', ')}}</div>
                        </div>`;
                    }}
                    
                    // Parent constructs section
                    if (construct.parent_constructs && construct.parent_constructs.length > 0) {{
                        content += `<div class="tooltip-section">
                            <div class="tooltip-title">所属构型</div>
                            <div class="tooltip-content">${{construct.parent_constructs.join(', ')}}</div>
                        </div>`;
                    }}
                    
                    // Combined similar constructs section (merge both directions)
                    {{
                        const allSimilar = [];
                        if (construct.similar_constructs && construct.similar_constructs.length > 0) {{
                            construct.similar_constructs.forEach(s => {{ if (s && s.name) allSimilar.push(s.name); }});
                        }}
                        if (construct.similar_to_constructs && construct.similar_to_constructs.length > 0) {{
                            construct.similar_to_constructs.forEach(s => {{ if (s && s.name) allSimilar.push(s.name); }});
                        }}
                        const uniqueSimilar = Array.from(new Set(allSimilar));
                        if (uniqueSimilar.length > 0) {{
                            content += `<div class=\"tooltip-section\">\n                                <div class=\"tooltip-title\">相似构型</div>`;
                            uniqueSimilar.forEach(name => {{
                                content += `<div class=\"tooltip-content\">• ${{name}}</div>`;
                            }});
                            content += `</div>`;
                        }}
                    }}
                    
                    // Measurements section
                    if (construct.measurements && construct.measurements.length > 0) {{
                        content += `<div class="tooltip-section">
                            <div class="tooltip-title">测量方式</div>`;
                        construct.measurements.forEach(meas => {{
                            if (meas.name && meas.paper_title) {{
                                content += `<div class="paper-info">
                                    <strong>测量:</strong> ${{meas.name}}<br>
                                    <strong>来源:</strong> ${{meas.paper_title}} ${{(meas.paper_authors || []).join(', ')}} (${{meas.paper_year || 'N/A'}})
                                </div>`;
                            }}
                        }});
                        content += '</div>';
                    }}
                    
                    // Check if this is a moderator and show moderator information
                    const moderatorEdges = edges.get().filter(e => 
                        e.moderatorInfo && e.moderatorInfo.moderator === construct.name
                    );
                    
                    if (moderatorEdges.length > 0) {{
                        const moderatorInfo = moderatorEdges[0].moderatorInfo;
                        content += `<div class="tooltip-section" style="border-top: 2px solid #6b7280; margin-top: 16px; padding-top: 16px;">
                            <div class="tooltip-title" style="color: #6b7280;">调节变量信息</div>
                            <div class="tooltip-content">
                                <strong>调节的关系:</strong> ${{moderatorInfo.source}} → ${{moderatorInfo.target}}<br>
                                <strong>调节作用:</strong> 作为调节变量影响上述关系的强度和方向<br>
                                <strong>关系状态:</strong> ${{moderatorInfo.relationship.status || 'N/A'}}<br>
                                <strong>证据类型:</strong> ${{moderatorInfo.relationship.evidence_type || 'N/A'}}<br>
                                <strong>效应方向:</strong> ${{moderatorInfo.relationship.effect_direction || 'N/A'}}
                            </div>
                        </div>`;
                    }}
                    
                    tooltip.innerHTML = content;
                    tooltip.style.left = event.pageX + 15 + 'px';
                    tooltip.style.top = event.pageY - 15 + 'px';
                    tooltip.classList.add('show');
                    typesetDebounced(tooltip);
                }}, 300);
            }}
            
            function showEdgeTooltip(edge, event) {{
                if (tooltipTimeout) clearTimeout(tooltipTimeout);
                
                tooltipTimeout = setTimeout(() => {{
                    // Check if this is a similarity edge
                    if (edge.id && edge.id.startsWith('similar_')) {{
                        showSimilarityTooltip(edge, event);
                        return;
                    }}
                    
                    // Check if this is a moderator edge
                    if (edge.moderatorInfo) {{
                        const moderatorInfo = edge.moderatorInfo;
                        let content = `<div class="tooltip-title" style="color: #6b7280;">调节变量连线</div>
                            <div class="tooltip-section">
                                <div class="tooltip-content">
                                    <strong>调节变量:</strong> ${{moderatorInfo.moderator}}<br>
                                    <strong>调节的关系:</strong> ${{moderatorInfo.source}} → ${{moderatorInfo.target}}<br>
                                    <strong>调节作用:</strong> 作为调节变量影响上述关系的强度和方向<br>
                                    <strong>关系状态:</strong> ${{moderatorInfo.relationship.status || 'N/A'}}<br>
                                    <strong>证据类型:</strong> ${{moderatorInfo.relationship.evidence_type || 'N/A'}}<br>
                                    <strong>效应方向:</strong> ${{moderatorInfo.relationship.effect_direction || 'N/A'}}
                                </div>
                            </div>`;
                        
                        tooltip.innerHTML = content;
                        tooltip.style.left = event.pageX + 15 + 'px';
                        tooltip.style.top = event.pageY - 15 + 'px';
                        tooltip.classList.add('show');
                        typesetDebounced(tooltip);
                        return;
                    }}
                    
                    const relationship = relationshipsData.find(r => 
                        r.source_construct === edge.from && r.target_construct === edge.to
                    );
                    if (!relationship) return;
                    
                    let content = `<div class="tooltip-title">关系详情</div>
                        <div class="tooltip-section">
                            <div class="tooltip-content">
                                <strong>从:</strong> ${{relationship.source_construct}}<br>
                                <strong>到:</strong> ${{relationship.target_construct}}<br>
                                <strong>状态:</strong> ${{relationship.status || 'N/A'}}<br>
                                <strong>证据类型:</strong> ${{relationship.evidence_type || 'N/A'}}<br>
                                <strong>方向:</strong> ${{relationship.effect_direction || 'N/A'}}<br>
                                <strong>因果验证:</strong> ${{relationship.is_validated_causality ? '是' : '否'}}<br>
                                <strong>元分析:</strong> ${{relationship.is_meta_analysis ? '是' : '否'}}
                            </div>
                        </div>`;
                    
                    // Relationship instances section (Blueprint schema)
                    if (relationship.relationship_instances && relationship.relationship_instances.length > 0) {{
                        content += `<div class="tooltip-section">
                            <div class="tooltip-title">关系实例 (${{relationship.relationship_instances.length}}个)</div>`;
                        
                        relationship.relationship_instances.slice(0, 3).forEach((ri, idx) => {{
                            let stats = null;
                            try {{
                                stats = ri.statistical_details ? JSON.parse(ri.statistical_details) : null;
                            }} catch(e) {{
                                stats = ri.statistical_details;
                            }}
                            
                            content += `<div class="stat-info">
                                <strong>论文:</strong> ${{ri.paper_title || 'N/A'}}<br>
                                <strong>描述:</strong> ${{htmlWithMathSafe(ri.description || ri.context_snippet || 'N/A')}}`;
                            
                            if (stats) {{
                                if (stats.p_value !== undefined) content += `<br><strong>P值:</strong> ${{stats.p_value}}`;
                                if (stats.beta_coefficient !== undefined) content += `<br><strong>β系数:</strong> ${{stats.beta_coefficient}}`;
                                if (stats.correlation !== undefined) content += `<br><strong>相关系数:</strong> ${{stats.correlation}}`;
                            }}
                            
                            // Qualitative findings
                            if (ri.qualitative_finding) {{
                                content += `<br><strong>定性发现:</strong> ${{ri.qualitative_finding}}`;
                            }}
                            
                            if (ri.supporting_quote) {{
                                content += `<br><strong>支持引用:</strong> "${{ri.supporting_quote}}"`;
                            }}
                            
                            // Boundary conditions
                            if (ri.boundary_conditions) {{
                                content += `<br><strong>边界条件:</strong> ${{ri.boundary_conditions}}`;
                            }}
                            
                            // Replication outcome
                            if (ri.replication_outcome) {{
                                content += `<br><strong>复制结果:</strong> ${{ri.replication_outcome}}`;
                            }}
                            
                            if (ri.theories && ri.theories.length > 0) {{
                                content += `<br><strong>理论:</strong> ${{ri.theories.join(', ')}}`;
                            }}
                            
                            content += `</div>`;
                        }});
                        
                        if (relationship.relationship_instances.length > 3) {{
                            content += `<div style="opacity:0.7; font-size:0.8em;">还有 ${{relationship.relationship_instances.length - 3}} 个实例...</div>`;
                        }}
                        
                        content += '</div>';
                    }}
                    
                    tooltip.innerHTML = content;
                    tooltip.style.left = event.pageX + 15 + 'px';
                    tooltip.style.top = event.pageY - 15 + 'px';
                    tooltip.classList.add('show');
                }}, 300);
            }}
            
            function showSimilarityTooltip(edge, event) {{
                if (tooltipTimeout) clearTimeout(tooltipTimeout);
                
                tooltipTimeout = setTimeout(() => {{
                    const similarityScore = (edge.similarity_score * 100).toFixed(1);
                    const confidence = (edge.llm_confidence * 100).toFixed(1);
                    
                    let content = `<div class="tooltip-title" style="color: #9ca3af;">相似构型连线</div>
                        <div class="tooltip-section">
                            <div class="tooltip-content">
                                <strong>构型A:</strong> ${{edge.source_name || edge.from}}<br>
                                <strong>构型B:</strong> ${{edge.target_name || edge.to}}<br>
                                <strong>相似度:</strong> <span style="color: #A78BFA;">${{similarityScore}}%</span><br>
                                <strong>置信度:</strong> <span style="color: #A78BFA;">${{confidence}}%</span>
                            </div>
                        </div>`;
                    
                    tooltip.innerHTML = content;
                    tooltip.style.left = event.pageX + 15 + 'px';
                    tooltip.style.top = event.pageY - 15 + 'px';
                    tooltip.classList.add('show');
                }}, 300);
            }}
            
            function hideTooltip() {{
                if (tooltipTimeout) clearTimeout(tooltipTimeout);
                if (tooltip) {{
                    tooltip.classList.remove('show');
                }}
            }}
            
            // Initialize network when page loads
            document.addEventListener('DOMContentLoaded', function() {{
                console.log('页面加载完成，开始创建网络...');
                
                // Check if vis.js is loaded
                if (typeof vis === 'undefined') {{
                    console.error('vis.js library not loaded');
                    return;
                }}
                console.log('vis.js库已加载:', vis);
                
                // Check container
                const container = document.getElementById('network-container');
                if (!container) {{
                    console.error('Container element not found');
                    return;
                }}
                console.log('容器元素:', container);
                
                // Create tooltip
                createTooltip();
                
                // DataSets and filtering state
                const nodes = new vis.DataSet();
                const edges = new vis.DataSet();
                const paperListEl = document.getElementById('paper-list');
                const searchEl = document.getElementById('paper-search');
                const selectAllBtn = document.getElementById('select-all');
                const clearAllBtn = document.getElementById('clear-all');
                let selectedPaperIds = new Set(papersData.map(p => p.id)); // 默认全选

                // Expose datasets globally for helper functions defined outside this scope
                window.nodes = nodes;
                window.edges = edges;
                
                // Enhanced filters and search
                // Relationship filter controls
                const relAll = document.getElementById('filter-rel-all');
                const relPos = document.getElementById('filter-rel-pos');
                const relNeg = document.getElementById('filter-rel-neg');
                const relInsig = document.getElementById('filter-rel-insig');
                const relS = document.getElementById('filter-rel-s');
                const relU = document.getElementById('filter-rel-u');
                const relInvU = document.getElementById('filter-rel-invu');
                const evidenceHasPvalue = null;
                const evidenceHighR = null;
                const evidenceLargeN = null;
                const globalSearch = document.getElementById('global-search');
                const searchResults = document.getElementById('search-results');

                
                // Search state
                let searchCurrentIndex = -1;
                let searchMatches = [];

                function renderPaperList(filterText = '') {{
                    paperListEl.innerHTML = '';
                    const normalized = (filterText || '').trim().toLowerCase();
                    const tokens = normalized.length ? normalized.split(/\\s+/).filter(Boolean) : [];
                    const filtered = papersData.filter(p => {{
                        if (tokens.length === 0) return true;
                        const titleStr = (typeof p.title === 'string' ? p.title : String(p.title || '')).toLowerCase();
                        const authorsStr = (Array.isArray(p.authors) ? p.authors.join(', ') : String(p.authors || '')).toLowerCase();
                        return tokens.every(t => titleStr.includes(t) || authorsStr.includes(t));
                    }});
                    filtered.forEach(p => {{
                        const wrapper = document.createElement('label');
                        wrapper.className = 'paper-item';
                        const cb = document.createElement('input');
                        cb.type = 'checkbox';
                        cb.checked = selectedPaperIds.has(p.id);
                        cb.addEventListener('change', () => {{
                            if (cb.checked) selectedPaperIds.add(p.id); else selectedPaperIds.delete(p.id);
                            applyFilter();
                        }});
                        const span = document.createElement('span');
                        // 格式化论文标题：使用标准的 Title Case 格式
                        const formatTitle = (title) => {{
                            if (!title) return '无标题';
                            
                            // 定义应该小写的词（介词、冠词、连词等）
                            const lowercaseWords = new Set([
                                'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'in', 'of', 'on', 'or', 'the', 'to', 'up', 'vs', 'vs.'
                            ]);
                            
                            return title.split(' ').map((word, index) => {{
                                if (word.length === 0) return word;
                                
                                // 第一个词和最后一个词总是大写
                                if (index === 0 || index === title.split(' ').length - 1) {{
                                    return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
                                }}
                                
                                // 检查是否应该小写
                                const cleanWord = word.replace(/[^a-zA-Z]/g, '').toLowerCase();
                                if (lowercaseWords.has(cleanWord)) {{
                                    return word.toLowerCase();
                                }}
                                
                                // 其他词首字母大写
                                return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
                            }}).join(' ');
                        }};
                        
                        const authorsText = Array.isArray(p.authors) ? p.authors.join(', ') : (p.authors || '');
                        span.innerHTML = `${{formatTitle(p.title)}}<br><span style="opacity:.7;font-size:.85em">${{authorsText}} (${{p.year || 'N/A'}})</span>`;
                        wrapper.appendChild(cb);
                        wrapper.appendChild(span);
                        paperListEl.appendChild(wrapper);
                    }});
                }}
                renderPaperList();
                let paperSearchDebounce = null;
                searchEl.addEventListener('input', () => {{
                    clearTimeout(paperSearchDebounce);
                    paperSearchDebounce = setTimeout(() => renderPaperList(searchEl.value), 150);
                }});
                selectAllBtn.onclick = () => {{ selectedPaperIds = new Set(papersData.map(p => p.id)); renderPaperList(searchEl.value); applyFilter(); }};
                clearAllBtn.onclick = () => {{ selectedPaperIds = new Set(); renderPaperList(searchEl.value); applyFilter(); }};

                // Paper selection ∩ year filter helper
                function isSelectedAndWithinYear(paperId) {{
                    const p = papersData.find(x => x.id === paperId);
                    if (!p) return false;
                    const inYear = (!p.year) || (p.year <= currentYear());
                    return inYear && selectedPaperIds.has(paperId);
                }}

                function constructMatchesSelection(c) {{
                    if (!c.paper_ids || c.paper_ids.length === 0) return false;
                    return c.paper_ids.some(isSelectedAndWithinYear);
                }}

                function relationshipMatchesSelection(r) {{
                    if (!r.paper_ids || r.paper_ids.length === 0) return false;
                    
                    // 单一互斥过滤：方向/显著性/形状
                    const dir = (r.effect_direction || '').toLowerCase();
                    const shape = (r.non_linear_type || '').toLowerCase();
                    if (relAll && relAll.checked) {{
                        // no-op
                    }} else if (relPos && relPos.checked) {{
                        if (dir !== 'positive') return false;
                    }} else if (relNeg && relNeg.checked) {{
                        if (dir !== 'negative') return false;
                    }} else if (relInsig && relInsig.checked) {{
                        if (dir !== 'insignificant') return false;
                    }} else if (relS && relS.checked) {{
                        if (!(shape === 's' || shape === 's-shaped' || shape === 's_shape' || shape === 's-shaped')) return false;
                    }} else if (relU && relU.checked) {{
                        if (!(shape === 'u' || shape === 'u-shape' || shape === 'u-shaped')) return false;
                    }} else if (relInvU && relInvU.checked) {{
                        if (!(shape === 'inverted_u' || shape === 'inverted-u' || shape === 'inverted u' || shape === 'inverted_u-shaped' || shape === 'inverted_u-shaped')) return false;
                    }}

                    // Evidence strength filters - 已移除
                    if (false) {{
                        // 检查relationship_instances中的统计信息
                        const hasValidStats = (r.relationship_instances || []).some(ri => {{
                            let stats = null;
                            try {{
                                stats = ri.statistical_details ? JSON.parse(ri.statistical_details) : null;
                            }} catch(e) {{
                                stats = ri.statistical_details;
                            }}
                            
                            if (!stats) return false;
                            
                            const pVal = stats.p_value;
                            const corr = stats.correlation || stats.beta_coefficient;
                            const n = stats.sample_size || stats.n;
                            
                            // 证据强度筛选已禁用
                            
                            return true;
                        }});
                        
                        if (!hasValidStats) return false;
                    }}
                    
                    // Require at least one supporting paper that is selected and within year
                    return r.paper_ids.some(isSelectedAndWithinYear);
                }}

                // --- Relationship symbol labeling ---
                function normalizeStatus(status) {{
                    const s = (status || '').toLowerCase();
                    if (s.includes('empirical')) return 'empirical';
                    if (s.includes('hypoth')) return 'hypothesis';
                    if (s.includes('propos')) return 'proposition';
                    return 'unknown';
                }}

                function symbolForInstance(ri) {{
                    const nonlin = (ri.non_linear_type || '').toLowerCase();
                    if (nonlin.includes('inverted')) return '∩';
                    if (nonlin.includes('u')) return '∪';
                    if (nonlin.includes('s')) return 'S';
                    const dir = (ri.effect_direction || '').toLowerCase();
                    if (dir === 'positive') return '+';
                    if (dir === 'negative') return '−';
                    return '·'; // unknown/insignificant
                }}

                function computeEdgeLabelForRel(rel) {{
                    const instances = (rel.relationship_instances || []).filter(ri => isSelectedAndWithinYear(ri.paper_uid));
                    if (instances.length === 0) return '';
                    const buckets = {{ empirical: new Set(), hypothesis: new Set(), proposition: new Set(), unknown: new Set() }};
                    instances.forEach(ri => {{ buckets[normalizeStatus(ri.status)].add(symbolForInstance(ri)); }});
                    // If only one status bucket used and size==1 => single concise symbol
                    const used = Object.entries(buckets).filter(([k,v]) => v.size > 0);
                    if (used.length === 1 && used[0][1].size === 1) {{
                        return Array.from(used[0][1])[0];
                    }}
                    // Compose grouped label
                    const seg = [];
                    if (buckets.empirical.size) seg.push('E:' + Array.from(buckets.empirical).join(''));
                    if (buckets.hypothesis.size) seg.push('H:' + Array.from(buckets.hypothesis).join(''));
                    if (buckets.proposition.size) seg.push('P:' + Array.from(buckets.proposition).join(''));
                    if (seg.length === 0 && buckets.unknown.size) seg.push(Array.from(buckets.unknown).join(''));
                    return seg.join(' | ');
                }}

                function evidenceCountWithinYear(pids) {{
                    if (!pids || !pids.length) return 0;
                    const uniq = new Set();
                    pids.forEach(pid => {{ if (isSelectedAndWithinYear(pid)) uniq.add(pid); }});
                    return uniq.size;
                }}

                // Toast notification system
                function showToast(message, type = 'info', duration = 3000) {{
                    const container = document.getElementById('toast-container');
                    const toast = document.createElement('div');
                    toast.className = `toast ${{type}}`;
                    toast.textContent = message;
                    toast.onclick = () => toast.remove();
                    
                    container.appendChild(toast);
                    setTimeout(() => toast.remove(), duration);
                }}
                
                // Update HUD display
                function updateHUD() {{
                    const papersCount = document.getElementById('papers-count');
                    const nodesCount = document.getElementById('nodes-count');
                    const edgesCount = document.getElementById('edges-count');
                    
                    if (papersCount) papersCount.textContent = selectedPaperIds.size;
                    if (nodesCount) nodesCount.textContent = nodes.length;
                    if (edgesCount) edgesCount.textContent = edges.length;
                    
                    // Update preset info in HUD
                    const presetNames = {{
                        'overview': '总览',
                        'causal': '因果',
                        'correlation': '相关',
                        'dense': '密集'
                    }};
                    const selectionSummary = document.getElementById('selection-summary');
                    if (selectionSummary) {{
                        const presetInfo = selectionSummary.querySelector('.preset-info');
                        if (presetInfo) {{
                            presetInfo.textContent = `视图: ${{presetNames[currentPreset] || '总览'}}`;
                        }} else {{
                            const div = document.createElement('div');
                            div.className = 'preset-info';
                            div.textContent = `视图: ${{presetNames[currentPreset] || '总览'}}`;
                            selectionSummary.appendChild(div);
                        }}
                    }}
                }}

                function applyFilter() {{
                    nodes.clear();
                    edges.clear();

                    // Step 1: Relationships that pass filters
                    const candidateRelationships = relationshipsData.filter(relationshipMatchesSelection);
                    const connectedIds = new Set();
                    candidateRelationships.forEach(r => {{
                        connectedIds.add(r.source_construct);
                        connectedIds.add(r.target_construct);
                        // Also include moderators and mediators from relationship instances to ensure they're visible
                        if (r.relationship_instances) {{
                            r.relationship_instances.forEach(ri => {{
                                if (ri.moderators) {{
                                    ri.moderators.forEach(m => connectedIds.add(m));
                                }}
                                if (ri.mediators) {{
                                    ri.mediators.forEach(m => connectedIds.add(m));
                                }}
                            }});
                        }}
                    }});

                    // Step 2: Only add constructs that are selected AND connected
                    const filteredConstructs = constructsData
                        .filter(constructMatchesSelection)
                        .filter(c => connectedIds.has(c.name));
                    filteredConstructs.forEach((construct, index) => {{
                        const evid = evidenceCountWithinYear(construct.paper_ids);
                        const nodeSize = Math.max(10, 6 + Math.sqrt(evid) * 4);
                        nodes.add({{
                            id: construct.name,
                            label: '',
                            color: {{ background: '#e5e7eb', border: '#c9d1d9', highlight: {{ background: '#f5f6f8', border: '#c9d1d9' }} }},
                            font: {{ color: '#2c3e50', size: 12, face: 'Times New Roman', bold: false }},
                            size: nodeSize,
                            shape: 'dot',
                            shadow: {{ enabled: true, color: 'rgba(0,0,0,0.15)', size: 6, x: 3, y: 3 }}
                        }});
                    }});

                    // Step 3: Render edges (non-reified mode only)
                    const filteredRelationships = candidateRelationships
                        .filter(r => nodes.get(r.source_construct) && nodes.get(r.target_construct));
                    filteredRelationships.forEach(rel => {{
                        const edgeWidth = Math.max(1.5, 1 + Math.sqrt(evidenceCountWithinYear(rel.paper_ids) || 0));
                            const edgeLabel = rel.status === 'Hypothesized' ? 'H' : 
                                             (rel.is_validated_causality ? 'C' : 'E');
                            edges.add({{
                                from: rel.source_construct,
                                to: rel.target_construct,
                                label: edgeLabel,
                                color: {{ color: getBlueprintRelationshipColor(rel), highlight: '#ecf0f1', hover: '#ecf0f1' }},
                                font: {{ color: '#2c3e50', size: 12, face: 'Times New Roman' }},
                                width: edgeWidth,
                                arrows: {{ to: {{ enabled: true, scaleFactor: 0.8 }} }},
                                shadow: {{ enabled: true, color: 'rgba(0,0,0,0.3)', size: 5, x: 3, y: 3 }}
                            }});

                            // Also visualize moderators in non-reified mode by linking them to both constructs
                            const visInstances = (rel.relationship_instances || []).filter(ri => isSelectedAndWithinYear(ri.paper_uid));
                            const moderatorSet = new Set();
                            visInstances.forEach(ri => {{
                                (ri.moderators || []).forEach(m => moderatorSet.add(m));
                            }});
                            moderatorSet.forEach(m => {{
                                if (nodes.get(m)) {{
                                    // Create moderator edges with special properties for highlighting
                                    const sourceEdge = {{ 
                                        from: m, 
                                        to: rel.source_construct, 
                                        dashes: true, 
                                        color: {{ color: '#6b7280' }}, 
                                        width: 1,
                                        // Store moderator relationship info for highlighting
                                        moderatorInfo: {{
                                            moderator: m,
                                            source: rel.source_construct,
                                            target: rel.target_construct,
                                            relationship: rel
                                        }}
                                    }};
                                    const targetEdge = {{ 
                                        from: m, 
                                        to: rel.target_construct, 
                                        dashes: true, 
                                        color: {{ color: '#6b7280' }}, 
                                        width: 1,
                                        // Store moderator relationship info for highlighting
                                        moderatorInfo: {{
                                            moderator: m,
                                            source: rel.source_construct,
                                            target: rel.target_construct,
                                            relationship: rel
                                        }}
                                    }};
                                    edges.add(sourceEdge);
                                    edges.add(targetEdge);
                                }}
                            }});
                        }});
                    

                    // Filter out isolated nodes (nodes with no connections)
                    const connectedNodeIds = new Set();
                    edges.forEach(edge => {{
                        connectedNodeIds.add(edge.from);
                        connectedNodeIds.add(edge.to);
                    }});
                    
                    // Remove isolated nodes
                    const isolatedNodes = nodes.getIds().filter(id => !connectedNodeIds.has(id));
                    if (isolatedNodes.length > 0) {{
                        console.log('Removing isolated nodes:', isolatedNodes);
                        isolatedNodes.forEach(id => nodes.remove(id));
                    }}
                    
                    if (network) {{
                        const view = loadViewState() || {{ scale: 1, position: {{ x: 0, y: 0 }} }};
                        network.setData({{ nodes, edges }});
                        network.moveTo(view);

                        // render external labels under nodes
                        const container = document.getElementById('network-container');
                        if (!window.__labelLayer) {{
                            window.__labelLayer = document.createElement('div');
                            window.__labelLayer.style.position = 'absolute';
                            window.__labelLayer.style.left = '0';
                            window.__labelLayer.style.top = '0';
                            window.__labelLayer.style.pointerEvents = 'none';
                            container.appendChild(window.__labelLayer);
                        }}
                        function drawLabels() {{
                            if (!window.__labelLayer) return;
                            window.__labelLayer.innerHTML = '';
                            nodes.forEach(n => {{
                                const pos = network.canvasToDOM(network.getPositions([n.id])[n.id]);
                                const el = document.createElement('div');
                                el.style.position = 'absolute';
                                el.style.transform = 'translate(' + pos.x + 'px, ' + (pos.y + (n.size || 10) + 6) + 'px)';
                                el.style.color = '#ecf0f1';
                                el.style.font = '12px Times New Roman';
                                el.style.whiteSpace = 'nowrap';
                                el.textContent = n.id;
                                window.__labelLayer.appendChild(el);
                            }});
                        }}
                        drawLabels();
                        if (!window.__afterDrawingHandlerSet) {{
                            network.off('afterDrawing', window.__afterDrawingLabels);
                            window.__afterDrawingLabels = drawLabels;
                            network.on('afterDrawing', window.__afterDrawingLabels);
                            window.__afterDrawingHandlerSet = true;
                        }}
                    }}
                    console.log('筛选后: 节点', nodes.length, ' 边', edges.length);
                    
                    updateHUD();
                }}
                
                // Network options
                const options = {{
                    physics: {{
                        enabled: false,
                        barnesHut: {{
                            gravitationalConstant: -2000,
                            centralGravity: 0.3,
                            springLength: 150,
                            springConstant: 0.04,
                            damping: 0.09
                        }},
                        stabilization: {{
                            enabled: false,
                            iterations: 100,
                            updateInterval: 25
                        }}
                    }},
                    interaction: {{
                        hover: true,
                        tooltipDelay: 200,
                        zoomView: true,
                        dragView: true
                    }},
                    nodes: {{
                        borderWidth: 2,
                        shadow: true
                    }},
                    edges: {{
                        smooth: false,
                        shadow: true,
                        color: {{ inherit: false }}
                    }},
                    layout: {{
                        improvedLayout: true,
                        hierarchical: false,
                        randomSeed: 1337
                    }}
                }};
                console.log('网络选项设置完成:', options);
                
                // Create network
                console.log('开始创建vis.Network...');
                const network = new vis.Network(container, {{ nodes, edges }}, options);
                window.network = network;
                console.log('网络创建成功:', network);
                // Persist positions when user drags nodes in the full graph
                network.on('dragEnd', function() {{
                    try {{
                        const ids = nodes.getIds();
                        persistPositions(ids);
                    }} catch(e) {{}}
                }});
                // Persist and restore view (center/zoom) to keep absolute positions stable across filters
                const VIEW_KEY = 'kg_saved_view';
                const VIEW_USER_KEY = 'kg_view_user_set';
                // 默认锁定视图，避免切换筛选/年份引起绝对位置偏移
                window.__kg_viewLock = true;
                function saveViewState() {{
                    if (window.__kg_viewLock) return;
                    try {{
                        const scale = network.getScale();
                        const position = network.getViewPosition();
                        localStorage.setItem(VIEW_KEY, JSON.stringify({{ scale, position }}));
                        localStorage.setItem(VIEW_USER_KEY, 'true');
                    }} catch(e) {{}}
                }}
                function loadViewState() {{
                    try {{ const s = localStorage.getItem(VIEW_KEY); if (!s) return null; const v = JSON.parse(s); return {{ scale: v.scale, position: v.position }}; }} catch(e) {{ return null; }}
                }}
                function hasUserView() {{
                    try {{ return localStorage.getItem(VIEW_USER_KEY) === 'true'; }} catch(e) {{ return false; }}
                }}
                const initialView = loadViewState() || {{ scale: 1, position: {{ x: 0, y: 0 }} }};
                // 仅当用户曾经调整过视图才恢复
                if (hasUserView() && initialView) {{ network.moveTo(initialView); }}
                network.on('zoom', saveViewState);
                network.on('dragEnd', params => {{ if (!params || !params.nodes || params.nodes.length === 0) saveViewState(); }});
                
                // Tooltip handlers
                network.on('hoverNode', function(params) {{
                    showNodeTooltip(nodes.get(params.node), params.event);
                }});
                
                network.on('blurNode', function(params) {{
                    hideTooltip();
                }});
                
                network.on('hoverEdge', function(params) {{
                    const edge = edges.get(params.edge);
                    if (!edge) {{ return; }}
                    if (edge.id && edge.id.startsWith('similar_')) {{
                        // Show similarity tooltip for similarity edges
                        showSimilarityTooltip(edge, params.event);
                    }} else {{
                        // Show regular edge tooltip for relationship edges
                        showEdgeTooltip(edge, params.event);
                    }}
                }});
                
                network.on('blurEdge', function(params) {{
                    hideTooltip();
                }});
                
                // --- Deterministic layout support ---
                // Persist node positions across re-renders (in-memory + localStorage)
                const savedPositions = (() => {{
                    try {{
                        return JSON.parse(localStorage.getItem('kg_saved_positions') || '{{}}');
                    }} catch(e) {{ return {{}}; }}
                }})();

                function persistPositions(ids) {{
                    const pos = network.getPositions(ids);
                    Object.keys(pos).forEach(id => {{ savedPositions[id] = pos[id]; }});
                    try {{ localStorage.setItem('kg_saved_positions', JSON.stringify(savedPositions)); }} catch(e) {{}}
                }}

                function hashCode(str) {{
                    let h = 0; for (let i = 0; i < str.length; i++) {{ h = ((h << 5) - h) + str.charCodeAt(i); h |= 0; }}
                    return h;
                }}

                function deterministicFallbackPosition(id) {{
                    const h = Math.abs(hashCode(String(id)));
                    const angle = (h % 360) / 360 * Math.PI * 2;
                    const ring = (Math.floor(h / 360) % 5) + 1; // 1..5 个同心环
                    const radius = ring * 320;
                    return {{ x: Math.cos(angle) * radius, y: Math.sin(angle) * radius }};
                }}

                function getNodePosition(id) {{
                    return savedPositions[id] || deterministicFallbackPosition(id);
                }}

                function performInitialLayout() {{
                    // Build a full-graph once to compute deterministic coordinates
                    const allNodes = new vis.DataSet();
                    const allEdges = new vis.DataSet();
                    // Sort to keep input order stable
                    const constructsSorted = [...constructsData].sort((a,b) => (a.name||'').localeCompare(b.name||''));
                    const relsSorted = [...relationshipsData].sort((a,b) => (a.source_construct+a.target_construct).localeCompare(b.source_construct+b.target_construct));

                    constructsSorted.forEach(c => {{
                        // If we already have a saved position, place it there; allow user to drag later
                        const pos = savedPositions[c.name];
                        if (pos) {{
                            allNodes.add({{ id: c.name, label: c.name, x: pos.x, y: pos.y }});
                        }} else {{
                            allNodes.add({{ id: c.name, label: c.name }});
                        }}
                    }});
                    relsSorted.forEach(r => allEdges.add({{ from: r.source_construct, to: r.target_construct }}));

                    // Deterministic seed so physics is repeatable
                    network.setOptions({{ physics: {{ enabled: true }}, layout: {{ randomSeed: 1337 }} }});
                    network.setData({{ nodes: allNodes, edges: allEdges }});
                    network.once('stabilized', () => {{
                        // Save positions for all nodes and freeze physics afterwards
                        persistPositions(allNodes.getIds());
                        network.setOptions({{ physics: false }});
                        applyFilter();
                    }});
                }}
                
                // Keep tooltip near cursor
                document.addEventListener('mousemove', function(e) {{
                    if (tooltip && tooltip.classList.contains('show')) {{
                        tooltip.style.left = e.pageX + 15 + 'px';
                        tooltip.style.top = e.pageY - 15 + 'px';
                    }}
                }});

                // Timeline controls
                const rangeEl = document.getElementById('year-range');
                const yearLabel = document.getElementById('year-label');
                rangeEl.min = minYear; rangeEl.max = maxYear; rangeEl.value = maxYear;

                function currentYear() {{ return parseInt(rangeEl.value, 10); }}

                function withinYear(paperId) {{
                    const p = papersData.find(x => x.id === paperId);
                    if (!p || !p.year) return true;
                    return p.year <= currentYear();
                }}

                // Current preset view
                let currentPreset = 'overview';
                
                function applyFilter() {{
                    nodes.clear(); edges.clear();
                    
                    // Step 1: relationships that are within timeline and match filters
                    let candidateRelationships = relationshipsData
                        .map(r => Object.assign({{}}, r, {{ paper_ids: (r.paper_ids || []).filter(withinYear) }}))
                        .filter(relationshipMatchesSelection);
                    
                    // Apply preset view filtering
                    if (currentPreset === 'causal') {{
                        // Show relationships that have at least one SELECTED+WITHIN-YEAR instance validated as causal
                        candidateRelationships = candidateRelationships.filter(r => {{
                            const vis = (r.relationship_instances || []).filter(ri => isSelectedAndWithinYear(ri.paper_uid));
                            return vis.length > 0 && vis.some(ri => ri.is_validated_causality === true);
                        }});
                    }} else if (currentPreset === 'correlation') {{
                        // Show relationships that have at least one SELECTED+WITHIN-YEAR instance that is NOT validated causal
                        candidateRelationships = candidateRelationships.filter(r => {{
                            const vis = (r.relationship_instances || []).filter(ri => isSelectedAndWithinYear(ri.paper_uid));
                            return vis.length > 0 && vis.some(ri => ri.is_validated_causality !== true);
                        }});
                    }} else if (currentPreset === 'dense') {{
                        // Show only the most connected constructs (top 50% by connection count)
                        const connectionCounts = new Map();
                        candidateRelationships.forEach(r => {{
                            connectionCounts.set(r.source_construct, (connectionCounts.get(r.source_construct) || 0) + 1);
                            connectionCounts.set(r.target_construct, (connectionCounts.get(r.target_construct) || 0) + 1);
                        }});
                        const sortedConstructs = Array.from(connectionCounts.entries())
                            .sort((a, b) => b[1] - a[1]);
                        const topCount = Math.ceil(sortedConstructs.length * 0.5);
                        const topConstructs = new Set(sortedConstructs.slice(0, topCount).map(([name]) => name));
                        candidateRelationships = candidateRelationships.filter(r => 
                            topConstructs.has(r.source_construct) && topConstructs.has(r.target_construct)
                        );
                    }}
                    
                    const connectedIds = new Set();
                    candidateRelationships.forEach(r => {{
                        connectedIds.add(r.source_construct);
                        connectedIds.add(r.target_construct);
                        // Include moderators/mediators from visible relationship instances so their nodes render
                        (r.relationship_instances || []).forEach(ri => {{
                            (ri.moderators || []).forEach(m => connectedIds.add(m));
                            (ri.mediators || []).forEach(m => connectedIds.add(m));
                        }});
                    }});
                    
                    // Step 2: only add constructs that are selected, within year, and connected
                    const filteredConstructs = constructsData
                        .filter(constructMatchesSelection)
                        .map(c => Object.assign({{}}, c, {{ paper_ids: (c.paper_ids || []).filter(withinYear) }}))
                        .filter(c => connectedIds.has(c.name));
                    
                    filteredConstructs.forEach((construct, index) => {{
                        // choose position by layout mode with validation
                        let posObj = null;
                        if (layoutMode === 'embedding' && embeddingPositions[construct.name]) {{
                            posObj = embeddingPositions[construct.name];
                        }} else if (centralityPositions[construct.name]) {{
                            posObj = centralityPositions[construct.name];
                        }} else {{
                            posObj = getNodePosition(construct.name);
                        }}
                        
                        // Validate coordinates to prevent layout issues
                        let pos = {{ x: 0, y: 0 }};
                        if (posObj && typeof posObj.x === 'number' && typeof posObj.y === 'number' && 
                            isFinite(posObj.x) && isFinite(posObj.y) && 
                            Math.abs(posObj.x) < 10000 && Math.abs(posObj.y) < 10000) {{
                            pos = {{ x: posObj.x, y: posObj.y }};
                        }} else {{
                            // Use fallback position if coordinates are invalid
                            pos = getNodePosition(construct.name);
                        }}
                        
                        const evid = evidenceCountWithinYear(construct.paper_ids);
                        const nodeSize = Math.max(10, 6 + Math.sqrt(evid) * 4);
                        nodes.add({{
                            id: construct.name,
                            label: '',
                            x: pos.x, y: pos.y,
                            color: {{ background: '#e5e7eb', border: '#c9d1d9', highlight: {{ background: '#f5f6f8', border: '#c9d1d9' }} }},
                            font: {{ color: '#2c3e50', size: 12, face: 'Times New Roman' }},
                            size: nodeSize,
                            shape: 'dot',
                            shadow: {{ enabled: true, color: 'rgba(0,0,0,0.25)', size: 8, x:4, y:4 }}
                        }});
                    }});
                    
                    // Step 3: add only relationships whose endpoints are visible
                    const filteredRelationships = candidateRelationships
                        .filter(r => nodes.get(r.source_construct) && nodes.get(r.target_construct));
                    
                    filteredRelationships.forEach(rel => {{
                        const edgeLabel = computeEdgeLabelForRel(rel) || '';
                        edges.add({{ from: rel.source_construct, to: rel.target_construct,
                                     label: edgeLabel,
                                     color: {{ color: getRelationshipColor(rel.type, rel.effect_direction), highlight: '#e5e7eb', hover: '#e5e7eb' }},
                                     font: {{ color: '#e5e7eb', size: 14, face: 'Times New Roman', bold: false, strokeWidth: 2, strokeColor: 'rgba(0,0,0,0.35)' }},
                                     width: 1.8, arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }},
                                     shadow: {{ enabled: false }} }});
                        // Also draw moderators in non-reified mode: dashed gray from moderator to both endpoints
                        const visInstances = (rel.relationship_instances || []).filter(ri => isSelectedAndWithinYear(ri.paper_uid));
                        const moderatorSet = new Set();
                        const mediatorSet = new Set();
                        visInstances.forEach(ri => {{ (ri.moderators || []).forEach(m => moderatorSet.add(m)); }});
                        visInstances.forEach(ri => {{ (ri.mediators || []).forEach(m => mediatorSet.add(m)); }});
                        moderatorSet.forEach(m => {{
                            if (nodes.get(m)) {{
                                const mi = {{
                                    moderator: m,
                                    source: rel.source_construct,
                                    target: rel.target_construct,
                                    relationship: rel
                                }};
                                edges.add({{ from: m, to: rel.source_construct, dashes: true, color: {{ color: '#6b7280' }}, width: 1, moderatorInfo: mi }});
                                edges.add({{ from: m, to: rel.target_construct, dashes: true, color: {{ color: '#6b7280' }}, width: 1, moderatorInfo: mi }});
                            }}
                        }});
                        // also draw mediators similarly with dotted dashes
                        mediatorSet.forEach(m => {{
                            if (nodes.get(m)) {{
                                const mi = {{ mediator: m, source: rel.source_construct, target: rel.target_construct, relationship: rel }};
                                edges.add({{ from: m, to: rel.source_construct, dashes: [2,6], color: {{ color: '#6b7280' }}, width: 1, mediatorInfo: mi }});
                                edges.add({{ from: m, to: rel.target_construct, dashes: [2,6], color: {{ color: '#6b7280' }}, width: 1, mediatorInfo: mi }});
                            }}
                        }});
                    }});
                    
                    // Step 4: Add similarity relationships between similar constructs (always visible)
                    filteredConstructs.forEach(construct => {{
                        if (construct.similar_constructs && construct.similar_constructs.length > 0) {{
                            construct.similar_constructs.forEach(similar => {{
                                // Only add similarity edges if both constructs are visible
                                if (nodes.get(similar.name)) {{
                                    // Create a unique edge ID for similarity relationships
                                    const similarityEdgeId = 'similar_' + construct.name + '_' + similar.name;
                                    
                                    // Check if we already added this similarity edge (avoid duplicates)
                                    if (!edges.get(similarityEdgeId)) {{
                                        // Get similarity score and confidence from the data
                                        const similarityData = similar;
                                        const similarityScore = similarityData.similarity_score || 0.85;
                                        const confidence = similarityData.llm_confidence || 0.9;
                                        
                                        edges.add({{
                                            id: similarityEdgeId,
                                            from: construct.name,
                                            to: similar.name,
                                            label: '~',
                                            color: {{ 
                                                color: '#9ca3af', // Gray color for similarity
                                                                highlight: '#9ca3af',
                hover: '#9ca3af'
                                            }},
                                            font: {{ 
                                                color: '#9ca3af', 
                                                size: 16, 
                                                face: 'Times New Roman', 
                                                bold: true,
                                                strokeWidth: 3, 
                                                strokeColor: 'rgba(139, 92, 246, 0.8)'
                                            }},
                                            width: 2,
                                            dashes: [8, 4], // Dashed line for similarity
                                            arrows: {{ to: {{ enabled: false }} }}, // No arrows for similarity
                                            shadow: {{ enabled: false }},
                                            smooth: {{ type: 'curvedCW', roundness: 0.3 }}, // Curved line
                                            // Store similarity metadata for tooltip
                                            similarity_score: similarityScore,
                                            llm_confidence: confidence,
                                            source_name: construct.name,
                                            target_name: similar.name
                                        }});
                                    }}
                                }}
                            }});
                        }}
                    }});
                    
                    if (network) {{
                        // 刷新数据后，若没有用户自定义视图，则请求自动适配
                        network.setData({{ nodes, edges }});
                        window.__fittedOnce = false;
                        if (hasUserView()) {{
                            const v = loadViewState();
                            if (v) network.moveTo(v);
                            window.__requestAutoFit = false;
                        }} else {{
                            window.__requestAutoFit = true;
                        }}
                        const container = document.getElementById('network-container');
                        if (!window.__labelLayer) {{
                            window.__labelLayer = document.createElement('div');
                            window.__labelLayer.style.position = 'absolute';
                            window.__labelLayer.style.left = '0';
                            window.__labelLayer.style.top = '0';
                            window.__labelLayer.style.pointerEvents = 'none';
                            container.appendChild(window.__labelLayer);
                        }}
                        function drawLabels2() {{
                            if (!window.__labelLayer) return;
                            window.__labelLayer.innerHTML = '';
                            const scale = network.getScale();
                            nodes.forEach(n => {{
                                const pos = network.canvasToDOM(network.getPositions([n.id])[n.id]);
                                const el = document.createElement('div');
                                el.style.position = 'absolute';
                                // node radius in screen space under current zoom
                                const nodeRadius = (n.size || 10) * (scale || 1);
                                el.style.left = pos.x + 'px';
                                const vOffset = nodeRadius + Math.max(6, 2 * scale);
                                el.style.top = (pos.y + vOffset) + 'px';
                                el.style.transform = 'translate(-50%, 0)';
                                // fade with zoom, hide when too small
                                const baseOpacity = Math.min(1, Math.max(0, (scale - 0.25) / 0.6));
                                // apply highlight-based fading for non-selected nodes
                                const isDim = window.__highlightNodes && !window.__highlightNodes.has(n.id);
                                const lowAlpha = (typeof window.__lowAlpha === 'number') ? window.__lowAlpha : 0.1;
                                const finalOpacity = isDim ? Math.min(baseOpacity, lowAlpha) : baseOpacity;
                                if (finalOpacity <= 0.02) return;
                                el.style.opacity = String(finalOpacity);
                                el.style.color = '#e5e7eb';
                                // font scales with node size and zoom
                                const fontPx = Math.max(10, Math.min(42, 10 + nodeRadius * 0.35));
                                el.style.font = fontPx + 'px Times New Roman';
                                el.style.whiteSpace = 'nowrap';
                                el.textContent = n.id;
                                window.__labelLayer.appendChild(el);
                            }});
                        }}
                        drawLabels2();
                        if (!window.__afterDrawingHandlerSet) {{
                            network.off('afterDrawing', window.__afterDrawingLabels);
                            window.__afterDrawingLabels = drawLabels2;
                            network.on('afterDrawing', window.__afterDrawingLabels);
                            window.__afterDrawingHandlerSet = true;
                        }}
                        // Auto-fit viewport to include all nodes and labels with padding on first render or when requested
                        if (!window.__fittedOnce || window.__requestAutoFit) {{
                            fitToViewport(160);
                            window.__fittedOnce = true;
                            window.__requestAutoFit = false;
                        }}
                        // Capture base styles after data render - only once
                        if (!window.__baseStylesCaptured) {{
                            initializeBaseStyles();
                            window.__baseStylesCaptured = true;
                        }}
                    }}
                    yearLabel.textContent = `年份: ${{currentYear()}}`;
                    
                    // Update HUD with preset info
                    updateHUD();
                }}

                rangeEl.addEventListener('input', applyFilter);

                // Smart search functionality
                function performSearch(query) {{
                    if (!query) {{
                        searchResults.style.display = 'none';
                        return;
                    }}
                    const q = query.toLowerCase();
                    
                    // 只搜索当前画布上显示的构型节点（不包括证据节点）
                    const constructNodeIds = nodes.getIds().filter(id => {{
                        const node = nodes.get(id);
                        // 只包含构型节点（圆形），排除证据节点（菱形）
                        return !node.shape || node.shape !== 'diamond';
                    }});
                    const visibleMatches = constructNodeIds.filter(id => id.toLowerCase().includes(q)).map(id => ({{ id: id }}));
                    
                    // 更新全局搜索匹配结果
                    searchMatches = visibleMatches;
                    
                    if (visibleMatches.length === 0) {{
                        searchResults.style.display = 'none';
                        return;
                    }}
                    
                    searchResults.innerHTML = '';
                    visibleMatches.slice(0, 8).forEach((match, idx) => {{
                        const div = document.createElement('div');
                        div.className = 'search-result';
                        
                        const title = document.createElement('div');
                        title.className = 'search-result-title';
                        title.textContent = match.id;
                        
                        const meta = document.createElement('div');
                        meta.className = 'search-result-meta';
                        
                        // 获取该节点的构型信息
                        const construct = constructsData.find(c => c.name === match.id);
                        if (construct) {{
                            const paperCount = (construct.paper_ids || []).length;
                            const dimensionCount = (construct.dimensions || []).length;
                            meta.textContent = `${{paperCount}} 篇论文${{dimensionCount > 0 ? ` • ${{dimensionCount}} 个维度` : ''}}`;
                        }} else {{
                            meta.textContent = '构型节点';
                        }}
                        
                        div.appendChild(title);
                        div.appendChild(meta);
                        
                        div.onclick = () => {{
                            const nodeId = match.id;
                            
                            // 1. 选中并聚焦节点
                            network.selectNodes([nodeId]);
                            network.focus(nodeId, {{ scale: 1.5, animation: true }});
                            
                            // 2. 模拟点击事件来触发完整的高亮逻辑
                            setTimeout(() => {{
                                // 创建一个模拟的点击事件参数
                                const simulatedParams = {{
                                    nodes: [nodeId],
                                    edges: [],
                                    pointer: {{
                                        DOM: {{ x: 0, y: 0 }}
                                    }}
                                }};
                                
                                // 如果网络图的点击处理函数已定义，直接调用
                                if (window.networkClickHandler) {{
                                    window.networkClickHandler(simulatedParams);
                                }}
                            }}, 100);
                            
                            // 3. 隐藏搜索结果并更新搜索框
                            searchResults.style.display = 'none';
                            globalSearch.value = nodeId;
                        }};
                        
                        searchResults.appendChild(div);
                    }});
                    searchResults.style.display = 'block';
                }}
                
                globalSearch && globalSearch.addEventListener('input', (e) => {{
                    performSearch(e.target.value.trim());
                }});
                
                globalSearch && globalSearch.addEventListener('keydown', (e) => {{
                    if (e.key === 'Enter') {{
                        if (searchMatches.length > 0) {{
                            searchCurrentIndex = (searchCurrentIndex + 1) % searchMatches.length;
                            const match = searchMatches[searchCurrentIndex];
                            const nodeId = match.id;
                            
                            // 1. 选中并聚焦节点
                            network.selectNodes([nodeId]);
                            network.focus(nodeId, {{ scale: 1.5, animation: true }});
                            
                            // 2. 模拟点击事件来触发完整的高亮逻辑
                            setTimeout(() => {{
                                // 创建一个模拟的点击事件参数
                                const simulatedParams = {{
                                    nodes: [nodeId],
                                    edges: [],
                                    pointer: {{
                                        DOM: {{ x: 0, y: 0 }}
                                    }}
                                }};
                                
                                // 如果网络图的点击处理函数已定义，直接调用
                                if (window.networkClickHandler) {{
                                    window.networkClickHandler(simulatedParams);
                                }}
                            }}, 100);
                            
                            // 3. 更新搜索框并隐藏结果
                            globalSearch.value = nodeId;
                            searchResults.style.display = 'none';
                        }}
                    }} else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {{
                        e.preventDefault();
                        const results = searchResults.querySelectorAll('.search-result');
                        if (results.length === 0) return;
                        
                        results.forEach(r => r.classList.remove('selected'));
                        if (e.key === 'ArrowUp') {{
                            searchCurrentIndex = searchCurrentIndex <= 0 ? results.length - 1 : searchCurrentIndex - 1;
                        }} else {{
                            searchCurrentIndex = searchCurrentIndex >= results.length - 1 ? 0 : searchCurrentIndex + 1;
                        }}
                        results[searchCurrentIndex].classList.add('selected');
                    }} else if (e.key === 'Escape') {{
                        searchResults.style.display = 'none';
                        globalSearch.blur();
                    }}
                }});
                
                // 动态定位过滤器弹窗
                function positionFilterDropdowns() {{
                    document.querySelectorAll('.filter-dropdown').forEach(dropdown => {{
                        const summary = dropdown.querySelector('summary');
                        const content = dropdown.querySelector('.filter-content');
                        if (!summary || !content) return;
                        // 将弹窗内容挂载到body，避免被任何局部容器截断
                        if (content.parentElement !== document.body) {{
                            content.__origParent = dropdown;
                            document.body.appendChild(content);
                        }}
                        
                        const rect = summary.getBoundingClientRect();
                        content.style.position = 'fixed';
                        content.style.top = (rect.bottom + 5) + 'px';
                        content.style.left = rect.left + 'px';
                        content.style.zIndex = '99999';
                    }});
                }}

                function restoreDropdownContent(dropdown) {{
                    // 将内容移回到原来的details中
                    const inBody = Array.from(document.body.querySelectorAll('.filter-content')).find(c => c.__origParent === dropdown);
                    if (inBody) {{ dropdown.appendChild(inBody); }}
                }}

                function closeOtherDropdowns(current) {{
                    document.querySelectorAll('.filter-dropdown').forEach(d => {{
                        if (d !== current && d.open) {{
                            // 先恢复内容，再关闭
                            restoreDropdownContent(d);
                            d.open = false;
                        }}
                    }});
                }}
                
                // 监听过滤器弹窗的展开/收起 - 使用click事件而不是toggle事件来避免竞态条件
                document.querySelectorAll('.filter-dropdown summary').forEach(summary => {{
                    summary.addEventListener('click', (e) => {{
                        e.preventDefault(); // 阻止默认的toggle行为
                        const dropdown = summary.parentElement;
                        
                        if (dropdown.open) {{
                            // 当前是打开的，点击后关闭
                            dropdown.open = false;
                            restoreDropdownContent(dropdown);
                        }} else {{
                            // 当前是关闭的，点击后打开（先关闭其他的）
                            closeOtherDropdowns(dropdown);
                            dropdown.open = true;
                            setTimeout(positionFilterDropdowns, 10);
                        }}
                    }});
                }});
                
                // View presets
                document.querySelectorAll('.preset-btn').forEach(btn => {{
                    btn.addEventListener('click', () => {{
                        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
                        btn.classList.add('active');
                        
                        const preset = btn.dataset.preset;
                        currentPreset = preset; // Update current preset
                        
                        // Reset all filters first
                        [relAll, relPos, relNeg, relInsig, relS, relU, relInvU]
                            .forEach(el => el && (el.checked = false));
                        
                        switch(preset) {{
                            case 'overview':
                                (relAll && (relAll.checked = true));
                                break;
                            case 'causal':
                                (relAll && (relAll.checked = true));
                                break;
                            case 'correlation':
                                (relAll && (relAll.checked = true));
                                break;
                            case 'dense':
                                (relAll && (relAll.checked = true));
                                break;
                        }}
                        applyFilter();
                    }});
                }});
                


                // Layout mode toggles (single-source-of-truth with active class)
                const layoutCentralityBtn = document.getElementById('layout-centrality');
                const layoutEmbeddingBtn = document.getElementById('layout-embedding');
                function setLayoutMode(mode) {{
                    const old = layoutMode;
                    layoutMode = mode;
                    if (layoutCentralityBtn) layoutCentralityBtn.classList.toggle('active', mode === 'centrality');
                    if (layoutEmbeddingBtn) layoutEmbeddingBtn.classList.toggle('active', mode === 'embedding');
                    if (old !== mode) {{
                        const container = document.getElementById('toast-container');
                        if (container) {{
                            showToast(mode === 'centrality' ? '切换到中心性布局' : '切换到语义布局', 'info');
                        }}
                        // request auto-fit after layout change
                        window.__requestAutoFit = true;
                    }}
                    applyFilter();
                }}
                layoutCentralityBtn && (layoutCentralityBtn.onclick = () => setLayoutMode('centrality'));
                layoutEmbeddingBtn && (layoutEmbeddingBtn.onclick = () => setLayoutMode('embedding'));
                if (layoutCentralityBtn) layoutCentralityBtn.classList.toggle('active', layoutMode === 'centrality');
                if (layoutEmbeddingBtn) layoutEmbeddingBtn.classList.toggle('active', layoutMode === 'embedding');

                // 首次渲染：先全图布局一次保存坐标，再进入筛选渲染
                performInitialLayout();

                // Filter change handlers
                [relAll, relPos, relNeg, relInsig, relS, relU, relInvU]
                    .forEach(el => el && el.addEventListener('change', applyFilter));

                // Details panel on click
                const detailsEl = document.getElementById('details-panel');
                
                // 定义网络点击处理函数为全局函数，以便搜索结果可以调用
                window.networkClickHandler = function(params) {{
                    hideTooltip();

                    // Normalize: if a node/edge is under the pointer, prefer those over stale selection arrays
                    let nodeAtPointer = null;
                    let edgeAtPointer = null;
                    try {{
                        if (params && params.pointer && params.pointer.DOM) {{
                            nodeAtPointer = network.getNodeAt(params.pointer.DOM);
                            edgeAtPointer = network.getEdgeAt(params.pointer.DOM);
                        }}
                    }} catch(e) {{ nodeAtPointer = null; edgeAtPointer = null; }}

                    if (nodeAtPointer) {{
                        params.nodes = [nodeAtPointer];
                        params.edges = [];
                    }} else if (edgeAtPointer) {{
                        params.nodes = [];
                        params.edges = [edgeAtPointer];
                    }}

                    // Background click: reset view (robust: neither node nor edge under pointer)
                    if (!nodeAtPointer && !edgeAtPointer) {{
                        console.log('Background clicked - initiating full reset.');

                        // 1. Clear vis.js internal selection
                        try {{
                            if (network && typeof network.unselectAll === 'function') {{
                                network.unselectAll();
                            }}
                        }} catch (e) {{ console.error('unselectAll error:', e); }}

                        // 2. Run our comprehensive reset function immediately
                        clearHighlight();

                        // 3. Reset the details panel
                        detailsEl.innerHTML = '<div style="opacity:0.8">点击中间的节点或连线查看详细信息</div>';
                        
                        // 4. Final override: A delayed second reset to fight any race conditions.
                        setTimeout(clearHighlight, 50);

                        return;
                    }}

                    // Node click highlighting (node + its incident edges and neighbor nodes) — handle first if present
                    if (params.nodes && params.nodes.length > 0) {{
                        clearHighlight();
                        const nodeId = params.nodes[0];
                        
                        // Check if this is a moderator node (has moderator edges)
                        const moderatorEdges = edges.get().filter(e => 
                            e.moderatorInfo && e.moderatorInfo.moderator === nodeId
                        );
                        
                        if (moderatorEdges.length > 0) {{
                            // This is a moderator node - highlight the entire triangle
                            const moderatorInfo = moderatorEdges[0].moderatorInfo;
                            const triangleNodes = [
                                moderatorInfo.moderator,
                                moderatorInfo.source,
                                moderatorInfo.target
                            ];
                            const triangleEdges = [
                                // Find the main relationship edge
                                ...edges.get().filter(e => 
                                    e.from === moderatorInfo.source && e.to === moderatorInfo.target
                                ),
                                // Include the moderator edges
                                ...moderatorEdges
                            ];
                            
                            fadeAllExcept(nodes, edges, triangleNodes, triangleEdges.map(e => e.id), 0.1);
                        }} else {{
                            // If this node is an endpoint of a moderated relationship, highlight the full triad(s)
                            const modEdgesForEndpoint = edges.get().filter(e => e.moderatorInfo && (e.to === nodeId || e.from === nodeId));
                            const medEdgesForEndpoint = edges.get().filter(e => e.mediatorInfo && (e.to === nodeId || e.from === nodeId));
                            if (modEdgesForEndpoint.length > 0) {{
                                const nodesToHighlight = new Set([nodeId]);
                                const edgeIdsToHighlight = new Set();
                                modEdgesForEndpoint.forEach(me => {{
                                    const mi = me.moderatorInfo;
                                    nodesToHighlight.add(mi.moderator);
                                    nodesToHighlight.add(mi.source);
                                    nodesToHighlight.add(mi.target);
                                    // add both moderator dashed edges
                                    edges.get().forEach(e => {{
                                        if (e.moderatorInfo && e.moderatorInfo.moderator === mi.moderator && e.moderatorInfo.source === mi.source && e.moderatorInfo.target === mi.target) {{
                                            edgeIdsToHighlight.add(e.id);
                                        }}
                                    }});
                                    // add the main relationship edge
                                    edges.get().forEach(e => {{
                                        if (e.from === mi.source && e.to === mi.target) {{
                                            edgeIdsToHighlight.add(e.id);
                                        }}
                                    }});
                                }});
                                fadeAllExcept(nodes, edges, Array.from(nodesToHighlight), Array.from(edgeIdsToHighlight), 0.1);
                            }} else if (medEdgesForEndpoint.length > 0) {{
                                const nodesToHighlight = new Set([nodeId]);
                                const edgeIdsToHighlight = new Set();
                                medEdgesForEndpoint.forEach(me => {{
                                    const mi = me.mediatorInfo;
                                    nodesToHighlight.add(mi.mediator);
                                    nodesToHighlight.add(mi.source);
                                    nodesToHighlight.add(mi.target);
                                    // add both mediator dotted edges
                                    edges.get().forEach(e => {{
                                        if (e.mediatorInfo && e.mediatorInfo.mediator === mi.mediator && e.mediatorInfo.source === mi.source && e.mediatorInfo.target === mi.target) {{
                                            edgeIdsToHighlight.add(e.id);
                                        }}
                                    }});
                                    // add the main relationship edge
                                    edges.get().forEach(e => {{
                                        if (e.from === mi.source && e.to === mi.target) {{
                                            edgeIdsToHighlight.add(e.id);
                                        }}
                                    }});
                                }});
                                fadeAllExcept(nodes, edges, Array.from(nodesToHighlight), Array.from(edgeIdsToHighlight), 0.1);
                            }} else {{
                                // Regular node highlighting
                                const incident = (typeof network.getConnectedEdges === 'function') ? (network.getConnectedEdges(nodeId) || []) : [];
                                const neighborList = (typeof network.getConnectedNodes === 'function') ? (network.getConnectedNodes(nodeId) || []) : [];
                                const neighbors = new Set(neighborList);
                                neighbors.add(nodeId);
                                fadeAllExcept(nodes, edges, Array.from(neighbors), incident, 0.1);
                            }}
                        }}
                    }} else if (params.edges && params.edges.length > 0) {{
                        // Edge click highlighting
                        clearHighlight();
                        const edgeId = params.edges[0];
                        const e = edges.get(edgeId);
                        if (e) {{
                            // If this edge is part of a moderated or mediated relationship, highlight the whole triad(s)
                            if (e.moderatorInfo) {{
                                const mi = e.moderatorInfo;
                                const nodesToHighlight = [mi.moderator, mi.source, mi.target];
                                const edgeIds = [];
                                // include main relationship edge (either orientation)
                                edges.get().forEach(ed => {{ if ((ed.from === mi.source && ed.to === mi.target) || (ed.from === mi.target && ed.to === mi.source)) edgeIds.push(ed.id); }});
                                // include both moderator dashed edges
                                edges.get().forEach(ed => {{ if (ed.moderatorInfo && ed.moderatorInfo.moderator === mi.moderator && ed.moderatorInfo.source === mi.source && ed.moderatorInfo.target === mi.target) edgeIds.push(ed.id); }});
                                fadeAllExcept(nodes, edges, nodesToHighlight, edgeIds, 0.1);
                            }} else if (e.mediatorInfo) {{
                                const mi = e.mediatorInfo;
                                const nodesToHighlight = [mi.mediator, mi.source, mi.target];
                                const edgeIds = [];
                                edges.get().forEach(ed => {{ if ((ed.from === mi.source && ed.to === mi.target) || (ed.from === mi.target && ed.to === mi.source)) edgeIds.push(ed.id); }});
                                edges.get().forEach(ed => {{ if (ed.mediatorInfo && ed.mediatorInfo.mediator === mi.mediator && ed.mediatorInfo.source === mi.source && ed.mediatorInfo.target === mi.target) edgeIds.push(ed.id); }});
                                fadeAllExcept(nodes, edges, nodesToHighlight, edgeIds, 0.1);
                            }} else {{
                                // Check if this is the main relationship edge for which moderator edges exist
                                const relatedModerators = edges.get().filter(ed => ed.moderatorInfo && ((ed.moderatorInfo.source === e.from && ed.moderatorInfo.target === e.to) || (ed.moderatorInfo.source === e.to && ed.moderatorInfo.target === e.from)));
                                const relatedMediators = edges.get().filter(ed => ed.mediatorInfo && ((ed.mediatorInfo.source === e.from && ed.mediatorInfo.target === e.to) || (ed.mediatorInfo.source === e.to && ed.mediatorInfo.target === e.from)));
                                if (relatedModerators.length > 0 || relatedMediators.length > 0) {{
                                    const nodesToHighlight = new Set([e.from, e.to]);
                                    const edgeIds = new Set([edgeId]);
                                    relatedModerators.forEach(ed => {{ nodesToHighlight.add(ed.moderatorInfo.moderator); edgeIds.add(ed.id); }});
                                    relatedMediators.forEach(ed => {{ nodesToHighlight.add(ed.mediatorInfo.mediator); edgeIds.add(ed.id); }});
                                    fadeAllExcept(nodes, edges, Array.from(nodesToHighlight), Array.from(edgeIds), 0.1);
                                }} else {{
                                    const highlightNodes = [e.from, e.to].filter(Boolean);
                                    const highlightEdges = [edgeId];
                                    fadeAllExcept(nodes, edges, highlightNodes, highlightEdges, 0.1);
                                }}
                            }}
                        }}
                    }}

                    if (params.nodes && params.nodes.length > 0) {{
                        const nodeId = params.nodes[0];
                        const construct = constructsData.find(c => c.name === nodeId);
                        if (!construct) return;
                        let html = `<div class="detail-section"><strong style="font-size:1.1rem">${{construct.name}}</strong></div>`;
                        // Show abstract definition with parent constructs (combined in one section)
                        if (construct.best_description) {{
                            html += `<div class="detail-section no-border">
                                <strong>摘要定义:</strong><br>
                                <div style="margin-top:8px;">${{htmlWithMathSafe(construct.best_description)}}</div>`;
                            
                            // Add parent constructs to the same section if they exist
                            if (construct.parent_constructs && construct.parent_constructs.length > 0) {{
                                html += `<div style="margin-top:16px;">
                                    <strong>所属构型:</strong> ${{construct.parent_constructs.join(', ')}}
                                </div>`;
                            }}
                            
                            html += `</div>`;
                        }}
                        
                        // Show dimensions
                        if (construct.dimensions && construct.dimensions.length > 0) {{
                            html += `<div class="detail-section">
                                <strong>维度:</strong><br>
                                ${{construct.dimensions.map(function(dim) {{ return '• ' + dim; }}).join('<br>')}}
                            </div>`;
                        }}
                        
                        // Show similar constructs (combined)
                        const allSimilarConstructs = [];
                        if (construct.similar_constructs && construct.similar_constructs.length > 0) {{
                            construct.similar_constructs.forEach(s => {{
                                if (s.name) allSimilarConstructs.push(s.name);
                            }});
                        }}
                        if (construct.similar_to_constructs && construct.similar_to_constructs.length > 0) {{
                            construct.similar_to_constructs.forEach(s => {{
                                if (s.name) allSimilarConstructs.push(s.name);
                            }});
                        }}
                        
                        if (allSimilarConstructs.length > 0) {{
                            html += `<div class="detail-section no-border">
                                <strong>相似构型:</strong> ${{allSimilarConstructs.join(', ')}}
                            </div>`;
                        }}
                        
                        // Check if this is a moderator and show moderator information
                        const moderatorEdges = edges.get().filter(e => 
                            e.moderatorInfo && e.moderatorInfo.moderator === construct.name
                        );
                        
                        if (moderatorEdges.length > 0) {{
                            const moderatorInfo = moderatorEdges[0].moderatorInfo;
                            html += `<div style="border-top: 2px solid #6b7280; margin: 20px 0; padding-top: 20px;">
                                <div class="detail-section" style="border-left: 4px solid #6b7280; padding-left: 16px;">
                                    <strong style="color: #6b7280; font-size: 1.05rem;">调节变量信息</strong>
                                    <div style="margin-top: 12px; opacity: 0.9;">
                                        <div style="margin-bottom: 8px;"><strong>调节的关系:</strong> ${{moderatorInfo.source}} → ${{moderatorInfo.target}}</div>
                                        <div style="margin-bottom: 8px;"><strong>调节作用:</strong> 作为调节变量影响上述关系的强度和方向</div>
                                        <div style="margin-bottom: 8px;"><strong>关系状态:</strong> ${{moderatorInfo.relationship.status || 'N/A'}}</div>
                                        <div style="margin-bottom: 8px;"><strong>证据类型:</strong> ${{moderatorInfo.relationship.evidence_type || 'N/A'}}</div>
                                        <div style="margin-bottom: 8px;"><strong>效应方向:</strong> ${{moderatorInfo.relationship.effect_direction || 'N/A'}}</div>
                                    </div>
                                </div>
                            </div>`;
                        }}
                        
                        // Note: Do NOT add a separator here. We'll add a single separator
                        // later only if there is actual paper content and no moderator block.
                        
                        const byPaper = new Map();
                        (construct.definitions || []).forEach(d => {{ if (!d.paper_uid) return; if (!byPaper.has(d.paper_uid)) byPaper.set(d.paper_uid, {{ defs: [], meas: [] }}); byPaper.get(d.paper_uid).defs.push(d); }});
                        (construct.measurements || []).forEach(m => {{ if (!m.paper_uid) return; if (!byPaper.has(m.paper_uid)) byPaper.set(m.paper_uid, {{ defs: [], meas: [] }}); byPaper.get(m.paper_uid).meas.push(m); }});
                        
                        // Only render papers that have actual content
                        const papersWithContent = [];
                        papersData.forEach(p => {{
                            if (!selectedPaperIds.has(p.id)) return;
                            const entry = byPaper.get(p.id);
                            if (!entry) return;
                            
                            // Check if this paper has any definitions or measurements
                            const hasDefinitions = entry.defs && entry.defs.length > 0;
                            const hasMeasurements = entry.meas && entry.meas.length > 0;
                            
                            if (hasDefinitions || hasMeasurements) {{
                                papersWithContent.push({{ paper: p, entry: entry, hasDefinitions, hasMeasurements }});
                            }}
                        }});
                        
                        // Only add separator if there are papers with content and no moderator section
                        if (papersWithContent.length > 0 && moderatorEdges.length === 0) {{
                            html += `<div style="border-top: 1px solid rgba(255,255,255,0.2); margin: 20px 0; padding-top: 20px;"></div>`;
                        }}
                        
                        // Render papers with content
                        papersWithContent.forEach(({{ paper: p, entry, hasDefinitions, hasMeasurements }}) => {{
                            // Paper header without bullet point
                            html += `<div class="detail-section">
                                <div style="margin-bottom:16px;">
                                    <div style="opacity:.85; font-weight:bold;">${{formatTitle(p.title)}}</div>
                                    <div style="opacity:.7; margin-top:4px;">${{(p.authors || []).join(', ')}} (${{p.year || 'N/A'}})</div>
                                </div>`;
                            
                            // Definitions section
                            if (hasDefinitions) {{
                                const manyDefs = entry.defs.length > 1;
                                html += `<div style="margin-top:16px;">
                                    <strong>定义:</strong>`;
                                entry.defs.forEach(d => {{
                                    html += `<div style="margin-top:8px; display:flex; align-items:flex-start; gap:8px;">` +
                                            (manyDefs ? `<span style="color:#9ca3af; font-size:12px; margin-top:4px;">•</span>` : `<span style="width:0"></span>`) +
                                            `<div style="flex:1; word-wrap:break-word;">${{htmlWithMathSafe(d.definition || '')}}</div>` +
                                            `</div>`;
                                }});
                                html += `</div>`;
                            }}
                            
                            // Measurements section
                            if (hasMeasurements) {{ 
                                const manyMeas = entry.meas.length > 1;
                                html += `<div style="margin-top:16px;">
                                    <strong>测量:</strong>`;
                                entry.meas.forEach(m => {{ 
                                    let measHtml = `<div style="margin-top:8px; display:flex;align-items:flex-start; gap:8px;">` +
                                                   (manyMeas ? `<span style="color:#9ca3af; font-size:12px; margin-top:4px;">•</span>` : `<span style="width:0"></span>`) +
                                                   `<div style="flex:1; word-wrap:break-word;"><strong>${{m.name || ''}}</strong>`;
                                    if (m.description) {{
                                        measHtml += `: ${{htmlWithMathSafe(m.description)}}`;
                                    }}
                                    measHtml += `</div></div>`;
                                    html += measHtml;
                                }}); 
                                html += `</div>`; 
                            }}
                            
                            html += `</div>`;
                        }});
                        detailsEl.innerHTML = html || '<div style="opacity:.8">所选论文中暂无该构型的详细信息</div>';
                        
                        // Re-render MathJax after updating content (normalized inline)
                        if (window.MathJax) {{
                            MathJax.typesetPromise([detailsEl]).catch((err) => console.log('MathJax error:', err));
                        }}
                    }} else if (params.edges && params.edges.length > 0) {{
                        const edgeId = params.edges[0];
                        const e = edges.get(edgeId);
                        // Map any clicked edge (main/moderator/mediator) to the underlying main relationship
                        let rel = null;
                        if (e && e.moderatorInfo) {{
                            const mi = e.moderatorInfo;
                            rel = relationshipsData.find(r => (r.source_construct === mi.source && r.target_construct === mi.target) || (r.source_construct === mi.target && r.target_construct === mi.source));
                        }} else if (e && e.mediatorInfo) {{
                            const mi = e.mediatorInfo;
                            rel = relationshipsData.find(r => (r.source_construct === mi.source && r.target_construct === mi.target) || (r.source_construct === mi.target && r.target_construct === mi.source));
                        }} else if (e) {{
                            rel = relationshipsData.find(r => (r.source_construct === e.from && r.target_construct === e.to) || (r.source_construct === e.to && r.target_construct === e.from));
                        }}
                        if (!rel) return;
                        let html = `<div class="detail-section"><strong style="font-size:1.05rem">关系：${{rel.source_construct}} → ${{rel.target_construct}}</strong>
                            <div style="margin-top:12px;opacity:.8">
                                <div style="margin-bottom:8px;display:flex;align-items:center;gap:8px;"><span style="color:#9ca3af; font-size:12px;">•</span><strong>状态:</strong> ${{rel.status || 'N/A'}}</div>
                                <div style="margin-bottom:8px;display:flex;align-items:center;gap:8px;"><span style="color:#9ca3af; font-size:12px;">•</span><strong>证据类型:</strong> ${{rel.evidence_type || 'N/A'}}</div>
                                <div style="margin-bottom:8px;display:flex;align-items:center;gap:8px;"><span style="color:#9ca3af; font-size:12px;">•</span><strong>方向:</strong> ${{rel.effect_direction || 'N/A'}}</div>
                                <div style="margin-bottom:8px;display:flex;align-items:center;gap:8px;"><span style="color:#9ca3af; font-size:12px;">•</span><strong>因果验证:</strong> ${{(rel.relationship_instances || []).some(ri => ri.is_validated_causality === true) ? '是' : '否'}}</div>
                                <div style="margin-bottom:8px;display:flex;align-items:center;gap:8px;"><span style="color:#9ca3af; font-size:12px;">•</span><strong>元分析:</strong> ${{rel.is_meta_analysis ? '是' : '否'}}</div>
                            </div></div>`;
                        
                        // Group relationship instances by paper
                        const paperInstances = new Map();
                        (rel.relationship_instances || []).forEach(ri => {{
                            if (!ri.paper_uid) return;
                            if (!paperInstances.has(ri.paper_uid)) {{
                                paperInstances.set(ri.paper_uid, []);
                            }}
                            paperInstances.get(ri.paper_uid).push(ri);
                        }});
                        
                        papersData.forEach(p => {{
                            if (!selectedPaperIds.has(p.id)) return;
                            const instances = paperInstances.get(p.id);
                            if (!instances || instances.length === 0) return;
                            
                            // Paper header without bullet point
                            html += `<div class="detail-section">
                                <div style="margin-bottom:16px;">
                                    <div style="opacity:.85; font-weight:bold;">${{formatTitle(p.title)}}</div>
                                    <div style="opacity:.7; margin-top:4px;">${{(p.authors || []).join(', ')}} (${{p.year || 'N/A'}})</div>
                                </div>`;
                            
                            instances.forEach(ri => {{
                                html += `<div style="margin-top:12px;padding:12px;background:rgba(255,255,255,0.05);border-radius:6px;word-wrap:break-word;overflow-wrap:break-word;">`;
                                
                                if (ri.description) {{
                                    html += `<div style="margin-bottom:8px;word-wrap:break-word;"><strong>描述:</strong> ${{ri.description}}</div>`;
                                }}
                                
                                if (ri.context_snippet) {{
                                    html += `<div style="margin-top:8px;font-size:0.9em;opacity:0.8;font-style:italic;display:flex;align-items:flex-start;gap:8px;">
                                        <span style="color:#9ca3af; font-size:12px; margin-top:4px;">•</span>
                                        <div style="flex:1; word-wrap:break-word;">"${{ri.context_snippet}}"</div>
                                    </div>`;
                                }}
                                
                                // Statistical details
                                let stats = null;
                                try {{
                                    stats = ri.statistical_details ? JSON.parse(ri.statistical_details) : null;
                                }} catch(e) {{
                                    stats = ri.statistical_details;
                                }}
                                
                                if (stats && Object.keys(stats).length > 0) {{
                                    html += `<div style="margin-top:4px;word-wrap:break-word;"><strong>统计信息:</strong> `;
                                    const statItems = [];
                                    if (stats.p_value !== undefined) statItems.push(`P值: ${{stats.p_value}}`);
                                    if (stats.beta_coefficient !== undefined) statItems.push(`β: ${{stats.beta_coefficient}}`);
                                    if (stats.correlation !== undefined) statItems.push(`r: ${{stats.correlation}}`);
                                    html += statItems.join('，') || '无';
                                    html += `</div>`;
                                }}
                                
                                // Qualitative findings
                                if (ri.qualitative_finding) {{
                                    html += `<div style="margin-top:4px;word-wrap:break-word;"><strong>定性发现:</strong> ${{ri.qualitative_finding}}</div>`;
                                }}
                                

                                
                                // Boundary conditions
                                if (ri.boundary_conditions) {{
                                    html += `<div style="margin-top:4px;word-wrap:break-word;"><strong>边界条件:</strong> ${{ri.boundary_conditions}}</div>`;
                                }}
                                
                                // Replication outcome
                                if (ri.replication_outcome) {{
                                    html += `<div style="margin-top:4px;word-wrap:break-word;"><strong>复制结果:</strong> ${{ri.replication_outcome}}</div>`;
                                }}
                                
                                // Theories
                                if (ri.theories && ri.theories.length > 0) {{
                                    html += `<div style="margin-top:4px;word-wrap:break-word;"><strong>理论基础:</strong> ${{ri.theories.join(', ')}}</div>`;
                                }}
                                
                                // Moderators and Mediators
                                if (ri.moderators && ri.moderators.length > 0) {{
                                    html += `<div style="margin-top:4px;word-wrap:break-word;"><strong>调节变量:</strong> ${{ri.moderators.join(', ')}}</div>`;
                                }}
                                if (ri.mediators && ri.mediators.length > 0) {{
                                    html += `<div style="margin-top:4px;word-wrap:break-word;"><strong>中介变量:</strong> ${{ri.mediators.join(', ')}}</div>`;
                                }}
                                
                                html += `</div>`;
                            }});
                            
                            html += `</div>`;
                        }});
                        detailsEl.innerHTML = html || '<div style="opacity:.8">所选论文中暂无该关系的详细信息</div>';
                        
                        // Re-render MathJax after updating content
                        if (window.MathJax) {{
                            MathJax.typesetPromise([detailsEl]).catch((err) => console.log('MathJax error:', err));
                        }}
                    }}
                }};
                
                // 注册网络点击事件监听器
                network.on('click', window.networkClickHandler);
                
                console.log('统计信息更新完成');
                console.log('网络状态:', {{
                    nodes: network.body.data.nodes.length,
                    edges: network.body.data.edges.length,
                    container: container
                }});
                // Close DOMContentLoaded handler
            }});
            function getBlueprintRelationshipColor(rel) {{
                // Blueprint-based coloring: direction takes precedence, then causality
                if (rel.effect_direction === 'Positive') return edge_colors.positive;
                if (rel.effect_direction === 'Negative') return edge_colors.negative;
                if (rel.is_validated_causality === true) return edge_colors.causal;
                if (rel.status === 'Empirical_Result') return edge_colors.correlational;
                return edge_colors.default;
            }}
            
            // Legacy function for backward compatibility
            function getRelationshipColor(type, direction) {{
                if (direction === 'positive') return edge_colors.positive;
                if (direction === 'negative') return edge_colors.negative;
                if (type === 'causal') return edge_colors.causal;
                if (type === 'correlational') return edge_colors.correlational;
                return edge_colors.default;
            }}
            
            // Define color schemes in global scope
            const node_colors = [
                '#9ca3af', '#6b7280', '#4b5563', '#374151', '#1f2937',
                '#9ca3af', '#6b7280', '#4b5563', '#374151', '#1f2937',
                '#9ca3af', '#6b7280', '#4b5563', '#374151', '#1f2937'
            ]
            
            const edge_colors = {{
                positive: '#9ca3af',
                negative: '#9ca3af',
                causal: '#9ca3af',
                correlational: '#9ca3af',
                default: '#9ca3af'
            }};

            // Focus/highlight utilities
            function hexToRgb(hex) {{
                try {{
                    const h = hex.replace('#','').trim();
                    const bigint = parseInt(h.length === 3 ? h.split('').map(c=>c+c).join('') : h, 16);
                    const r = (bigint >> 16) & 255;
                    const g = (bigint >> 8) & 255;
                    const b = bigint & 255;
                    return [r,g,b];
                }} catch(e) {{ return [229,231,235]; }}
            }}
            function toRgba(color, alpha) {{
                if (!color) return 'rgba(229,231,235,' + alpha + ')';
                const c = String(color).trim();
                if (c.startsWith('rgba(')) {{
                    const parts = c.slice(5,-1).split(',').map(x=>x.trim());
                    return 'rgba(' + parts[0] + ', ' + parts[1] + ', ' + parts[2] + ', ' + alpha + ')';
                }}
                if (c.startsWith('rgb(')) {{
                    const parts = c.slice(4,-1).split(',').map(x=>x.trim());
                    return 'rgba(' + parts[0] + ', ' + parts[1] + ', ' + parts[2] + ', ' + alpha + ')';
                }}
                if (c.startsWith('#')) {{
                    const [r,g,b] = hexToRgb(c);
                    return 'rgba(' + r + ', ' + g + ', ' + b + ', ' + alpha + ')';
                }}
                return c; // fallback
            }}
            function ensureOpaque(color) {{
                if (!color) return '#e5e7eb';
                const c = String(color).trim();
                if (c.startsWith('rgba(')) {{
                    const parts = c.slice(5,-1).split(',').map(x=>x.trim());
                    return 'rgba(' + parts[0] + ', ' + parts[1] + ', ' + parts[2] + ', 1)';
                }}
                if (c.startsWith('rgb(') || c.startsWith('#')) return c;
                return color;
            }}
            function fadeAllExcept(nodes, edges, highlightNodeIds, highlightEdgeIds, lowAlpha=0.1) {{
                const hiNodes = new Set(highlightNodeIds || []);
                const hiEdges = new Set(highlightEdgeIds || []);
                // expose for label-layer rendering
                window.__highlightNodes = hiNodes;
                window.__highlightEdges = hiEdges;
                window.__lowAlpha = lowAlpha;
                // Nodes
                nodes.getIds().forEach(id => {{
                    const n = nodes.get(id);
                    if (!n) return;
                    const isHi = hiNodes.has(id);
                    if (isHi) {{
                        const b = window.__baseNodeStyles && window.__baseNodeStyles.get(id);
                        // force opaque bright style for highlighted node
                        const baseColor = (b && b.color) || (n.color || {{ background: '#e5e7eb', border: '#c9d1d9' }});
                        const bg = ensureOpaque(baseColor.background || baseColor.color || '#e5e7eb');
                        const bd = ensureOpaque(baseColor.border || baseColor.color || '#c9d1d9');
                        const fontStyle = (b && b.font) || n.font || {{ color: '#e5e7eb' }};
                        nodes.update({{ id, color: {{ background: bg, border: bd, highlight: {{ background: bg, border: bd }} }}, font: Object.assign({{}}, fontStyle, {{ color: '#e5e7eb' }}), size: (b && b.size) || n.size }});
                    }} else {{
                        const bg = toRgba(n.color && n.color.background || '#e5e7eb', lowAlpha);
                        const bd = toRgba(n.color && n.color.border || '#c9d1d9', lowAlpha);
                        const fcol = toRgba(n.font && n.font.color || '#e5e7eb', lowAlpha);
                        nodes.update({{ id, color: {{ background: bg, border: bd, highlight: {{ background: bg, border: bd }} }}, font: Object.assign({{}}, n.font||{{}}, {{ color: fcol }}) }});
                    }}
                }});
                // Edges
                edges.getIds().forEach(id => {{
                    const e = edges.get(id);
                    if (!e) return;
                    const isHi = hiEdges.has(id);
                    if (isHi) {{
                        // Restore to full visibility and original colors for highlighted edges
                        const b = window.__baseEdgeStyles && window.__baseEdgeStyles.get(id);
                        const baseColor = ensureOpaque(b ? b.color : ((e.color && (e.color.color || e.color)) || '#e5e7eb'));
                        const baseFont = b ? b.font : (e.font || {{ color: '#e5e7eb' }});
                        const width = Math.max((b ? b.width : (e.width || 1.8)), 3); // Make highlighted edges thicker
                        edges.update({{ id, color: {{ color: baseColor, highlight: baseColor, hover: baseColor }}, font: baseFont, width }});
                    }} else {{
                        // Fade non-highlighted edges
                        const baseEdgeColor = (e.color && (e.color.color || e.color)) || '#6b7280';
                        const faded = toRgba(baseEdgeColor, lowAlpha);
                        const baseFontColor = (e.font && e.font.color) || '#e5e7eb';
                        const fadedFont = toRgba(baseFontColor, lowAlpha);
                        edges.update({{ id, color: {{ color: faded, highlight: faded, hover: faded }}, font: Object.assign({{}}, e.font||{{}}, {{ color: fadedFont }}), width: Math.max(1, (e.width || 1.8)) }});
                    }}
                }});
            }}

            // Fit the view so that all nodes (plus label area) are visible with padding
            function fitToViewport(paddingPx) {{
                try {{
                    const ids = nodes.getIds();
                    if (!ids || ids.length === 0) return;
                    // First, let vis-network compute best fit for all nodes
                    network.fit({{ nodes: ids, animation: false }});
                    // Then slightly zoom out to reserve margin for external labels
                    const currentScale = network.getScale();
                    const currentPos = network.getViewPosition();
                    const marginFactor = 0.90; // keep 10% margin around
                    network.moveTo({{ position: currentPos, scale: currentScale * marginFactor, animation: false }});
                }} catch (e) {{}}
            }}

            // Focus/highlight utilities
            // Define fixed base styles that never change
            const FIXED_BASE_STYLES = {{
                node: {{
                    color: {{
                        background: '#e5e7eb',
                        border: '#c9d1d9',
                        highlight: {{
                            background: '#e5e7eb',
                            border: '#c9d1d9'
                        }}
                    }},
                    font: {{
                        color: '#e5e7eb',
                        size: 12,
                        face: 'arial'
                    }},
                    size: 20,
                    opacity: 1.0
                }},
                edge: {{
                    color: {{
                        color: '#6b7280',
                        highlight: '#6b7280',
                        hover: '#6b7280'
                    }},
                    font: {{
                        color: '#e5e7eb',
                        size: 12,
                        face: 'arial'
                    }},
                    width: 1.8,
                    opacity: 1.0
                }}
            }};

            // Initialize base styles (not needed anymore since we use fixed styles)
            function initializeBaseStyles() {{
                console.log('Using fixed base styles - no capture needed');
                // Just log for debugging
                console.log('Fixed base styles:', FIXED_BASE_STYLES);
            }}

            function resetStylesToBase() {{
                try {{
                    console.log('Resetting all styles to base...');
                    
                    // Reset all nodes to their base styles with full opacity
                    if (window.__baseNodeStyles) {{
                        nodes.getIds().forEach(id => {{
                            const s = window.__baseNodeStyles.get(id) || {{}};
                            const baseColor = s.color || {{ background: '#e5e7eb', border: '#c9d1d9' }};
                            const baseFont = s.font || {{ color: '#e5e7eb' }};
                            const baseSize = s.size || 20;
                            
                            // Ensure full opacity for all nodes
                            const bg = ensureOpaque(baseColor.background || baseColor.color || '#e5e7eb');
                            const bd = ensureOpaque(baseColor.border || baseColor.color || '#c9d1d9');
                            const fontColor = ensureOpaque(baseFont.color || '#e5e7eb');
                            
                            nodes.update({{ 
                                id, 
                                color: {{
                                    background: bg,
                                    border: bd,
                                    highlight: {{
                                        background: bg,
                                        border: bd
                                    }}
                                }}, 
                                font: Object.assign({{}}, baseFont, {{ color: fontColor }}), 
                                size: baseSize 
                            }});
                        }});
                    }}
                    
                    // Reset all edges to their base styles with full opacity
                    if (window.__baseEdgeStyles) {{
                        edges.getIds().forEach(id => {{
                            const s = window.__baseEdgeStyles.get(id) || {{}};
                            const baseColor = s.color || '#6b7280';
                            const baseFont = s.font || {{ color: '#e5e7eb' }};
                            const baseWidth = s.width || 1.8;
                            
                            // Ensure full opacity for all edges
                            const edgeColor = ensureOpaque(baseColor);
                            const fontColor = ensureOpaque(baseFont.color || '#e5e7eb');
                            
                            edges.update({{ 
                                id, 
                                color: {{ 
                                    color: edgeColor, 
                                    highlight: edgeColor, 
                                    hover: edgeColor 
                                }}, 
                                font: Object.assign({{}}, baseFont, {{ color: fontColor }}), 
                                width: baseWidth 
                            }});
                        }});
                    }}
                    
                    // Clear highlight tracking
                    window.__highlightNodes = null;
                    window.__highlightEdges = null;
                    window.__lowAlpha = null;
                    
                    console.log('All styles reset to base with full opacity');
                }} catch(e) {{
                    console.log('Error in resetStylesToBase:', e);
                }}
            }}

            // New function: completely reset all highlighting and restore original state
            function completelyResetAllHighlights() {{
                try {{
                    console.log('Completely resetting all highlights...');
                    
                    // First, clear all global highlight state
                    window.__highlightNodes = null;
                    window.__highlightEdges = null;
                    window.__lowAlpha = null;
                    
                    // Force all nodes back to full opacity and original styles
                    nodes.getIds().forEach(id => {{
                        const n = nodes.get(id);
                        if (!n) return;
                        
                        // Get the original base style
                        const baseStyle = window.__baseNodeStyles ? window.__baseNodeStyles.get(id) : null;
                        
                        if (baseStyle) {{
                            // Use base style with full opacity
                            const bg = ensureOpaque(baseStyle.color.background || baseStyle.color.color || '#e5e7eb');
                            const bd = ensureOpaque(baseStyle.color.border || baseStyle.color.color || '#c9d1d9');
                            const fontColor = ensureOpaque(baseStyle.font.color || '#e5e7eb');
                            
                            nodes.update({{ 
                                id, 
                                color: {{
                                    background: bg,
                                    border: bd,
                                    highlight: {{
                                        background: bg,
                                        border: bd
                                    }}
                                }}, 
                                font: Object.assign({{}}, baseStyle.font, {{ color: fontColor }}), 
                                size: baseStyle.size || 20
                            }});
                        }} else {{
                            // Fallback: force full opacity on current style
                            const currentColor = n.color || {{ background: '#e5e7eb', border: '#c9d1d9' }};
                            const bg = ensureOpaque(currentColor.background || currentColor.color || '#e5e7eb');
                            const bd = ensureOpaque(currentColor.border || currentColor.color || '#c9d1d9');
                            const fontColor = ensureOpaque((n.font || {{ color: '#e5e7eb' }}).color || '#e5e7eb');
                            
                            nodes.update({{ 
                                id, 
                                color: {{
                                    background: bg,
                                    border: bd,
                                    highlight: {{
                                        background: bg,
                                        border: bd
                                    }}
                                }}, 
                                font: Object.assign({{}}, n.font || {{}}, {{ color: fontColor }}), 
                                size: n.size || 20
                            }});
                        }}
                    }});
                    
                    // Force all edges back to full opacity and original styles
                    edges.getIds().forEach(id => {{
                        const e = edges.get(id);
                        if (!e) return;
                        
                        // Get the original base style
                        const baseStyle = window.__baseEdgeStyles ? window.__baseEdgeStyles.get(id) : null;
                        
                        if (baseStyle) {{
                            // Use base style with full opacity
                            const edgeColor = ensureOpaque(baseStyle.color);
                            const fontColor = ensureOpaque(baseStyle.font.color || '#e5e7eb');
                            
                            edges.update({{ 
                                id, 
                                color: {{ 
                                    color: edgeColor, 
                                    highlight: edgeColor, 
                                    hover: edgeColor 
                                }}, 
                                font: Object.assign({{}}, baseStyle.font, {{ color: fontColor }}), 
                                width: baseStyle.width || 1.8
                            }});
                        }} else {{
                            // Fallback: force full opacity on current style
                            const currentColor = (e.color && (e.color.color || e.color)) || '#6b7280';
                            const edgeColor = ensureOpaque(currentColor);
                            const fontColor = ensureOpaque((e.font || {{ color: '#e5e7eb' }}).color || '#e5e7eb');
                            
                            edges.update({{ 
                                id, 
                                color: {{ 
                                    color: edgeColor, 
                                    highlight: edgeColor, 
                                    hover: edgeColor 
                                }}, 
                                font: Object.assign({{}}, e.font || {{}}, {{ color: fontColor }}), 
                                width: e.width || 1.8
                            }});
                        }}
                    }});
                    
                    console.log('All highlights completely reset');
                }} catch(e) {{
                    console.log('Error in completelyResetAllHighlights:', e);
                }}
            }}

            // Helper: reset only colors/opacity (keep positions, sizes, zoom intact)
            function resetColorsOnly() {{
                try {{
                    console.log('resetColorsOnly: Starting color reset to FIXED base styles...');
                    
                    // Build batched updates using FIXED_BASE_STYLES only
                    const nodeUpdates = [];
                    const edgeUpdates = [];

                    // Nodes: ALWAYS use the fixed base style - no exceptions
                    nodes.getIds().forEach(id => {{
                        nodeUpdates.push({{ 
                            id, 
                            color: {{
                                background: FIXED_BASE_STYLES.node.color.background,
                                border: FIXED_BASE_STYLES.node.color.border,
                                highlight: {{
                                    background: FIXED_BASE_STYLES.node.color.highlight.background,
                                    border: FIXED_BASE_STYLES.node.color.highlight.border
                                }}
                            }},
                            font: {{
                                color: FIXED_BASE_STYLES.node.font.color,
                                size: FIXED_BASE_STYLES.node.font.size,
                                face: FIXED_BASE_STYLES.node.font.face
                            }},
                            opacity: FIXED_BASE_STYLES.node.opacity
                        }});
                    }});

                    // Edges: ALWAYS use the fixed base style - no exceptions  
                    edges.getIds().forEach(id => {{
                        edgeUpdates.push({{ 
                            id, 
                            color: {{
                                color: FIXED_BASE_STYLES.edge.color.color,
                                highlight: FIXED_BASE_STYLES.edge.color.highlight,
                                hover: FIXED_BASE_STYLES.edge.color.hover
                            }},
                            font: {{
                                color: FIXED_BASE_STYLES.edge.font.color,
                                size: FIXED_BASE_STYLES.edge.font.size,
                                face: FIXED_BASE_STYLES.edge.font.face
                            }},
                            width: FIXED_BASE_STYLES.edge.width,
                            opacity: FIXED_BASE_STYLES.edge.opacity
                        }});
                    }});

                    // Apply all updates at once
                    if (nodeUpdates.length) {{
                        console.log('Resetting', nodeUpdates.length, 'nodes to fixed base style');
                        nodes.update(nodeUpdates);
                    }}
                    if (edgeUpdates.length) {{
                        console.log('Resetting', edgeUpdates.length, 'edges to fixed base style');
                        edges.update(edgeUpdates);
                    }}
                    
                    console.log('resetColorsOnly: All elements reset to FIXED base styles');
                }} catch(e) {{ 
                    console.log('resetColorsOnly error:', e); 
                }}
            }}

            function clearHighlight() {{
                // 1. Clear all global state that controls highlighting
                window.__highlightNodes = null;
                window.__highlightEdges = null;
                window.__lowAlpha = null;

                // 2. Reset all node/edge styles to the fixed, default appearance
                resetColorsOnly();

                // 3. Force vis.js to redraw itself with the new styles
                try {{
                    if (network && typeof network.redraw === 'function') {{
                        network.redraw();
                    }}
                }} catch (e) {{
                    console.error('Error during network.redraw() in clearHighlight:', e);
                }}

                // 4. Force the external label layer to redraw itself
                try {{
                    if (window.__afterDrawingLabels && typeof window.__afterDrawingLabels === 'function') {{
                        window.__afterDrawingLabels();
                    }}
                }} catch (e) {{
                    console.error('Error during __afterDrawingLabels() in clearHighlight:', e);
                }}
                console.log('clearHighlight: All visual states have been reset.');
            }}

            // Global function for re-rendering MathJax after dynamic content updates
            window.renderMathAfterLoad = function() {{
                if (window.MathJax) {{
                    MathJax.typesetPromise().catch((err) => console.log('MathJax error:', err));
                }}
            }};

            // Initialize MathJax after page load
            document.addEventListener('DOMContentLoaded', function() {{
                // Wait a bit for MathJax to be ready
                setTimeout(() => {{
                    if (window.MathJax) {{
                        MathJax.typesetPromise().catch((err) => console.log('MathJax error:', err));
                    }}
                }}, 1000);
                
                // Capture base styles after network is fully loaded
                setTimeout(() => {{
                    if (typeof initializeBaseStyles === 'function') {{
                        initializeBaseStyles();
                        console.log('Base styles captured for reset functionality');
                    }}
                }}, 2000);
            }});

        </script>
    </body>
    </html>
    """

    return html_content
    
# --- 5. MAIN EXECUTION ---
def main():
    print("--- Starting Constructs Network Visualization ---")
    try:
        graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        graph.run("RETURN 1")
        print("Successfully connected to Neo4j.")
    except Exception as e:
        print(f"Failed to connect to Neo4j. Error: {e}")
        return

    constructs, relationships, papers = fetch_constructs_network_data(graph)

    if not constructs:
        print("No constructs found in the database. Creating test data for highlighting verification...")
        
    # Compute layouts (embedding + centrality)
    embed_pos, central_pos = _compute_layouts(constructs, relationships)

    # Generate HTML content, inject layout JSON
    html_content = create_constructs_network_page(constructs, relationships, papers)
    html_content = html_content.replace('__EMBED_POS__', json.dumps(embed_pos, ensure_ascii=False))
    html_content = html_content.replace('__CENTRAL_POS__', json.dumps(central_pos, ensure_ascii=False))
    
    # Write to file
    try:
        with open(OUTPUT_HTML_FILE, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML file generated successfully: {OUTPUT_HTML_FILE}")
    except Exception as e:
        print(f"Error writing HTML file: {e}")
        return
    
    print("--- Constructs Network Visualization Finished ---")

if __name__ == "__main__":
    main()