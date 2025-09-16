import argparse
import json
from .config import CONFIG
from .db.neo4j import get_graph
from .data.fetchers import fetch_constructs, fetch_relationships, fetch_papers
from .render.page import render_from_jinja, render_template
from .layout import compute_layouts
# Import build routine dynamically to avoid relative import issues when run via PYTHONPATH
def _build_main():
    from importlib import import_module
    mod = import_module('build_graph')
    return mod.main()


def cmd_visualize(args: argparse.Namespace) -> int:
    graph = get_graph()
    graph.run("RETURN 1")
    constructs = fetch_constructs(graph)
    relationships = fetch_relationships(graph)
    papers = fetch_papers(graph)
    embed_pos, central_pos = compute_layouts(constructs, relationships)
    try:
        html_content = render_from_jinja(
            constructs=constructs,
            relationships=relationships,
            papers=papers,
            embed_pos=embed_pos,
            central_pos=central_pos,
        )
    except Exception:
        from visualize_graph import create_constructs_network_page
        html_content = create_constructs_network_page(constructs, relationships, papers)
        html_content = render_template(
            html_content,
            embed_pos_json=json.dumps(embed_pos, ensure_ascii=False),
            central_pos_json=json.dumps(central_pos, ensure_ascii=False),
        )
    out = args.output or CONFIG.output_html
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='construct-graph')
    sub = p.add_subparsers(dest='command', required=True)
    v = sub.add_parser('visualize', help='Generate network visualization HTML')
    v.add_argument('-o', '--output', help='Output HTML file (default: index.html)')
    v.set_defaults(func=cmd_visualize)
    b = sub.add_parser('build', help='Ingest PDFs and build/augment the knowledge graph')
    b.set_defaults(func=lambda args: _build_main() or 0)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())


