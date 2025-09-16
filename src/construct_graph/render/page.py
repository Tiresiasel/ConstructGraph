from __future__ import annotations

from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape


STATIC_DIR = Path(__file__).resolve().parent / "static"

def ensure_static_structure() -> None:
    """Create the expected static folder structure if it doesn't exist.

    This is a no-op in production; helpful in dev environments.
    """
    (STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)
    (STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)


def load_template() -> str:
    """Load the network HTML template from the package data.

    Note: we keep a plain .html file with placeholders and inject JSON later.
    """
    tpl_path = Path(__file__).with_name('templates') / 'constructs_network.html.tpl'
    return tpl_path.read_text(encoding='utf-8')


def render_template(template: str, *, embed_pos_json: str, central_pos_json: str) -> str:
    """Inject precomputed layout JSON strings into the template placeholders."""
    return (template
            .replace('__EMBED_POS__', embed_pos_json)
            .replace('__CENTRAL_POS__', central_pos_json))


def render_from_jinja(*, constructs, relationships, papers, embed_pos, central_pos) -> str:
    """Render constructs page using Jinja2 template if available.

    Expects a template named 'constructs_network.html.j2' under templates/.
    """
    templates_dir = Path(__file__).with_name('templates')
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(['html', 'xml'])
    )
    template = env.get_template('constructs_network.html.j2')
    return template.render(
        constructs=constructs,
        relationships=relationships,
        papers=papers,
        embed_pos=embed_pos,
        central_pos=central_pos,
    )


