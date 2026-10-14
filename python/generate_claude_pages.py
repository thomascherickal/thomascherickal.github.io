#!/usr/bin/env python3
"""
generate_claude_pages.py
Generates the 10 production-grade Claude Agents HTML pages with tutorials,
Mermaid diagrams, working code, and matching design aesthetics.
"""

import os

REPO_DIR = "/home/thomas/GitHub.io/thomascherickal.github.io"

# Common head and CSS styles matching Claude Agents Suite
def get_head(title, description, canonical_url):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta content="width=device-width, initial-scale=1.0" name="viewport" />
  <title>{title}</title>
  <meta content="{description}" name="description" />
  <meta content="Thomas Cherickal" name="author" />
  <link href="{canonical_url}" rel="canonical" />
  <!-- Open Graph -->
  <meta content="website" property="og:type" />
  <meta content="{canonical_url}" property="og:url" />
  <meta content="Thomas Cherickal — The Digital Futurist" property="og:site_name" />
  <meta content="{title}" property="og:title" />
  <meta content="{description}" property="og:description" />
  <meta content="https://thomascherickal.github.io/assets/og-card.png" property="og:image" />
  <meta content="en_US" property="og:locale" />
  <!-- Favicons -->
  <link href="assets/favicon.ico" rel="icon" sizes="any" />
  <link href="assets/favicon.svg" rel="icon" type="image/svg+xml" />
  <link href="assets/apple-touch-icon.png" rel="apple-touch-icon" sizes="180x180" />
  <link href="/site.webmanifest" rel="manifest" />
  <meta content="#020408" name="theme-color" />
  <!-- Fonts -->
  <link href="https://fonts.googleapis.com" rel="preconnect" />
  <link crossorigin="" href="https://fonts.gstatic.com" rel="preconnect" />
  <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet" />
  <!-- Mermaid.js for diagrams -->
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script>
    document.addEventListener("DOMContentLoaded", function() {{
      mermaid.initialize({{ startOnLoad: true, theme: 'dark', themeVariables: {{ 'primaryColor': '#0c111d', 'primaryTextColor': '#38bdf8', 'primaryBorderColor': '#38bdf8', 'lineColor': '#fbbf24', 'secondaryColor': '#060912', 'tertiaryColor': '#131926' }} }});
    }});
  </script>
  <style>
    :root {{
      --bg-0: #020408;
      --bg-1: #060912;
      --bg-2: #0b0f19;
      --bg-card: #0c111d;
      --border: rgba(56, 189, 248, 0.2);
      --border-gold: rgba(251, 191, 36, 0.35);
      --cyan: #38bdf8;
      --gold: #fbbf24;
      --claude-accent: #D97757;
      --claude-light: #F38B6C;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --font-main: 'Inter', -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
      --font-display: 'Cinzel', serif;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg-0);
      color: var(--text);
      font-family: var(--font-main);
      line-height: 1.65;
      padding-top: 72px;
      overflow-x: hidden;
    }}

    .container {{
      max-width: 1140px;
      margin: 0 auto;
      padding: 0 1.5rem 5rem;
    }}

    .hero {{
      padding: 4.5rem 0 2.5rem;
      text-align: center;
    }}

    .hero-badge {{
      font-family: var(--font-mono);
      font-size: 0.8rem;
      letter-spacing: 0.15em;
      color: var(--claude-accent);
      text-transform: uppercase;
      margin-bottom: 0.85rem;
      font-weight: 700;
    }}

    .hero-title {{
      font-size: clamp(2rem, 4.5vw, 3rem);
      font-weight: 800;
      line-height: 1.25;
      margin-bottom: 1.15rem;
      background: linear-gradient(135deg, #ffffff 40%, #e2e8f0 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}

    .hero-desc {{
      color: #94a3b8;
      max-width: 780px;
      margin: 0 auto 1.75rem;
      font-size: 1.05rem;
      line-height: 1.7;
    }}


    .content-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 2.5rem;
      margin-bottom: 2.25rem;
      text-align: left;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }}

    .content-card h2 {{
      color: var(--gold);
      font-size: 1.45rem;
      font-weight: 700;
      margin-bottom: 1.25rem;
      line-height: 1.35;
      border-bottom: 1px solid rgba(251, 191, 36, 0.15);
      padding-bottom: 0.6rem;
    }}

    .content-card h3 {{
      color: var(--cyan);
      font-size: 1.15rem;
      font-weight: 600;
      margin: 1.85rem 0 0.85rem;
      line-height: 1.4;
    }}

    .content-card p {{
      color: #e2e8f0;
      font-size: 0.98rem;
      margin-bottom: 1.15rem;
      line-height: 1.8;
    }}

    .content-card ul, .content-card ol {{
      color: #cbd5e1;
      font-size: 0.96rem;
      margin-left: 1.6rem;
      margin-bottom: 1.35rem;
      line-height: 1.8;
    }}

    .content-card li {{
      margin-bottom: 0.75rem;
    }}

    .content-card strong {{
      color: #ffffff;
      font-weight: 600;
    }}

    .content-card a {{
      color: var(--gold);
      text-decoration: underline;
      text-underline-offset: 3px;
    }}

    .content-card a:hover {{
      color: var(--cyan);
    }}

    code:not(pre code) {{
      font-family: var(--font-mono);
      font-size: 0.88em;
      background: rgba(56, 189, 248, 0.08);
      border: 1px solid rgba(56, 189, 248, 0.22);
      color: #7dd3fc;
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
      word-break: break-word;
    }}

    .code-window {{
      background: #040711;
      border: 1px solid rgba(56, 189, 248, 0.25);
      border-radius: 12px;
      margin: 1.85rem 0;
      overflow: hidden;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
    }}

    .code-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.75rem 1.25rem;
      background: rgba(15, 23, 42, 0.95);
      border-bottom: 1px solid rgba(56, 189, 248, 0.15);
      font-family: var(--font-mono);
      font-size: 0.8rem;
      color: #94a3b8;
    }}

    .code-pre {{
      padding: 1.35rem 1.6rem;
      overflow-x: auto;
      font-family: var(--font-mono);
      font-size: 0.86rem;
      line-height: 1.68;
      color: #e2e8f0;
      tab-size: 4;
    }}

    .mermaid-container {{
      background: #050814;
      border: 1px solid rgba(56, 189, 248, 0.25);
      border-radius: 12px;
      padding: 1.5rem;
      margin: 1.75rem 0;
      text-align: center;
      overflow-x: auto;
    }}

    .step-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1.25rem;
      margin: 1.5rem 0;
    }}

    .step-card {{
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid rgba(56, 189, 248, 0.2);
      border-radius: 10px;
      padding: 1.25rem;
      transition: transform 0.2s ease, border-color 0.2s ease;
    }}

    .step-card:hover {{
      transform: translateY(-2px);
      border-color: var(--cyan);
    }}

    .step-num {{
      font-family: var(--font-mono);
      font-size: 0.75rem;
      color: var(--cyan);
      font-weight: 700;
      text-transform: uppercase;
      margin-bottom: 0.4rem;
    }}

    .step-title {{
      font-size: 1.05rem;
      font-weight: 700;
      color: #ffffff;
      margin-bottom: 0.5rem;
    }}

    .step-desc {{
      font-size: 0.88rem;
      color: #94a3b8;
      line-height: 1.55;
    }}

    .cta-banner {{
      background: #070c18;
      border: 1.5px solid var(--border-gold);
      border-radius: 16px;
      padding: 3rem 2rem;
      text-align: center;
      margin: 3rem 0 2rem;
      box-shadow: 0 0 28px rgba(251, 191, 36, 0.18);
    }}

    .btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.8rem 1.65rem;
      border-radius: 8px;
      font-family: var(--font-mono);
      font-size: 0.86rem;
      font-weight: 700;
      text-decoration: none;
      cursor: pointer;
      transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
    }}

    .btn:hover {{ transform: translateY(-2px); }}
    .btn-primary {{
      background: var(--cyan);
      color: #020408 !important;
      box-shadow: 0 0 16px rgba(56, 189, 248, 0.3);
    }}
    .btn-primary:hover {{
      background: #7dd3fc;
      box-shadow: 0 0 24px rgba(56, 189, 248, 0.5);
    }}
    .btn-outline {{
      background: transparent;
      border: 1.5px solid var(--border-gold);
      color: var(--gold) !important;
      box-shadow: 0 0 12px rgba(251, 191, 36, 0.15);
      margin-left: 0.85rem;
    }}
    .btn-outline:hover {{
      background: rgba(251, 191, 36, 0.12);
      border-color: var(--gold);
    }}
  </style>
</head>
<body>
"""

def get_nav_strip(active_slug):
    # Rule: Do not add new navbars for new content
    return ""

def get_footer_cta():
    return """
    <!-- CTA BANNER -->
    <section class="cta-banner">
      <h2 style="color:var(--gold);font-size:1.8rem;margin-bottom:0.75rem;">Deploy Production Claude Agent Systems</h2>
      <p style="color:#94a3b8;max-width:680px;margin:0 auto 1.5rem;font-size:0.95rem;">
        Looking to architect custom Claude Opus 5.5 / Sonnet 5.5 multi-agent teams, self-healing Claude Code workflows, or hybrid local LLM routing? Let's design your enterprise roadmap.
      </p>
      <div>
        <a class="btn btn-primary" href="collaboration.html">Schedule Architecture Consultation →</a>
        <a class="btn btn-outline" href="pricing.html">View Advisory Packages</a>
      </div>
    </section>
  </div>
  <!-- START:FOOTER -->
  <!-- END:FOOTER -->
</body>
</html>
"""

print("Helper definitions loaded.")
