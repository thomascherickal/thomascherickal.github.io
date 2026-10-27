#!/usr/bin/env python3
"""
generate_og_card.py — Generates assets/og-card.png (1200x630) using Headless Google Chrome.
"""

import base64
import os
import subprocess
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(REPO_DIR, "assets")
OUTPUT_PATH = os.path.join(ASSETS_DIR, "og-card.png")
ICON_PATH = os.path.join(ASSETS_DIR, "site-icon.jpeg")

with open(ICON_PATH, "rb") as f:
    icon_b64 = base64.b64encode(f.read()).decode("ascii")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>
    @font-face {{
      font-family: 'Inter';
      src: local('Inter ExtraBold'), local('Inter-ExtraBold');
      font-weight: 800;
    }}
    @font-face {{
      font-family: 'Inter';
      src: local('Inter Black'), local('Inter-Black');
      font-weight: 900;
    }}
    @font-face {{
      font-family: 'Inter';
      src: local('Inter Bold'), local('Inter-Bold');
      font-weight: 700;
    }}
    @font-face {{
      font-family: 'Inter';
      src: local('Inter SemiBold'), local('Inter-SemiBold');
      font-weight: 600;
    }}
    @font-face {{
      font-family: 'JetBrains Mono';
      src: local('JetBrains Mono Bold'), local('JetBrainsMono-Bold');
      font-weight: 700;
    }}
    @font-face {{
      font-family: 'JetBrains Mono';
      src: local('JetBrains Mono ExtraBold'), local('JetBrainsMono-ExtraBold');
      font-weight: 800;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    html, body {{
      width: 1200px;
      height: 630px;
      background: #030712;
      color: #F8FAFC;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    .canvas {{
      position: relative;
      width: 1200px;
      height: 630px;
      background:
        radial-gradient(circle at 14% 20%, rgba(0, 240, 255, 0.16) 0%, transparent 45%),
        radial-gradient(circle at 86% 80%, rgba(251, 191, 36, 0.14) 0%, transparent 45%),
        radial-gradient(circle at 82% 22%, rgba(168, 85, 247, 0.16) 0%, transparent 42%),
        linear-gradient(175deg, #070d1e 0%, #030712 100%);
      padding: 44px 52px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}

    /* Cyber border styling matching site aesthetics */
    .border-frame {{
      position: absolute;
      top: 18px;
      left: 18px;
      right: 18px;
      bottom: 18px;
      border: 1.5px solid rgba(0, 240, 255, 0.4);
      border-radius: 20px;
      pointer-events: none;
      box-shadow:
        inset 0 0 40px rgba(0, 240, 255, 0.06),
        0 0 35px rgba(0, 240, 255, 0.16);
    }}

    .border-frame::before {{
      content: '';
      position: absolute;
      top: -2px;
      left: 50px;
      width: 160px;
      height: 3px;
      background: linear-gradient(90deg, transparent, #00f0ff, transparent);
    }}

    .border-frame::after {{
      content: '';
      position: absolute;
      bottom: -2px;
      right: 50px;
      width: 160px;
      height: 3px;
      background: linear-gradient(90deg, transparent, #fbbf24, transparent);
    }}

    /* Main Content Area */
    .hero-container {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: auto;
      margin-bottom: auto;
      padding-bottom: 8px;
    }}

    .hero-info {{
      max-width: 860px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}

    .eyebrow {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 16px;
      font-weight: 700;
      color: #00F0FF;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 2px;
    }}

    .eyebrow-pill {{
      display: inline-block;
      width: 9px;
      height: 9px;
      background: #00F0FF;
      border-radius: 50%;
      box-shadow: 0 0 14px #00F0FF;
    }}

    .title-name {{
      font-size: 66px;
      font-weight: 900;
      line-height: 1.05;
      letter-spacing: -0.035em;
      color: #FBBF24;
      text-shadow: 0 0 38px rgba(251, 191, 36, 0.38);
    }}

    .title-role {{
      font-size: 36px;
      font-weight: 800;
      color: #FFFFFF;
      letter-spacing: -0.02em;
      line-height: 1.15;
    }}

    .pillars-tags {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-top: 10px;
    }}

    .pillars-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: nowrap;
    }}

    .pillar-chip {{
      border-radius: 9px;
      padding: 7px 15px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 14.5px;
      font-weight: 700;
      letter-spacing: 0.02em;
      white-space: nowrap;
    }}

    .chip-cyan {{
      background: rgba(56, 189, 248, 0.1);
      border: 1px solid rgba(56, 189, 248, 0.45);
      color: #38BDF8;
      box-shadow: 0 0 14px rgba(56, 189, 248, 0.16);
    }}

    .chip-gold {{
      background: rgba(251, 191, 36, 0.1);
      border: 1px solid rgba(251, 191, 36, 0.45);
      color: #FBBF24;
      box-shadow: 0 0 14px rgba(251, 191, 36, 0.16);
    }}

    .chip-emerald {{
      background: rgba(52, 211, 153, 0.1);
      border: 1px solid rgba(52, 211, 153, 0.45);
      color: #34D399;
      box-shadow: 0 0 14px rgba(52, 211, 153, 0.16);
    }}

    .chip-blue {{
      background: rgba(96, 165, 250, 0.1);
      border: 1px solid rgba(96, 165, 250, 0.45);
      color: #60A5FA;
      box-shadow: 0 0 14px rgba(96, 165, 250, 0.16);
    }}

    .chip-purple {{
      background: rgba(192, 132, 252, 0.1);
      border: 1px solid rgba(192, 132, 252, 0.45);
      color: #C084FC;
      box-shadow: 0 0 14px rgba(192, 132, 252, 0.16);
    }}

    .chip-slate {{
      background: rgba(226, 232, 240, 0.08);
      border: 1px solid rgba(226, 232, 240, 0.35);
      color: #E2E8F0;
      box-shadow: 0 0 14px rgba(226, 232, 240, 0.12);
    }}

    .pillar-sep {{
      color: rgba(255, 255, 255, 0.25);
      font-size: 15px;
      font-weight: 400;
    }}

    .avatar-wrapper {{
      width: 175px;
      height: 175px;
      border-radius: 28px;
      border: 2.5px solid rgba(0, 240, 255, 0.7);
      box-shadow:
        0 0 45px rgba(0, 240, 255, 0.4),
        inset 0 0 25px rgba(0, 240, 255, 0.25);
      overflow: hidden;
      flex-shrink: 0;
      margin-left: 30px;
      background: #091322;
    }}

    .avatar-wrapper img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}

    /* Divider */
    .divider-line {{
      height: 2px;
      width: 100%;
      background: linear-gradient(90deg, rgba(0, 240, 255, 0.85) 0%, rgba(192, 132, 252, 0.7) 45%, rgba(251, 191, 36, 0.85) 100%);
      box-shadow: 0 0 14px rgba(0, 240, 255, 0.35);
      margin-bottom: 24px;
    }}

    /* 4 Metrics Grid */
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 18px;
    }}

    .metric-card {{
      background: linear-gradient(160deg, rgba(15, 23, 42, 0.88) 0%, rgba(8, 14, 28, 0.96) 100%);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 14px;
      padding: 18px 22px;
      backdrop-filter: blur(10px);
      display: flex;
      flex-direction: column;
      justify-content: center;
    }}

    .card-gold {{
      border-color: rgba(251, 191, 36, 0.35);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(251, 191, 36, 0.08);
    }}

    .card-cyan {{
      border-color: rgba(56, 189, 248, 0.35);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(56, 189, 248, 0.08);
    }}

    .card-emerald {{
      border-color: rgba(52, 211, 153, 0.35);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(52, 211, 153, 0.08);
    }}

    .card-purple {{
      border-color: rgba(192, 132, 252, 0.35);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(192, 132, 252, 0.08);
    }}

    .metric-val {{
      font-size: 50px;
      font-weight: 900;
      line-height: 1.05;
      margin-bottom: 6px;
      font-family: 'Inter', sans-serif;
    }}

    .val-gold {{
      color: #FBBF24;
      text-shadow: 0 0 22px rgba(251, 191, 36, 0.45);
    }}

    .val-cyan {{
      color: #38BDF8;
      text-shadow: 0 0 22px rgba(56, 189, 248, 0.45);
    }}

    .val-emerald {{
      color: #34D399;
      text-shadow: 0 0 22px rgba(52, 211, 153, 0.45);
    }}

    .val-purple {{
      color: #C084FC;
      text-shadow: 0 0 22px rgba(192, 132, 252, 0.45);
    }}

    .metric-lbl {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 13.5px;
      font-weight: 700;
      color: #CBD5E1;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
  </style>
</head>
<body>
  <div class="canvas">
    <div class="border-frame"></div>
    
    <div class="hero-container">
      <div class="hero-info">
        <div class="eyebrow">
          <span class="eyebrow-pill"></span>
          <span>// The Digital Futurist &middot; AI Systems Training</span>
        </div>
        <h1 class="title-name">Thomas Cherickal</h1>
        <h2 class="title-role">Generative AI Consultant</h2>
        <div class="pillars-tags">
          <div class="pillars-row">
            <span class="pillar-chip chip-gold">Claude AI Consultant</span>
            <span class="pillar-sep">&bull;</span>
            <span class="pillar-chip chip-gold">Gemini AI Consultant</span>
            <span class="pillar-sep">&bull;</span>
            <span class="pillar-chip chip-emerald">AI Agent Orchestrator</span>
          </div>
          <div class="pillars-row">
            <span class="pillar-chip chip-blue">TDD AI-assisted Dev</span>
            <span class="pillar-sep">&bull;</span>
            <span class="pillar-chip chip-purple">Digital AI Transformation</span>
            <span class="pillar-sep">&bull;</span>
            <span class="pillar-chip chip-slate">Office Automation</span>
          </div>
        </div>
      </div>
      <div class="avatar-wrapper">
        <img src="data:image/jpeg;base64,{icon_b64}" alt="Thomas Cherickal Site Icon" />
      </div>
    </div>

    <div>
      <div class="divider-line"></div>
      <div class="metrics-grid">
        <div class="metric-card card-gold">
          <div class="metric-val val-gold">500+</div>
          <div class="metric-lbl">Publications Online</div>
        </div>
        <div class="metric-card card-cyan">
          <div class="metric-val val-cyan">10+</div>
          <div class="metric-lbl">Publishing Platforms</div>
        </div>
        <div class="metric-card card-emerald">
          <div class="metric-val val-emerald">10+</div>
          <div class="metric-lbl">Coding Languages</div>
        </div>
        <div class="metric-card card-purple">
          <div class="metric-val val-purple">2016</div>
          <div class="metric-lbl">Writing &amp; Coding Since</div>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""

with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
    temp_html = f.name
    f.write(html_content)

try:
    chrome_cmd = [
        "google-chrome",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        "--window-size=1200,630",
        f"--screenshot={OUTPUT_PATH}",
        temp_html
    ]
    subprocess.run(chrome_cmd, check=True)
    print(f"Generated {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH):,} bytes)")
finally:
    if os.path.exists(temp_html):
        os.remove(temp_html)
