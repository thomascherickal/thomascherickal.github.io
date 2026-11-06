#!/usr/bin/env python3
"""
test_python_tools.py - Comprehensive Unit Tests for Python Utility Tools.
Covers build.py, build_components.py, build_md_files.py, md_html_sync.py,
sync_schemas.py, sync_pricing.py, generate_sitemap.py, generate_metrics.py,
generate_favicons.py, generate_og_card.py, sync_og_card.py, delete_page.py,
and run_tests.py.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from bs4 import BeautifulSoup
from PIL import Image

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(REPO_DIR, "python")
EXPERT_DIR = os.path.join(REPO_DIR, "nlp-fuzzy-expert")
import sys
for d in (PYTHON_DIR, EXPERT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import build
import build_components
import build_md_files
import delete_page
import generate_favicons
import generate_metrics
import generate_og_card
import generate_sitemap
import md_html_sync
import run_tests
import sync_og_card
import sync_pricing
import sync_schemas


class TestPythonTools(unittest.TestCase):
    """Verifies core build, synchronization, and maintenance scripts in python/."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ==========================================================================
    # 1. build.py & build_components.py
    # ==========================================================================
    def test_build_wrapper_main(self):
        """Verifies build.py wrapper dispatches build and watch modes."""
        with patch("build.build") as mock_b, patch("build.watch") as mock_w:
            build.main([])
            mock_b.assert_called_once()
            mock_w.assert_not_called()

            mock_b.reset_mock()
            build.main(["--watch"])
            mock_b.assert_called_once()
            mock_w.assert_called_once()

    def test_build_components_extract_nav_and_footer(self):
        """Verifies extracting nav and footer tags and handling missing tags."""
        header_sample = '<header><nav class="nav">Nav Content</nav></header>'
        footer_sample = '<div class="wrap"><footer>Footer Content</footer></div>'
        self.assertEqual(build_components.extract_nav(header_sample), '<nav class="nav">Nav Content</nav>')
        self.assertEqual(build_components.extract_footer(footer_sample), '<footer>Footer Content</footer>')

        with self.assertRaises(ValueError):
            build_components.extract_nav("<div>No nav</div>")
        with self.assertRaises(ValueError):
            build_components.extract_footer("<div>No footer</div>")

    def test_build_components_sandbox_all_branches(self):
        """Verifies build_components.build() and watch() across root and subdir HTML templates."""
        header_f = os.path.join(self.temp_dir, "header.html")
        footer_f = os.path.join(self.temp_dir, "footer.html")
        with open(header_f, "w", encoding="utf-8") as f:
            f.write('<nav><a href="index.html">Home</a><a href="business-services/s.html">S</a><img src="assets/logo.png"></nav>')
        with open(footer_f, "w", encoding="utf-8") as f:
            f.write('<footer><a href="contact.html">Contact</a><img src="assets/icon.png"></footer>')

        # 1. Page with skip-link and footer inside <main>, plus legacy mermaid script
        p1 = os.path.join(self.temp_dir, "page1.html")
        with open(p1, "w", encoding="utf-8") as f:
            f.write(
                '<html><head><script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>'
                '<script>mermaid.initialize({});</script></head>'
                '<body><a class="skip-link" href="#main">Skip</a>'
                '<main><h1>P1</h1><!-- START:FOOTER --><footer>Old</footer><!-- END:FOOTER --></main>'
                '</body></html>'
            )

        # 2. Page without skip-link, without </main>, only <body> and </body>
        p2 = os.path.join(self.temp_dir, "page2.html")
        with open(p2, "w", encoding="utf-8") as f:
            f.write('<html><body><h1>P2</h1></body></html>')

        # 3. Subdirectory page with existing header/footer blocks
        sub_dir = os.path.join(self.temp_dir, "business-services")
        os.makedirs(sub_dir, exist_ok=True)
        p3 = os.path.join(sub_dir, "sub.html")
        with open(p3, "w", encoding="utf-8") as f:
            f.write(
                '<html><body><!-- START:HEADER --><nav>Old</nav><!-- END:HEADER -->'
                '<main>Sub</main><script src="./components.js"></script></body></html>'
            )

        # 4. Excluded files
        with open(os.path.join(self.temp_dir, "google43d123.html"), "w", encoding="utf-8") as f:
            f.write("google")
        with open(os.path.join(self.temp_dir, "article-footer_dark.html"), "w", encoding="utf-8") as f:
            f.write("footer")

        with patch.object(build_components, "REPO_DIR", self.temp_dir), \
             patch.object(build_components, "HEADER_SRC", header_f), \
             patch.object(build_components, "FOOTER_SRC", footer_f), \
             patch("synchronizer.ExpertSynchronizer.sync_all", side_effect=RuntimeError("mock sync err")), \
             patch("sync_schemas.sync_html_to_schemas", side_effect=RuntimeError("mock schema err")), \
             patch("md_html_sync.run_sync", side_effect=RuntimeError("mock md err")), \
             patch("generate_metrics.generate_metrics", side_effect=RuntimeError("mock metrics err")), \
             patch("sync_og_card.sync_og_card", side_effect=RuntimeError("mock og err")), \
             patch("sys.stdout", new_callable=io.StringIO):
            build_components.build()
            # Run second time to cover 'No changes needed' branch
            build_components.build()

        with open(p1, "r", encoding="utf-8") as f:
            c1 = f.read()
        self.assertIn("<!-- START:HEADER -->", c1)
        self.assertIn("<!-- START:FOOTER -->", c1)
        self.assertNotIn("mermaid@10.9.1", c1)
        self.assertIn('src="components.js"', c1)

        with open(p3, "r", encoding="utf-8") as f:
            c3 = f.read()
        self.assertIn('href="../index.html"', c3)
        self.assertIn('src="../components.js"', c3)

        # Test watch() loop with mtime change then KeyboardInterrupt
        call_count = {"n": 0}

        def fake_sleep(_):
            call_count["n"] += 1
            if call_count["n"] == 1:
                os.utime(header_f, (999999999, 999999999))
            else:
                raise KeyboardInterrupt()

        with patch.object(build_components, "HEADER_SRC", header_f), \
             patch.object(build_components, "FOOTER_SRC", footer_f), \
             patch.object(build_components, "build") as mock_b, \
             patch("time.sleep", side_effect=fake_sleep), \
             patch("sys.stdout", new_callable=io.StringIO):
            build_components.watch()
            self.assertEqual(mock_b.call_count, 1)

    # ==========================================================================
    # 2. sync_schemas.py
    # ==========================================================================
    def test_sync_schemas_full_lifecycle(self):
        """Verifies sync_schemas filename mapping, extraction, sync, check_only, and main CLI."""
        schema_dir = Path(self.temp_dir) / "schema"
        schema_dir.mkdir(parents=True, exist_ok=True)
        (schema_dir / "service-custom-schema.json").write_text("{}", encoding="utf-8")
        (schema_dir / "custom-doc.json").write_text("{}", encoding="utf-8")

        with patch.object(sync_schemas, "SCHEMA_DIR", schema_dir):
            self.assertEqual(sync_schemas.get_target_schema_filename("index.html"), "home-schema.json")
            self.assertEqual(sync_schemas.get_target_schema_filename("service-custom.html"), "service-custom-schema.json")
            self.assertEqual(sync_schemas.get_target_schema_filename("service-new.html"), "service-new.json")
            self.assertEqual(sync_schemas.get_target_schema_filename("custom-doc.html"), "custom-doc.json")
            self.assertEqual(sync_schemas.get_target_schema_filename("other.html"), "other-schema.json")

        # Create sample HTML pages in temp repo
        repo_path = Path(self.temp_dir)
        valid_html = repo_path / "index.html"
        valid_html.write_text(
            '<html><head><script type="application/ld+json">{"@type": "WebPage", "name": "Home"}</script></head></html>',
            encoding="utf-8"
        )
        invalid_html = repo_path / "bad.html"
        invalid_html.write_text(
            '<html><head><script type="application/ld+json">{bad json</script></head></html>',
            encoding="utf-8"
        )
        no_schema_html = repo_path / "noschema.html"
        no_schema_html.write_text('<html><body>No schema</body></html>', encoding="utf-8")

        new_schema_dir = repo_path / "new_schema"
        with patch.object(sync_schemas, "REPO_DIR", repo_path), \
             patch.object(sync_schemas, "SCHEMA_DIR", new_schema_dir), \
             patch("sys.stdout", new_callable=io.StringIO):
            # Missing schema_dir in check_only mode
            res_missing = sync_schemas.sync_html_to_schemas(check_only=True)
            self.assertEqual(res_missing["errors"], 1)

            # Create schemas
            res_create = sync_schemas.sync_html_to_schemas(check_only=False, verbose=True)
            self.assertEqual(res_create["created"], 1)

            # Unchanged pass
            res_unchanged = sync_schemas.sync_html_to_schemas(check_only=False, verbose=True)
            self.assertEqual(res_unchanged["unchanged"], 1)

            # Modify schema to trigger drift / update
            (new_schema_dir / "home-schema.json").write_text('{"@type": "Old"}\n', encoding="utf-8")
            res_drift = sync_schemas.sync_html_to_schemas(check_only=True, verbose=True)
            self.assertEqual(res_drift["updated"], 1)

            res_update = sync_schemas.sync_html_to_schemas(check_only=False)
            self.assertEqual(res_update["updated"], 1)

            # Test main() exit codes
            with patch("sys.argv", ["sync_schemas.py", "--check", "-v"]):
                with self.assertRaises(SystemExit) as cm:
                    sync_schemas.main()
                self.assertEqual(cm.exception.code, 0)

    # ==========================================================================
    # 3. generate_metrics.py & generate_sitemap.py
    # ==========================================================================
    def test_generate_metrics_and_sitemap(self):
        """Verifies generate_metrics and generate_sitemap functions."""
        self.assertIn("500 bytes", generate_metrics.format_bytes(500))
        self.assertIn("KB", generate_metrics.format_bytes(2048))
        self.assertIn("MB", generate_metrics.format_bytes(2 * 1024 * 1024))
        self.assertIn("git version", generate_metrics.get_git_output(["--version"]))
        self.assertEqual(generate_metrics.get_git_output(["nonexistent-subcommand-xyz"]), "")

        with patch("sys.stdout", new_callable=io.StringIO):
            generate_metrics.generate_metrics()

        temp_sitemap = os.path.join(self.temp_dir, "sitemap.xml")
        with patch.object(generate_sitemap, "SITEMAP_FILE", temp_sitemap), \
             patch("sys.stdout", new_callable=io.StringIO):
            generate_sitemap.generate_sitemap()
            self.assertTrue(os.path.exists(temp_sitemap))

    # ==========================================================================
    # 4. delete_page.py
    # ==========================================================================
    def test_delete_page_full_sandbox_purge(self):
        """Verifies delete_page across HTML, schema, markdown, state, sitemap, header, llms.txt, and links."""
        repo_path = Path(self.temp_dir)
        schema_dir = repo_path / "schema"
        sync_dir = repo_path / "md-html-sync"
        script_dir = repo_path / "python"
        sub_dir = repo_path / "google-training"
        for d in [schema_dir, sync_dir, script_dir, sub_dir, sync_dir / "google-training"]:
            d.mkdir(parents=True, exist_ok=True)

        # Create target HTML in subdirectory
        target_html = sub_dir / "google-temp.html"
        target_html.write_text("<html><body>Temp</body></html>", encoding="utf-8")

        # Create matching schema (both direct and content-matched)
        (schema_dir / "google-temp-schema.json").write_text("{}", encoding="utf-8")
        (schema_dir / "other-ref.json").write_text('{"@id": "https://site/google-temp.html"}', encoding="utf-8")

        # Create matching markdown mirrors
        (sync_dir / "google-training" / "google-temp.md").write_text("# Temp", encoding="utf-8")
        (sync_dir / "google-temp.md").write_text("# Temp root", encoding="utf-8")

        # Create .sync-state.json
        state_file = sync_dir / ".sync-state.json"
        state_file.write_text(json.dumps({
            "google-training/google-temp.html": {"md_name": "google-training/google-temp.md"},
            "other.html": {"md_name": "other.md"}
        }), encoding="utf-8")

        # Create generate_sitemap.py, header.html, llms.txt, and an incoming link file
        (script_dir / "generate_sitemap.py").write_text('PAGES = [\n    "google-temp.html",\n]\n', encoding="utf-8")
        (repo_path / "header.html").write_text('<ul>\n  <li><a href="google-training/google-temp.html">Temp</a></li>\n</ul>\n', encoding="utf-8")
        (repo_path / "llms.txt").write_text("- [Temp](google-training/google-temp.html)\n- [Keep](index.html)\n", encoding="utf-8")
        (repo_path / "linking.html").write_text('<a href="google-training/google-temp.html">Link</a>\n', encoding="utf-8")
        for s in ["build_components.py", "sync_schemas.py", "md_html_sync.py", "check_links.py"]:
            (script_dir / s).write_text("# mock", encoding="utf-8")

        with patch.object(delete_page, "REPO_DIR", repo_path), \
             patch.object(delete_page, "SCHEMA_DIR", schema_dir), \
             patch.object(delete_page, "SYNC_DIR", sync_dir), \
             patch.object(delete_page, "STATE_FILE", state_file), \
             patch.object(delete_page, "SCRIPT_DIR", script_dir), \
             patch("subprocess.run"), \
             patch("sys.stdout", new_callable=io.StringIO):
            # 1. Dry-run first
            delete_page.delete_page("google-temp", dry_run=True, verify=False)
            self.assertTrue(target_html.exists())

            # 2. Actual deletion with verify=True via main()
            with patch("sys.argv", ["delete_page.py", "google-temp"]):
                delete_page.main()

            self.assertFalse(target_html.exists())
            self.assertFalse((schema_dir / "google-temp-schema.json").exists())
            self.assertFalse((sync_dir / "google-temp.md").exists())

            # 3. Non-existent file branch
            delete_page.delete_page("already-gone.html", dry_run=False, verify=False)

    # ==========================================================================
    # 5. build_md_files.py & md_html_sync.py
    # ==========================================================================
    def test_build_md_files_all_converters_and_main(self):
        """Verifies all converter functions and main() in build_md_files.py."""
        self.assertEqual(build_md_files.clean_text(""), "")
        self.assertEqual(build_md_files.clean_text(" a   b \n\n\n c "), "a b \n\n c")

        # Create synthetic HTML files in temp_dir to exercise all converter branches fast
        sync_out = os.path.join(self.temp_dir, "md-html-sync")
        for sub in ["business-services", "personal-services", "claude-training", "google-training"]:
            os.makedirs(os.path.join(self.temp_dir, sub), exist_ok=True)

        with open(os.path.join(self.temp_dir, "articles.html"), "w", encoding="utf-8") as f:
            f.write(
                '<html><body><div class="case-study-card"><h2>Study 1</h2>'
                '<span class="role-chip">AI</span><a class="case-study-cta-left" href="https://example.com">Read</a>'
                '<img class="case-study-cover-img" src="img.png" alt="Alt">'
                '<div class="case-study-meta">2026</div>'
                '<div class="case-study-summary">Executive Summary: Great summary.</div>'
                '<div class="case-study-section-title">The Brief</div><p>Brief text</p>'
                '<div class="case-study-section-title">The Approach</div><ul><li>Step A</li></ul>'
                '<div class="case-study-section-title">What Shipped</div><p>Shipped text</p>'
                '<span class="tech-tag">Python</span></div></body></html>'
            )

        with open(os.path.join(self.temp_dir, "writing.html"), "w", encoding="utf-8") as f:
            f.write(
                '<html><body><div><h3 class="writing-category-title">Cat 1</h3>'
                '<a class="article-card" href="https://example.com"><div class="article-title">Art 1</div>'
                '<span class="meta-pill">5 min</span></a>'
                '<a class="article-card" href="https://example.com/2"><div class="article-title">Art 2</div></a>'
                '</div></body></html>'
            )

        with open(os.path.join(self.temp_dir, "pricing.html"), "w", encoding="utf-8") as f:
            f.write(
                '<html><body>'
                '<div class="package-card"><h2>Pkg 1</h2><div>$3000 USD</div>'
                '<div style="border-radius:9999px">Flagship</div><ul><li>Item 1</li></ul>'
                '<a class="btn" href="contact.html">Book</a></div>'
                '<div class="package-card"><h2>Pkg 2</h2></div>'
                '<div class="price-card-faq"><div class="price-card-title">Pick</div>'
                '<div class="price-amount">$300</div><div style="color:var(--text-secondary)">Desc</div></div>'
                '<div class="parity-tier-card"><div class="parity-badge">T1</div>'
                '<div class="parity-title">Tier 1</div><div class="parity-subtitle">Sub</div>'
                '<div class="parity-rate-val">$3000</div><div class="parity-rate-sub">100%</div>'
                '<div class="parity-desc">Desc</div></div>'
                '</body></html>'
            )

        with open(os.path.join(self.temp_dir, "expertise.html"), "w", encoding="utf-8") as f:
            f.write(
                '<html><body><div class="role-card"><h3>Role 1</h3><p class="role-card-desc">Desc</p>'
                '<ul><li>Item</li></ul><span class="tech-tag">Rust</span></div>'
                '<div class="tech-category"><h3>Group 1</h3><span class="tech-tag">PyTorch</span></div>'
                '</body></html>'
            )

        with open(os.path.join(self.temp_dir, "faqs.html"), "w", encoding="utf-8") as f:
            f.write(
                '<html><body><div class="faq-card"><div class="faq-num">01</div>'
                '<span class="faq-category-badge">General</span><h2>Q1?</h2>'
                '<div class="faq-answer"><p>Ans 1</p><ul><li>Point</li></ul></div></div>'
                '<div class="faq-card"><h3>Q2?</h3><p class="faq-answer">Simple ans</p></div>'
                '</body></html>'
            )

        with open(os.path.join(self.temp_dir, "contact.html"), "w", encoding="utf-8") as f:
            f.write("<html><body>Contact</body></html>")
        with open(os.path.join(self.temp_dir, "404.html"), "w", encoding="utf-8") as f:
            f.write("<html><body>404</body></html>")

        with open(os.path.join(self.temp_dir, "all-business-services.html"), "w", encoding="utf-8") as f:
            f.write(
                '<html><body><main><h1>All Business Services</h1>'
                '<p class="eyebrow">Eyebrow</p><p class="hero-sub">Sub</p>'
                '<a class="skip-link">Skip</a><script>ignore</script>'
                '<h2>Sec 2</h2><h3>Sec 3</h3><h4>Sec 4</h4><p>Para</p>'
                '<ul><li>Bullet</li></ul><ol><li>Num</li></ol>'
                '<pre><code>print(1)</code></pre>'
                '<table><tr><th>H</th></tr><tr><td>V</td></tr></table>'
                '<a class="home-nav-card" href="s.html"><h3>Card</h3><p>Desc</p><span class="role-chip">Tag</span></a>'
                '</main></body></html>'
            )

        with open(os.path.join(self.temp_dir, "business-services", "business-service-test.html"), "w", encoding="utf-8") as f:
            f.write(
                '<html><body><main><h1>Service Test</h1>'
                '<p class="service-hero-badge">Badge</p><p class="service-hero-sub">Sub</p>'
                '<h2>Ignore Me</h2><h2>1. Overview</h2><p>Overview text</p>'
                '<ul><li>Top<ul><li>Nested UL</li></ul><ol><li>Nested OL</li></ol></li></ul>'
                '<table class="data-table"><tr><th>Col</th></tr><tr><td>Val</td></tr></table>'
                '<h2>2. Next</h2></main></body></html>'
            )

        with open(os.path.join(self.temp_dir, "claude-training", "claude-test.html"), "w", encoding="utf-8") as f:
            f.write(
                '<html><body><header class="hero"><h1>Claude Test</h1><p class="hero-badge">Badge</p><p>Sub</p></header>'
                '<div class="pricing-hero-card"><span class="pricing-val">$300</span><span>Note</span></div>'
                '<main><article class="content-card"><h2>Deep Dive</h2><h3>Subhead</h3><p>Text</p>'
                '<ul><li>U1</li></ul><ol><li>O1</li></ol>'
                '<div class="code-window"><div class="code-bar"><span>app.py</span></div><code>x = 1</code></div>'
                '<div class="table-container"><table><tr><th>A</th></tr><tr><td>B</td></tr></table></div>'
                '<div class="step-grid"><div class="step-card"><div class="step-num">01</div>'
                '<div class="step-title">T</div><div class="step-desc">D</div></div></div>'
                '<table><tr><th>X</th></tr><tr><td>Y</td></tr></table>'
                '</article></main></body></html>'
            )

        with patch("sys.stdout", new_callable=io.StringIO):
            build_md_files.main(repo_dir=self.temp_dir, sync_dir=sync_out)

        self.assertTrue(os.path.exists(os.path.join(sync_out, "articles.md")))
        self.assertTrue(os.path.exists(os.path.join(sync_out, "writing.md")))
        self.assertTrue(os.path.exists(os.path.join(sync_out, "pricing.md")))
        self.assertTrue(os.path.exists(os.path.join(sync_out, "expertise.md")))
        self.assertTrue(os.path.exists(os.path.join(sync_out, "faqs.md")))
        self.assertTrue(os.path.exists(os.path.join(sync_out, "all-business-services.md")))
        self.assertTrue(os.path.exists(os.path.join(sync_out, "business-services", "business-service-test.md")))
        self.assertTrue(os.path.exists(os.path.join(sync_out, "claude-training", "claude-test.md")))

        # Empty table check
        empty_soup = BeautifulSoup("<table></table>", "html.parser")
        self.assertEqual(build_md_files.html_table_to_md(empty_soup.find("table")), "")

    def test_md_html_sync_bidirectional_and_cli(self):
        """Verifies md_html_sync HTML->MD, MD->HTML, run_sync conflict resolution, watch, and CLI."""
        sync_dir = os.path.join(self.temp_dir, "md-html-sync")
        os.makedirs(os.path.join(sync_dir, "business-services"), exist_ok=True)
        os.makedirs(os.path.join(sync_dir, "claude-training"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "business-services"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "claude-training"), exist_ok=True)
        state_file = os.path.join(sync_dir, ".sync-state.json")

        # Create index.html, faqs.html, pricing.html, service page, and claude page
        index_html = os.path.join(self.temp_dir, "index.html")
        with open(index_html, "w", encoding="utf-8") as f:
            f.write(
                '<html><body><main><h1 class="hero-name">Thomas Cherickal</h1>'
                '<h2 class="hero-subtitle">Consultant</h2>'
                '<p class="hero-eyebrow"><img src="logo.png">The Future Is Now</p>'
                '<p class="hero-tagline">Tagline</p><p class="hero-bio">Bio text</p>'
                '<span class="role-chip">AI Architect</span>'
                '<div class="stat-item"><div class="stat-num">500+</div><div class="stat-label">Articles</div></div>'
                '<div class="home-services-grid"><a class="home-nav-card" href="s.html">'
                '<span class="home-nav-icon">01</span><h3 class="home-nav-title">Svc</h3>'
                '<p class="home-nav-desc">Desc</p></a></div>'
                '</main></body></html>'
            )

        faqs_html = os.path.join(self.temp_dir, "faqs.html")
        with open(faqs_html, "w", encoding="utf-8") as f:
            f.write(
                '<html><body><main><div class="faq-card"><h2>Q1?</h2>'
                '<p class="faq-answer">Old answer</p></div></main></body></html>'
            )

        pricing_html = os.path.join(self.temp_dir, "pricing.html")
        with open(pricing_html, "w", encoding="utf-8") as f:
            f.write(
                '<html><body><main><div class="package-card"><h2>Pkg</h2>'
                '<div>$3000 - Flagship</div></div></main></body></html>'
            )

        svc_html = os.path.join(self.temp_dir, "business-services", "svc.html")
        with open(svc_html, "w", encoding="utf-8") as f:
            f.write(
                '<html><body><main><h1>Old Svc</h1><h2>1. Overview</h2><p>Text</p>'
                '<table class="data-table"><tbody><tr><td><strong>Week 1</strong></td>'
                '<td>F</td><td>D</td><td>C</td></tr></tbody></table></main></body></html>'
            )

        claude_html = os.path.join(self.temp_dir, "claude-training", "c1.html")
        with open(claude_html, "w", encoding="utf-8") as f:
            f.write('<html><body><main><h1>Old Claude</h1></main></body></html>')

        mappings = {
            "index.html": "home.md",
            "faqs.html": "faqs.md",
            "pricing.html": "pricing.md",
            "business-services/svc.html": "business-services/svc.md",
            "claude-training/c1.html": "claude-training/c1.md"
        }

        with patch.object(md_html_sync, "REPO_DIR", self.temp_dir), \
             patch.object(md_html_sync, "SYNC_DIR", sync_dir), \
             patch.object(md_html_sync, "STATE_FILE", state_file), \
             patch.object(md_html_sync, "PAGE_MAPPINGS", mappings), \
             patch("sys.stdout", new_callable=io.StringIO):
            # Missing file checks
            self.assertEqual(md_html_sync.get_file_hash(os.path.join(self.temp_dir, "missing.txt")), "")
            self.assertEqual(md_html_sync.get_file_mtime(os.path.join(self.temp_dir, "missing.txt")), 0.0)
            self.assertFalse(md_html_sync.sync_html_to_md("missing.html", "missing.md"))
            self.assertFalse(md_html_sync.sync_md_to_html("missing.md", "missing.html"))

            # Initial sync (creates MD files)
            count = md_html_sync.run_sync()
            self.assertEqual(count, 5)

            # Unchanged sync pass
            self.assertEqual(md_html_sync.run_sync(), 0)

            # Force html_to_md and md_to_html via main()
            md_html_sync.main(["--html-to-md"])
            md_html_sync.main(["--md-to-html"])

            # Edit markdown files to trigger MD -> HTML propagation
            home_md = os.path.join(sync_dir, "home.md")
            with open(home_md, "r", encoding="utf-8") as f:
                hm = f.read()
            with open(home_md, "w", encoding="utf-8") as f:
                f.write(hm.replace("# Thomas Cherickal", "# Thomas C."))

            svc_md = os.path.join(sync_dir, "business-services", "svc.md")
            with open(svc_md, "w", encoding="utf-8") as f:
                f.write("# Updated Svc\n\n| **Week 1** | Focus 1 | Deliv 1 | Crit 1 |\n")

            self.assertGreaterEqual(md_html_sync.run_sync(), 2)

            # Test both changed conflict resolution (HTML newer vs MD newer)
            with open(index_html, "a", encoding="utf-8") as f:
                f.write("<!-- edit -->")
            with open(home_md, "a", encoding="utf-8") as f:
                f.write("\n<!-- md edit -->\n")
            os.utime(index_html, (2000000000, 2000000000))
            os.utime(home_md, (1900000000, 1900000000))
            self.assertEqual(md_html_sync.run_sync(), 1)

            # Test orphan pruning in state
            state = md_html_sync.load_state()
            state["deleted_orphan.html"] = {"html_hash": "123"}
            md_html_sync.save_state(state)
            md_html_sync.run_sync()
            self.assertNotIn("deleted_orphan.html", md_html_sync.load_state())

            # Test watch() with change then KeyboardInterrupt
            w_calls = {"n": 0}

            def fake_watch_sleep(_):
                w_calls["n"] += 1
                if w_calls["n"] == 1:
                    with open(home_md, "a", encoding="utf-8") as f:
                        f.write("\n")
                else:
                    raise KeyboardInterrupt()

            with patch("time.sleep", side_effect=fake_watch_sleep):
                md_html_sync.main(["--watch"])

    # ==========================================================================
    # 6. generate_favicons.py, generate_og_card.py, sync_og_card.py
    # ==========================================================================
    def test_generate_favicons_and_og_card_suite(self):
        """Verifies generate_favicons, generate_og_card, and sync_og_card in a sandbox."""
        # 1. Missing source image raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            generate_favicons.generate_favicons(assets_dir=self.temp_dir)

        # 2. Valid synthetic JPEG generates WEBP, PNGs, ICO, and SVG
        jpeg_path = os.path.join(self.temp_dir, "site-icon.jpeg")
        img = Image.new("RGB", (32, 32), color=(10, 20, 40))
        img.save(jpeg_path, format="JPEG")

        with patch("sys.stdout", new_callable=io.StringIO):
            res = generate_favicons.generate_favicons(assets_dir=self.temp_dir)
            self.assertTrue(res)
            self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "site-icon.webp")))
            self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "favicon.ico")))
            self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "favicon.svg")))

        # 3. generate_og_card
        out_og = os.path.join(self.temp_dir, "og-card.png")

        def fake_chrome_run(cmd, check=True):
            Image.new("RGB", (120, 63), color=(0, 0, 0)).save(out_og, format="PNG")

        with patch("subprocess.run", side_effect=fake_chrome_run), \
             patch("sys.stdout", new_callable=io.StringIO):
            gen_path = generate_og_card.generate_og_card(icon_path=jpeg_path, output_path=out_og)
            self.assertEqual(gen_path, out_og)
            self.assertTrue(os.path.exists(out_og))

        # 4. sync_og_card
        fake_index = os.path.join(self.temp_dir, "index.html")
        with open(fake_index, "w", encoding="utf-8") as f:
            f.write(
                '<div class="stats-strip">'
                '<div class="stat-num">500+</div><div class="stat-label">Articles</div>'
                '<div class="stat-num">10+</div><div class="stat-label">Platforms</div>'
                '<div class="stat-num">10+</div><div class="stat-label">Languages</div>'
                '<div class="stat-num">2020</div><div class="stat-label">Since</div>'
                '</div><!-- end -->'
            )

        fake_gen_script = os.path.join(self.temp_dir, "generate_og_card.py")
        shutil.copyfile(os.path.join(PYTHON_DIR, "generate_og_card.py"), fake_gen_script)

        with patch.object(sync_og_card, "INDEX_HTML", fake_index), \
             patch.object(sync_og_card, "GENERATE_SCRIPT", fake_gen_script), \
             patch.object(sync_og_card, "OG_CARD_PATH", out_og), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="Generated")), \
             patch("sys.stdout", new_callable=io.StringIO):
            stats = sync_og_card.extract_index_stats()
            self.assertEqual(len(stats), 4)
            sync_og_card.sync_og_card(force_render=True)
            sync_og_card.sync_og_card(force_render=False)

            # Error branches in sync_og_card
            with patch("subprocess.run", return_value=MagicMock(returncode=1, stderr="fail")):
                with self.assertRaises(RuntimeError):
                    sync_og_card.regenerate_og_card()

            with open(fake_index, "w", encoding="utf-8") as f:
                f.write("<html>No strip</html>")
            with self.assertRaises(ValueError):
                sync_og_card.extract_index_stats()

    # ==========================================================================
    # 7. sync_pricing.py
    # ==========================================================================
    def test_sync_pricing_sandbox(self):
        """Verifies sync_pricing schema and HTML JSON-LD synchronization in a temp directory."""
        s_dir = os.path.join(self.temp_dir, "schema")
        os.makedirs(s_dir, exist_ok=True)

        pricing_schema = os.path.join(s_dir, "pricing-schema.json")
        with open(pricing_schema, "w", encoding="utf-8") as f:
            json.dump({"@graph": [{"@type": "WebPage"}, {"@type": "OfferCatalog"}]}, f)

        home_schema = os.path.join(s_dir, "home-schema.json")
        with open(home_schema, "w", encoding="utf-8") as f:
            json.dump({"@graph": [{"@type": "OfferCatalog", "@id": "#pricing"}]}, f)

        svc_schema = os.path.join(s_dir, "service-generative-ai-transformation.json")
        with open(svc_schema, "w", encoding="utf-8") as f:
            json.dump({"@graph": [{"@type": "Service", "offers": {}}]}, f)

        pricing_html = os.path.join(self.temp_dir, "pricing.html")
        with open(pricing_html, "w", encoding="utf-8") as f:
            f.write(
                '<html><head><script type="application/ld+json">'
                '{"@graph": [{"@type": "WebPage"}, {"@type": "OfferCatalog"}]}'
                '</script></head></html>'
            )

        self.assertTrue(sync_pricing.sync_all_pricing(repo_dir=self.temp_dir, schema_dir=s_dir))

        with open(pricing_schema, "r", encoding="utf-8") as f:
            ps_data = json.load(f)
        self.assertIn("$3000 USD", ps_data["@graph"][0]["description"])

        with open(svc_schema, "r", encoding="utf-8") as f:
            ss_data = json.load(f)
        self.assertEqual(ss_data["@graph"][0]["offers"]["price"], "300.00")

        # Missing file branches
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        self.assertFalse(sync_pricing.sync_pricing_schema_json(schema_dir=empty_dir))
        self.assertFalse(sync_pricing.sync_home_schema_json(schema_dir=empty_dir))
        self.assertFalse(sync_pricing.sync_pricing_html_embedded_schema(repo_dir=empty_dir))

    # ==========================================================================
    # 8. run_tests.py
    # ==========================================================================
    def test_run_tests_recording_result_and_runner(self):
        """Verifies RecordingTestResult methods and run_all_tests outcomes."""
        stream = io.StringIO()
        res = run_tests.RecordingTestResult(stream, descriptions=True, verbosity=1)

        class DummyTest(unittest.TestCase):
            def test_dummy(self):
                """Dummy doc."""
                pass

        t = DummyTest("test_dummy")
        res.startTest(t)
        res.addSuccess(t)
        res.startTest(t)
        try:
            raise AssertionError("fail msg")
        except AssertionError:
            res.addFailure(t, sys.exc_info())
        res.startTest(t)
        try:
            raise RuntimeError("err msg")
        except RuntimeError:
            res.addError(t, sys.exc_info())
        res.startTest(t)
        res.addSkip(t, "skip reason")

        Statuses = [r["status"] for r in res.test_records]
        self.assertEqual(Statuses, ["PASSED", "FAILED", "ERROR", "SKIPPED"])

        # Test run_all_tests with mocked runner (success, failure, and SSOT tamper detection)
        fake_result_ok = MagicMock()
        fake_result_ok.testsRun = 5
        fake_result_ok.failures = []
        fake_result_ok.errors = []
        fake_result_ok.skipped = []
        fake_result_ok.wasSuccessful.return_value = True
        fake_result_ok.test_records = res.test_records

        temp_cache_dir = os.path.join(self.temp_dir, "tests_cache")
        os.makedirs(temp_cache_dir, exist_ok=True)

        with patch.object(run_tests, "TESTS_DIR", temp_cache_dir), \
             patch("unittest.TextTestRunner.run", return_value=fake_result_ok), \
             patch("sys.stdout", new_callable=io.StringIO):
            code_ok = run_tests.run_all_tests()
            self.assertEqual(code_ok, 0)

        # Simulate test failure
        fake_result_fail = MagicMock()
        fake_result_fail.testsRun = 5
        fake_result_fail.failures = [(t, "AssertionError")]
        fake_result_fail.errors = [(t, "RuntimeError")]
        fake_result_fail.skipped = []
        fake_result_fail.wasSuccessful.return_value = False
        fake_result_fail.test_records = res.test_records

        with patch.object(run_tests, "TESTS_DIR", temp_cache_dir), \
             patch("unittest.TextTestRunner.run", return_value=fake_result_fail), \
             patch("auto_healer.generate_failed_tests_report"), \
             patch("sys.stdout", new_callable=io.StringIO):
            code_fail = run_tests.run_all_tests()
            self.assertEqual(code_fail, 1)

        # Simulate SSOT data mutation detection
        with patch.object(run_tests, "TESTS_DIR", temp_cache_dir), \
             patch("unittest.TextTestRunner.run", return_value=fake_result_ok), \
             patch.object(run_tests, "get_data_dir_hashes", side_effect=[{"a.json": "h1"}, {"a.json": "h2", "new.json": "h3"}]), \
             patch("sys.stdout", new_callable=io.StringIO):
            code_tamper = run_tests.run_all_tests()
            self.assertEqual(code_tamper, 1)


if __name__ == "__main__":
    unittest.main()
