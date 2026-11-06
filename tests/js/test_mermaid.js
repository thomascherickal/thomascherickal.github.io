/**
 * test_mermaid.js - Unit Tests for mermaid.js Markdown Link Transformations.
 *
 * Strict Repo Compliance:
 * - Zero emojis.
 * - Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
 */

class MockLink {
  constructor(href, text = '') {
    this.tagName = 'A';
    this.attributes = { href };
    this.href = href;
    this.textContent = text;
  }

  getAttribute(name) {
    return this.attributes[name] !== undefined ? this.attributes[name] : null;
  }

  setAttribute(name, val) {
    this.attributes[name] = String(val);
  }
}

async function run(harness) {
  const { assert, setSection } = harness;
  setSection("Mermaid Dynamic MD Link Transformation Tests");

  const { transformMdLinks } = await import('../../mermaid.js');

  // Test 1: Standard markdown link transformation in root directory
  {
    const link = new MockLink('curriculum.md', 'Curriculum Overview');
    const mockRoot = {
      querySelectorAll: (sel) => {
        if (sel.includes('a[href*=".md"]')) return [link];
        return [];
      }
    };
    global.document = {
      querySelector: () => null
    };
    global.window = {
      location: { pathname: '/index.html', origin: 'https://example.com' }
    };

    transformMdLinks(mockRoot);

    assert(
      "Mermaid Links: Transforms relative .md link to mermaid.html?file=...",
      link.href === 'mermaid.html?file=curriculum.md',
      `Expected mermaid.html?file=curriculum.md, got ${link.href}`
    );
    assert(
      "Mermaid Links: Preserves original link text if non-empty",
      link.textContent === 'Curriculum Overview',
      `Expected 'Curriculum Overview', got '${link.textContent}'`
    );
  }

  // Test 2: Subdirectory relative path handling (e.g., /claude/ or /google/)
  {
    const link = new MockLink('../guides/spec.md', '');
    const mockRoot = {
      querySelectorAll: (sel) => [link]
    };
    global.document = {
      querySelector: () => null
    };
    global.window = {
      location: { pathname: '/claude/claude-code.html', origin: 'https://example.com' }
    };

    transformMdLinks(mockRoot);

    assert(
      "Mermaid Links: Uses ../mermaid.html in subdirectories and formats filename link text",
      link.href.startsWith('../mermaid.html?file=') && link.textContent === 'Spec',
      `Expected ../mermaid.html?file=... and text 'Spec', got href=${link.href}, text=${link.textContent}`
    );
  }

  // Test 3: Ignores external links
  {
    const extLink = new MockLink('https://external-site.com/doc.md', 'External Spec');
    const mockRoot = {
      querySelectorAll: () => [extLink]
    };
    global.document = { querySelector: () => null };
    global.window = {
      location: { pathname: '/index.html', origin: 'https://example.com' }
    };

    transformMdLinks(mockRoot);

    assert(
      "Mermaid Links: Preserves external .md URLs without modification",
      extLink.href === 'https://external-site.com/doc.md',
      `Expected external URL unchanged, got ${extLink.href}`
    );
  }

  // Test 4: initMermaid converts code fences and renders SVGs
  {
    const { initMermaid } = await import('../../mermaid.js');
    let initialized = false;
    let runCalled = false;
    global.window = {
      location: { pathname: '/index.html', origin: 'https://example.com' },
      mermaid: {
        initialize: (cfg) => { initialized = true; },
        run: async () => { runCalled = true; }
      }
    };

    const diagramContainer = new MockLink('dummy');
    diagramContainer.classList = {
      add: (cls) => { diagramContainer.className = cls; }
    };

    const pre = {
      tagName: 'PRE',
      className: 'language-mermaid',
      textContent: 'graph TD\nA-->B',
      parentElement: { replaceChild() {} },
      replaceWith: (newEl) => {}
    };

    global.document = {
      querySelector: () => null,
      querySelectorAll: (sel) => {
        if (sel.includes('.diagram-container')) return [diagramContainer];
        if (sel.includes('language-mermaid')) return [pre];
        return [];
      },
      createElement: (tag) => ({
        tagName: tag.toUpperCase(),
        className: '',
        textContent: ''
      })
    };

    await initMermaid();

    assert(
      "Mermaid Init: Initializes configuration and runs diagram rendering",
      initialized && runCalled && diagramContainer.className === 'mermaid-rendered',
      `Expected initialized=true, runCalled=true, rendered container, got init=${initialized}, run=${runCalled}, class=${diagramContainer.className}`
    );
  }
}

module.exports = { run };
