// Load Mermaid
let mermaid;
try {
  mermaid = (await import('https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs')).default;
} catch (e) {
  mermaid = typeof window !== 'undefined' ? window.mermaid : undefined;
}

/**
 * Dynamically transforms all links pointing to .md files across the document
 * into the syntax: <a href="mermaid.html?file=[FILENAME].md">[Link Text]</a>
 * Preserves existing link text or dynamically formats it from the filename.
 */
export function transformMdLinks(root = document) {
  // Determine if current page is in a subdirectory (e.g. /claude/, /google/, etc.)
  const scriptEl = document.querySelector('script[src*="mermaid.js"]');
  const isSubdir = (scriptEl && scriptEl.getAttribute('src').startsWith('../')) ||
                   window.location.pathname.includes('/claude/') ||
                   window.location.pathname.includes('/google/') ||
                   window.location.pathname.includes('/business-services/') ||
                   window.location.pathname.includes('/personal-services/');
  const mermaidBase = isSubdir ? '../mermaid.html' : 'mermaid.html';

  const mdLinks = root.querySelectorAll('a[href*=".md"]');
  mdLinks.forEach((link) => {
    const rawHref = link.getAttribute('href');
    if (!rawHref || rawHref.includes('mermaid.html?file=')) return;

    const urlWithoutHash = rawHref.split('#')[0].split('?')[0];
    if (!urlWithoutHash.endsWith('.md')) return;

    // Skip external domains
    if (rawHref.startsWith('http://') || rawHref.startsWith('https://')) {
      try {
        const parsed = new URL(rawHref);
        if (parsed.origin !== window.location.origin) {
          return;
        }
      } catch (e) {
        return;
      }
    }

    // Resolve relative path to repo root for mermaid.html?file=
    let targetFile = rawHref;
    if (isSubdir && !rawHref.startsWith('/') && !rawHref.startsWith('http')) {
      const parts = window.location.pathname.split('/').filter(Boolean);
      const subFolder = parts.length > 1 ? parts[parts.length - 2] : '';
      if (rawHref.startsWith('../')) {
        targetFile = rawHref.replace(/^\.\.\//, '');
      } else if (subFolder) {
        targetFile = `${subFolder}/${rawHref}`;
      }
    } else if (targetFile.startsWith('/')) {
      targetFile = targetFile.replace(/^\//, '');
    } else if (targetFile.startsWith('./')) {
      targetFile = targetFile.replace(/^\.\//, '');
    }

    // Set href to mermaid.html?file=[FILENAME].md
    link.href = `${mermaidBase}?file=${encodeURIComponent(targetFile)}`;

    // Set or format Link Text dynamically if empty
    if (!link.textContent || !link.textContent.trim()) {
      const filename = targetFile.split('/').pop().replace(/\.md$/, '').replace(/[-_]/g, ' ');
      link.textContent = filename.charAt(0).toUpperCase() + filename.slice(1);
    }
  });
}

// Expose globally for dynamic page transitions or markdown readers
if (typeof window !== 'undefined') {
  window.transformMdLinks = transformMdLinks;
}

async function initMermaid() {
  try {
    // 1. Transform all .md links dynamically
    transformMdLinks();

    const m = mermaid || (typeof window !== 'undefined' ? window.mermaid : undefined);
    if (!m) return;

    // 2. Configure theme to match dark aesthetic
    m.initialize({
      startOnLoad: false,
      theme: 'dark',
      securityLevel: 'loose',
      themeVariables: {
        primaryColor: '#1a1b27',
        primaryTextColor: '#c0caf5',
        primaryBorderColor: '#7aa2f7',
        lineColor: '#7aa2f7',
        secondaryColor: '#24283b',
        tertiaryColor: '#1f2335'
      }
    });

    // 3. Transform Markdown code fences (```mermaid) into clean Mermaid containers
    const codeBlocks = document.querySelectorAll('pre code.language-mermaid, pre.language-mermaid');
    codeBlocks.forEach((block) => {
      const parentPre = block.tagName === 'PRE' ? block : block.parentElement;
      const container = document.createElement('div');
      container.className = 'mermaid';
      container.textContent = block.textContent.trim();
      parentPre.replaceWith(container);
    });

    // 4. Render all diagrams into SVGs
    await m.run();

    // 5. Reveal diagram containers and hide fallbacks
    document.querySelectorAll('.diagram-container').forEach((container) => {
      container.classList.add('mermaid-rendered');
    });
  } catch (err) {
    console.warn('Mermaid rendering error:', err);
  }
}

export { initMermaid };

if (typeof document !== 'undefined' && typeof document.querySelector === 'function' && typeof window !== 'undefined' && window.location) {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMermaid);
  } else {
    initMermaid();
  }
}
