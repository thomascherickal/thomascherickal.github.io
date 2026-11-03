// Load Mermaid 10 via CDN
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';

async function initMermaid() {
  try {
    // 1. Configure theme to match dark aesthetic
    mermaid.initialize({
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

    // 2. Transform Markdown code fences (```mermaid) into clean Mermaid containers
    const codeBlocks = document.querySelectorAll('pre code.language-mermaid, pre.language-mermaid');
    codeBlocks.forEach((block) => {
      const parentPre = block.tagName === 'PRE' ? block : block.parentElement;
      const container = document.createElement('div');
      container.className = 'mermaid';
      container.textContent = block.textContent.trim();
      parentPre.replaceWith(container);
    });

    // 3. Render all diagrams into SVGs
    await mermaid.run();

    // 4. Reveal diagram containers and hide fallbacks
    document.querySelectorAll('.diagram-container').forEach((container) => {
      container.classList.add('mermaid-rendered');
    });
  } catch (err) {
    console.warn('Mermaid rendering error:', err);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initMermaid);
} else {
  initMermaid();
}
