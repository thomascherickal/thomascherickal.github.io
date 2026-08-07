(function () {
  const headerHTML = `<nav aria-label="Main navigation">
    <a href="index.html" class="nav-brand">Thomas Cherickal</a>
    <input type="checkbox" id="nav-toggle" class="nav-toggle">
    <div class="nav-right-container">
      <label for="nav-toggle" class="nav-toggle-label" aria-label="Toggle Navigation Menu">
        <span></span>
      </label>
    </div>
    <ul class="nav-links">
      <li><a href="index.html">Home</a></li>
      <li><a href="portfolio.html">Portfolio</a></li>
      <li><a href="writing.html">Publications</a></li>
      <li><a href="about.html">Capabilities</a></li>
      <li><a href="services.html">Services</a></li>
      <li><a href="pricing.html">Pricing</a></li>
      <li><a href="faqs.html">FAQs</a></li>
    </ul>
  </nav>`;

  const footerHTML = `<footer id="links">
    <div class="footer-top">
      <div>
        <p class="footer-brand-name">The Digital Futurist (2020–present)</p>
        <p class="footer-brand-tagline">Thomas Cherickal · Technical Writer &amp; Developer Educator</p>
      </div>
      <div>
        <div class="newsletter-card">
          <div class="newsletter-info">
            <p class="newsletter-name">📧 The Digital Futurist Newsletter</p>
            <p class="newsletter-desc">How to understand and build emerging technologies.</p>
          </div>
          <a href="https://thomascherickal.kit.com" target="_blank" rel="noopener" class="btn btn-primary" style="white-space:nowrap;">Subscribe Free →</a>
        </div>
      </div>
    </div>
    <p class="footer-label" id="contact-label">Find Me Online</p>
    <div class="social-grid">
      <a href="https://thomascherickal.com" target="_blank" rel="noopener" class="social-link highlight" aria-label="Thomas Cherickal Profile Site"><span class="social-icon">🌐</span>Profile</a>
      <a href="https://github.com/thomascherickal" target="_blank" rel="noopener" class="social-link highlight" aria-label="Thomas Cherickal GitHub Profile"><span class="social-icon">🐙</span>GitHub</a>
      <a href="https://linkedin.com/in/thomascherickal" target="_blank" rel="noopener" class="social-link highlight" aria-label="Thomas Cherickal LinkedIn Profile"><span class="social-icon">💼</span>LinkedIn</a>
      <a href="https://hackernoon.com/u/thomascherickal" target="_blank" rel="noopener" class="social-link highlight" aria-label="Thomas Cherickal HackerNoon Profile"><span class="social-icon">🗞</span>HackerNoon</a>
      <a href="https://thomascherickal.medium.com" target="_blank" rel="noopener" class="social-link" aria-label="Thomas Cherickal Medium Profile"><span class="social-icon">✍️</span>Medium</a>
      <a href="https://thomascherickal.hashnode.dev" target="_blank" rel="noopener" class="social-link" aria-label="Thomas Cherickal Hashnode Blog"><span class="social-icon">🔷</span>Hashnode</a>
      <a href="https://thesingularitypoint.substack.com" target="_blank" rel="noopener" class="social-link" aria-label="Thomas Cherickal Substack"><span class="social-icon">📬</span>Substack</a>
      <a href="https://gitlab.com/thomascherickal" target="_blank" rel="noopener" class="social-link" aria-label="Thomas Cherickal GitLab Profile"><span class="social-icon">🦊</span>GitLab</a>
      <a href="https://thomascherickal.quora.com" target="_blank" rel="noopener" class="social-link" aria-label="Thomas Cherickal Quora Profile"><span class="social-icon">❓</span>Quora</a>
      <a href="https://reddit.com/user/thomascherickal" target="_blank" rel="noopener" class="social-link" aria-label="Thomas Cherickal Reddit Profile"><span class="social-icon">🤖</span>Reddit</a>
      <a href="https://www.kaggle.com/thomascherickal" target="_blank" rel="noopener" class="social-link" aria-label="Thomas Cherickal Kaggle Profile"><span class="social-icon">📊</span>Kaggle</a>
      <a href="https://exercism.org/profiles/thomascherickal" target="_blank" rel="noopener" class="social-link" aria-label="Thomas Cherickal Exercism Profile"><span class="social-icon">🧪</span>Exercism</a>
      <a href="https://profile.codersrank.io/user/thomascherickal/" target="_blank" rel="noopener" class="social-link" aria-label="Thomas Cherickal CodersRank Profile"><span class="social-icon">🏅</span>CodersRank</a>
      <a href="https://www.geeksforgeeks.org/profile/thomascherickal" target="_blank" rel="noopener" class="social-link" aria-label="Thomas Cherickal GeeksforGeeks Profile"><span class="social-icon">🤓</span>Geek4Geeks</a>
      <a href="https://hubpages.com/@thomascherickal" target="_blank" rel="noopener" class="social-link" aria-label="Thomas Cherickal HubPages Profile"><span class="social-icon">📄</span>HubPages</a>
      <a href="https://www.deep-ml.com/profile/thomascherickal" target="_blank" rel="noopener" class="social-link" aria-label="Thomas Cherickal Deep-ML Profile"><span class="social-icon">🧠</span>Deep-ML</a>
      <a href="https://hackerrank.com/profile/thomascherickal" target="_blank" rel="noopener" class="social-link" aria-label="Thomas Cherickal HackerRank Profile"><span class="social-icon">🏆</span>HackerRank</a>
      <a href="https://leetcode.com/u/thomascherickal" target="_blank" rel="noopener" class="social-link" aria-label="Thomas Cherickal LeetCode Profile"><span class="social-icon">💡</span>LeetCode</a>
      <a href="https://linktr.ee/thomascherickal" target="_blank" rel="noopener" class="social-link highlight" aria-label="Thomas Cherickal Linktree Profile"><span class="social-icon">🔗</span>Linktree</a>
      <a href="https://patreon.com/thomascherickal" target="_blank" rel="noopener" class="social-link highlight" aria-label="Thomas Cherickal Patreon Profile"><span class="social-icon">🎨</span>Patreon</a>
      <a href="https://thomascherickal.gumroad.com" target="_blank" rel="noopener" class="social-link highlight" aria-label="Thomas Cherickal Gumroad Store"><span class="social-icon">🛒</span>Gumroad</a>
      <a href="https://topmate.io/thomascherickal" target="_blank" rel="noopener" class="social-link highlight" aria-label="Thomas Cherickal Topmate Mentoring"><span class="social-icon">📅</span>Topmate</a>
    </div>
    <div class="footer-divider"></div>
    <div class="footer-bottom">
      <p class="footer-copy">© 2026 Thomas Cherickal · The Digital Futurist (2020–present)</p>
      <p class="footer-location">📍 Chennai, India 🇮🇳</p>
    </div>
  </footer>`;

  function initComponentLoader() {
    const headerContainer = document.getElementById('site-header');
    const footerContainer = document.getElementById('site-footer');

    // Load Header
    if (headerContainer) {
      fetch('components/header.html')
        .then(res => {
          if (!res.ok) throw new Error('HTTP error');
          return res.text();
        })
        .then(html => {
          headerContainer.innerHTML = html;
          highlightNav();
          bindNavEvents();
        })
        .catch(() => {
          headerContainer.innerHTML = headerHTML;
          highlightNav();
          bindNavEvents();
        });
    }

    // Load Footer
    if (footerContainer) {
      fetch('components/footer.html')
        .then(res => {
          if (!res.ok) throw new Error('HTTP error');
          return res.text();
        })
        .then(html => {
          footerContainer.innerHTML = html;
        })
        .catch(() => {
          footerContainer.innerHTML = footerHTML;
        });
    }
  }

  function highlightNav() {
    const path = window.location.pathname;
    let page = path.substring(path.lastIndexOf('/') + 1);
    if (!page || page === '') page = 'index.html';

    const links = document.querySelectorAll('.nav-links a');
    links.forEach(link => {
      const href = link.getAttribute('href');
      const lowerHref = href ? href.toLowerCase() : '';
      const lowerPage = page ? page.toLowerCase() : '';
      if (lowerHref === lowerPage || (lowerPage === 'index.html' && (lowerHref === 'index.html' || lowerHref === './'))) {
        link.classList.add('active');
        link.setAttribute('aria-current', 'page');
      } else {
        link.classList.remove('active');
        link.removeAttribute('aria-current');
      }
    });
  }

  function bindNavEvents() {
    const navToggle = document.getElementById('nav-toggle');
    const navLinks = document.querySelectorAll('.nav-links a');

    if (navToggle) {
      navLinks.forEach(link => {
        link.addEventListener('click', () => {
          if (navToggle.checked) {
            navToggle.checked = false;
            document.body.style.overflow = '';
          }
        });
      });

      navToggle.addEventListener('change', () => {
        document.body.style.overflow = navToggle.checked ? 'hidden' : '';
      });

      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && navToggle.checked) {
          navToggle.checked = false;
          document.body.style.overflow = '';
        }
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initComponentLoader);
  } else {
    initComponentLoader();
  }
})();
