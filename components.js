(function () {
  // Auto-detect Live Production vs IDE / Local Environment
  try {
    var host = window.location.hostname;
    var isLive = host === 'thedigitalfuturist.com' || host.indexOf('github.io') !== -1;
    if (isLive) {
      document.documentElement.setAttribute('data-env', 'production');
    } else {
      document.documentElement.setAttribute('data-env', 'ide');
    }
  } catch (e) {}

  // Highlight active links in navigation
  function syncActiveNav() {
    var path = window.location.pathname;
    var page = path.substring(path.lastIndexOf('/') + 1);
    if (!page || page === '') page = 'index.html';

    var links = document.querySelectorAll('.nav-links > li > a');
    links.forEach(function (link) {
      var href = link.getAttribute('href');
      var lowerHref = href ? href.toLowerCase() : '';
      var lowerPage = page ? page.toLowerCase() : '';
      if (
        lowerHref === lowerPage || 
        (lowerPage === 'index.html' && (lowerHref === 'index.html' || lowerHref === './' || lowerHref === '/'))
      ) {
        link.classList.add('active');
        link.setAttribute('aria-current', 'page');
      } else {
        link.classList.remove('active');
        link.removeAttribute('aria-current');
      }
    });

    var dropdownLinks = document.querySelectorAll('.dropdown-menu a');
    dropdownLinks.forEach(function (link) {
      var href = link.getAttribute('href');
      var lowerHref = href ? decodeURIComponent(href).toLowerCase() : '';
      var lowerPage = page ? decodeURIComponent(page).toLowerCase() : '';
      if (lowerHref === lowerPage) {
        link.style.color = 'var(--gold)';
        link.style.background = 'rgba(255, 201, 71, 0.16)';
      }
    });

    // Highlight category toggle if inside that category
    var lowerDecodedPage = page ? decodeURIComponent(page).toLowerCase() : '';
        var orchestrationPages = [
      'claude-101.html',
      'claude-projects.html',
      'claude-code.html',
      'claude-desktop.html',
      'claude-office.html',
      'claude-skills.html',
      'claude-mcp.html',
      'claude-plugins.html',
      'claude-connectors.html',
      'claude-orchestration.html',
      'claude-design.html',
      'claude-science.html',
      'claude-local-llms.html',
      'claude-artifacts.html',
      'claude-dashboards.html',
      'claude-mods.html',
      'claude-whiteboards.html',
      'claude-fable.html'
    ];
    if (orchestrationPages.indexOf(lowerDecodedPage) !== -1) {
      var orchToggles = document.querySelectorAll('.nav-dropdown');
      orchToggles.forEach(function(dd) {
        var toggle = dd.querySelector('.nav-dropdown-toggle');
        if (toggle && (toggle.textContent.indexOf('Agents') !== -1 || toggle.textContent.indexOf('Orchestration') !== -1 || toggle.textContent.indexOf('Claude') !== -1)) {
          toggle.style.color = 'var(--gold)';
        }
      });
    }
  }

  // Bind mobile navigation events
  function bindNavEvents() {
    var navToggle = document.getElementById('nav-toggle');
    var navLinks = document.querySelectorAll('.nav-links a');
    var dropdownParents = document.querySelectorAll('.nav-dropdown');

    if (navToggle) {
      function closeAllDropdowns() {
        dropdownParents.forEach(function (p) {
          p.classList.remove('open');
          var m = p.querySelector('.dropdown-menu');
          if (m) {
            m.classList.remove('mobile-open');
            m.classList.add('mobile-hidden');
          }
          var c = p.querySelector('.dropdown-caret');
          if (c) {
            c.style.transform = '';
          }
        });
      }

      // Keep all dropdown lists closed by default
      closeAllDropdowns();

      navLinks.forEach(function (link) {
        if (link.classList.contains('nav-dropdown-toggle') || !link.getAttribute('href')) {
          return;
        }
        link.addEventListener('click', function () {
          if (navToggle.checked) {
            navToggle.checked = false;
            document.body.style.overflow = '';
            closeAllDropdowns();
          }
        });
      });

      dropdownParents.forEach(function (parent) {
        var toggle = parent.querySelector('.nav-dropdown-toggle');
        var menu = parent.querySelector('.dropdown-menu');
        if (toggle && menu) {
          toggle.addEventListener('click', function (e) {
            if (window.innerWidth <= 1024) {
              e.preventDefault();
              e.stopPropagation();

              var wasOpen = parent.classList.contains('open') || menu.classList.contains('mobile-open');

              // Always close all open dropdowns
              closeAllDropdowns();

              // If it was closed, open it now
              if (!wasOpen) {
                parent.classList.add('open');
                menu.classList.add('mobile-open');
                menu.classList.remove('mobile-hidden');
                var caret = toggle.querySelector('.dropdown-caret');
                if (caret) caret.style.transform = 'rotate(180deg)';
              }
            }
          });
        }
      });

      navToggle.addEventListener('change', function () {
        document.body.style.overflow = navToggle.checked ? 'hidden' : '';
        if (!navToggle.checked) {
          closeAllDropdowns();
        }
      });

      document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && navToggle.checked) {
          navToggle.checked = false;
          document.body.style.overflow = '';
          closeAllDropdowns();
        }
      });

      window.addEventListener('resize', function () {
        if (window.innerWidth > 1024) {
          closeAllDropdowns();
          if (navToggle.checked) {
            navToggle.checked = false;
            document.body.style.overflow = '';
          }
        }
      });
    }
  }

  // Articles Marquee & Carousel (index.html)
  function initArticlesCarousel() {
    var marqueeContainer = document.getElementById('articlesMarqueeContainer');
    var marqueeTrack = document.getElementById('articlesMarqueeTrack');
    
    if (marqueeContainer && marqueeTrack) {
      // Touch pause/resume for mobile devices
      var isPaused = false;
      marqueeContainer.addEventListener('touchstart', function () {
        marqueeTrack.style.animationPlayState = 'paused';
      }, { passive: true });

      marqueeContainer.addEventListener('touchend', function () {
        if (!isPaused) {
          marqueeTrack.style.animationPlayState = 'running';
        }
      }, { passive: true });
    }

    // Discrete carousel fallback if used
    var track = document.getElementById('articlesTrack');
    if (!track) return;

    var cards = track.querySelectorAll('.article-carousel-card');
    if (!cards.length) return;

    var currentIndex = 0;
    var autoPlayTimer = null;
    var prevBtn = document.getElementById('articlesPrev');
    var nextBtn = document.getElementById('articlesNext');
    var dotsContainer = document.getElementById('articlesDots');
    var progressText = document.getElementById('articlesProgress');
    var totalCards = cards.length;

    function getCardsPerView() {
      var w = window.innerWidth;
      if (w <= 640) return 1;
      if (w <= 1024) return 2;
      return 3;
    }

    function getMaxIndex() {
      var cpv = getCardsPerView();
      return Math.max(0, totalCards - cpv);
    }

    function updatePosition() {
      var maxIdx = getMaxIndex();
      if (currentIndex > maxIdx) currentIndex = maxIdx;
      if (currentIndex < 0) currentIndex = 0;

      var cardElem = cards[0];
      if (cardElem) {
        var cardWidth = cardElem.offsetWidth;
        var gap = 24;
        if (cards.length > 1) {
          var diff = cards[1].offsetLeft - (cards[0].offsetLeft + cardWidth);
          if (diff > 0) gap = diff;
        }
        var offset = currentIndex * (cardWidth + gap);
        track.style.transform = 'translateX(-' + offset + 'px)';
      }

      if (dotsContainer) {
        var dots = dotsContainer.querySelectorAll('.carousel-dot');
        dots.forEach(function (dot, i) {
          if (i === currentIndex) {
            dot.classList.add('active');
            if (dotsContainer.scrollWidth > dotsContainer.clientWidth) {
              var dotLeft = dot.offsetLeft;
              var dotWidth = dot.offsetWidth;
              var containerWidth = dotsContainer.clientWidth;
              dotsContainer.scrollTo({
                left: dotLeft - (containerWidth / 2) + (dotWidth / 2),
                behavior: 'smooth'
              });
            }
          } else {
            dot.classList.remove('active');
          }
        });
      }

      if (progressText) {
        var cpv = getCardsPerView();
        var end = Math.min(currentIndex + cpv, totalCards);
        progressText.textContent = (currentIndex + 1) + '-' + end + ' of ' + totalCards;
      }
    }

    function renderDots() {
      if (!dotsContainer) return;
      dotsContainer.innerHTML = '';
      var maxIdx = getMaxIndex();
      var numDots = maxIdx + 1;
      for (var i = 0; i < numDots; i++) {
        (function (idx) {
          var dot = document.createElement('button');
          dot.className = 'carousel-dot' + (idx === currentIndex ? ' active' : '');
          dot.setAttribute('aria-label', 'Go to article slide ' + (idx + 1));
          dot.addEventListener('click', function () {
            currentIndex = idx;
            updatePosition();
            restartAutoplay();
          });
          dotsContainer.appendChild(dot);
        })(i);
      }
    }

    function nextSlide() {
      var maxIdx = getMaxIndex();
      if (currentIndex >= maxIdx) {
        currentIndex = 0;
      } else {
        currentIndex++;
      }
      updatePosition();
    }

    function prevSlide() {
      var maxIdx = getMaxIndex();
      if (currentIndex <= 0) {
        currentIndex = maxIdx;
      } else {
        currentIndex--;
      }
      updatePosition();
    }

    function startAutoplay() {
      stopAutoplay();
      autoPlayTimer = setInterval(nextSlide, 4500);
      if (autoPlayTimer && typeof autoPlayTimer.unref === 'function') {
        autoPlayTimer.unref();
      }
    }

    function stopAutoplay() {
      if (autoPlayTimer) {
        clearInterval(autoPlayTimer);
        autoPlayTimer = null;
      }
    }

    function restartAutoplay() {
      stopAutoplay();
      startAutoplay();
    }

    if (nextBtn) {
      nextBtn.addEventListener('click', function () {
        nextSlide();
        restartAutoplay();
      });
    }

    if (prevBtn) {
      prevBtn.addEventListener('click', function () {
        prevSlide();
        restartAutoplay();
      });
    }

    var wrapper = document.getElementById('articlesCarouselWrapper');
    if (wrapper) {
      wrapper.addEventListener('mouseenter', stopAutoplay);
      wrapper.addEventListener('mouseleave', startAutoplay);
    }

    // Touch & Swipe gestures with vertical scroll protection
    var startX = 0;
    var startY = 0;
    var isTouching = false;

    track.addEventListener('touchstart', function (e) {
      if (!e.touches || e.touches.length === 0) return;
      startX = e.touches[0].clientX;
      startY = e.touches[0].clientY;
      isTouching = true;
      stopAutoplay();
    }, { passive: true });

    track.addEventListener('touchmove', function (e) {
      if (!isTouching || !e.touches || e.touches.length === 0) return;
    }, { passive: true });

    track.addEventListener('touchend', function (e) {
      if (!isTouching) return;
      isTouching = false;
      if (!e.changedTouches || e.changedTouches.length === 0) {
        restartAutoplay();
        return;
      }
      var endX = e.changedTouches[0].clientX;
      var endY = e.changedTouches[0].clientY;
      var diffX = startX - endX;
      var diffY = startY - endY;

      if (Math.abs(diffX) > 35 && Math.abs(diffX) > Math.abs(diffY)) {
        if (diffX > 0) {
          nextSlide();
        } else {
          prevSlide();
        }
      }
      restartAutoplay();
    }, { passive: true });

    var resizeTimer = null;
    window.addEventListener('resize', function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        renderDots();
        updatePosition();
      }, 50);
    });

    window.addEventListener('orientationchange', function () {
      setTimeout(function () {
        renderDots();
        updatePosition();
      }, 150);
    });

    renderDots();
    updatePosition();
    startAutoplay();
  }

  function initEmailObfuscation() {
    var emailLinks = document.querySelectorAll('.email-reveal-link');
    emailLinks.forEach(function (link) {
      var user = link.getAttribute('data-user');
      var domain = link.getAttribute('data-domain');
      if (user && domain) {
        var email = user + '@' + domain;
        link.setAttribute('href', 'mailto:' + email);
        var displayEl = link.querySelector('.email-display');
        if (displayEl) {
          displayEl.textContent = email;
        }
      }
    });
  }

  function initResponsiveTables() {
    document.querySelectorAll('main table').forEach(function (table) {
      if (table.querySelector('[colspan], [rowspan]')) return;
      var headRow = table.querySelector('thead tr') || table.rows[0];
      if (!headRow) return;
      var labels = Array.prototype.map.call(headRow.cells, function (c) {
        return c.textContent.replace(/\s+/g, ' ').trim();
      });
      var rows = table.querySelectorAll('tbody tr');
      if (!rows.length) rows = Array.prototype.slice.call(table.rows, 1);
      Array.prototype.forEach.call(rows, function (tr) {
        Array.prototype.forEach.call(tr.cells, function (cell, i) {
          if (labels[i]) cell.setAttribute('data-label', labels[i]);
        });
      });
      var wrap = table.parentElement;
      if (!wrap || !wrap.classList.contains('rt-wrap')) {
        wrap = document.createElement('div');
        wrap.className = 'rt-wrap';
        table.parentNode.insertBefore(wrap, table);
        wrap.appendChild(table);
      }
      table.classList.add('rt-stack');
      wrap.setAttribute('role', 'region');
      var cap = table.querySelector('caption');
      wrap.setAttribute('aria-label', cap ? cap.textContent.trim() : 'Data table');
      if (wrap.scrollWidth > wrap.clientWidth) {
        wrap.setAttribute('tabindex', '0');
      }
    });
  }

  // Universal Code Sample Copy Buttons & Sharp Boundaries
  function initCodeCopyButtons() {
    var preElements = document.querySelectorAll('pre');
    preElements.forEach(function (pre) {
      // Find or establish container
      var container = pre.closest('.code-window, .codewrap, .code, .code-box');
      if (!container) {
        var styledParent = pre.parentElement;
        if (styledParent && styledParent.tagName === 'DIV' && styledParent.children.length === 1 &&
            (styledParent.getAttribute('style') || '').indexOf('#090d14') !== -1) {
          container = styledParent;
          container.classList.add('code-box');
        } else {
          var wrapper = document.createElement('div');
          wrapper.className = 'code-window';
          pre.parentNode.insertBefore(wrapper, pre);
          wrapper.appendChild(pre);
          container = wrapper;
        }
      }

      // Check for existing header/bar
      var bar = container.querySelector('.code-bar, .code-head, .cap, .code-header');
      if (!bar) {
        bar = document.createElement('div');
        bar.className = 'code-bar';
        
        var dots = document.createElement('div');
        dots.className = 'code-dots';
        dots.innerHTML = '<span class="code-dot red"></span><span class="code-dot yellow"></span><span class="code-dot green"></span>';
        bar.appendChild(dots);

        var label = document.createElement('span');
        var codeEl = pre.querySelector('code');
        var lang = '';
        if (codeEl && codeEl.className) {
          var m = codeEl.className.match(/language-([a-zA-Z0-9_-]+)/);
          if (m) lang = m[1];
        }
        label.textContent = lang ? lang.toUpperCase() : 'Code';
        label.style.marginLeft = '8px';
        bar.appendChild(label);

        container.insertBefore(bar, pre);
      }

      // Ensure a copy button exists in the bar
      var copyBtn = bar.querySelector('.copy, .copy-btn');
      if (!copyBtn) {
        copyBtn = document.createElement('button');
        copyBtn.className = 'copy';
        copyBtn.type = 'button';
        copyBtn.setAttribute('aria-label', 'Copy code to clipboard');
        copyBtn.textContent = 'Copy';
        bar.appendChild(copyBtn);
      }
    });

    // Global click delegation for all copy buttons
    document.addEventListener('click', function (e) {
      var btn = e.target.closest ? e.target.closest('button.copy, .copy-btn') : null;
      if (!btn) return;

      var container = btn.closest('.code-window, .codewrap, .code, .code-box') || (btn.parentElement ? btn.parentElement.parentElement : null);
      var code = container ? container.querySelector('code') : null;
      var text = '';
      if (code) {
        text = code.innerText || code.textContent;
      } else {
        var pre = container ? container.querySelector('pre') : null;
        if (pre) text = pre.innerText || pre.textContent;
      }
      if (!text) return;

      function onCopied() {
        var origText = btn.textContent;
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        var timer = setTimeout(function () {
          btn.textContent = origText;
          btn.classList.remove('copied');
        }, 1500);
        if (timer && typeof timer.unref === 'function') timer.unref();
      }

      function fallback(t) {
        var ta = document.createElement('textarea');
        ta.value = t;
        ta.setAttribute('readonly', '');
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try {
          document.execCommand('copy');
          onCopied();
        } catch (err) {
          btn.textContent = 'Failed';
          var failTimer = setTimeout(function () { btn.textContent = 'Copy'; }, 1500);
          if (failTimer && typeof failTimer.unref === 'function') failTimer.unref();
        }
        document.body.removeChild(ta);
      }

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(onCopied, function () {
          fallback(text);
        });
      } else {
        fallback(text);
      }
    });
  }

  function init() {
    syncActiveNav();
    bindNavEvents();
    initArticlesCarousel();
    initEmailObfuscation();
    initResponsiveTables();
    initCodeCopyButtons();
  }

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      window.addEventListener('DOMContentLoaded', init);
    } else {
      init();
    }
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      init: init,
      syncActiveNav: syncActiveNav,
      bindNavEvents: bindNavEvents,
      initArticlesCarousel: initArticlesCarousel,
      initEmailObfuscation: initEmailObfuscation,
      initResponsiveTables: initResponsiveTables,
      initCodeCopyButtons: initCodeCopyButtons
    };
  }
})();
