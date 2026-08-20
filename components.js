(function () {
  // Listen for iframe height messages from header.html and footer.html
  window.addEventListener('message', function (e) {
    if (e.data && e.data.frameId && e.data.height) {
      var iframe = document.getElementById(e.data.frameId);
      if (iframe) {
        iframe.style.height = e.data.height + 'px';
      }
    }
  });

  // Send parent viewport height to the header iframe so the mobile menu
  // can size its scrollable panel correctly
  function sendViewportHeight() {
    var headerFrame = document.getElementById('header-frame');
    if (headerFrame && headerFrame.contentWindow) {
      headerFrame.contentWindow.postMessage({
        type: 'parentViewportHeight',
        height: window.innerHeight
      }, '*');
    }
  }

  window.addEventListener('resize', sendViewportHeight);

  // Highlight active links inside header iframe
  function syncActiveNav() {
    var path = window.location.pathname;
    var page = path.substring(path.lastIndexOf('/') + 1);
    if (!page || page === '') page = 'index.html';

    var headerFrame = document.getElementById('header-frame');
    if (headerFrame && headerFrame.contentWindow) {
      try {
        var doc = headerFrame.contentDocument || headerFrame.contentWindow.document;
        if (doc) {
          var links = doc.querySelectorAll('.nav-links > li > a');
          links.forEach(function(link) {
            var href = link.getAttribute('href');
            var lowerHref = href ? href.toLowerCase() : '';
            var lowerPage = page ? page.toLowerCase() : '';
            if (
              lowerHref === lowerPage || 
              (lowerPage === 'index.html' && (lowerHref === 'index.html' || lowerHref === './')) ||
              (lowerPage.startsWith('service-') && lowerHref === 'services.html')
            ) {
              link.classList.add('active');
              link.setAttribute('aria-current', 'page');
            } else {
              link.classList.remove('active');
              link.removeAttribute('aria-current');
            }
          });

          var dropdownLinks = doc.querySelectorAll('.dropdown-menu a');
          dropdownLinks.forEach(function(link) {
            var href = link.getAttribute('href');
            var lowerHref = href ? href.toLowerCase() : '';
            var lowerPage = page ? page.toLowerCase() : '';
            if (lowerHref === lowerPage) {
              link.style.color = 'var(--gold)';
              link.style.background = 'rgba(255, 201, 71, 0.16)';
            }
          });
        }
      } catch (err) {
        // Cross-origin fallback when running file:// protocol in some browsers
      }
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
        progressText.textContent = (currentIndex + 1) + '–' + end + ' of ' + totalCards;
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

  window.addEventListener('DOMContentLoaded', function() {
    initArticlesCarousel();
    var headerFrame = document.getElementById('header-frame');
    if (headerFrame) {
      headerFrame.addEventListener('load', function() {
        syncActiveNav();
        sendViewportHeight();
      });
    }
  });
})();
