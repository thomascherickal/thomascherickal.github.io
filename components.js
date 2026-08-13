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

  window.addEventListener('DOMContentLoaded', function() {
    var headerFrame = document.getElementById('header-frame');
    if (headerFrame) {
      headerFrame.addEventListener('load', function() {
        syncActiveNav();
        sendViewportHeight();
      });
    }

    // Initialize Reviews System
    initReviewsSystem();
  });
})();

/* ==========================================================================
   REVIEWS CAROUSEL & MODERATION SYSTEM LOGIC (reads reviews.md database)
   ========================================================================== */

function initReviewsSystem() {
  var track = document.getElementById('reviewsTrack');
  if (!track) return; // Exit if not on index.html page

  var STORAGE_KEY = 'tc_reviews_v1';

  // Default fallback reviews (empty until real reviews added)
  var defaultReviews = [];

  // Parse Markdown DB format from reviews.md
  function parseReviewsMarkdown(mdText) {
    if (!mdText) return [];
    mdText = mdText.replace(/<!--[\s\S]*?-->/g, '');
    var blocks = mdText.split('---');
    var parsed = [];

    blocks.forEach(function(block) {
      var lines = block.trim().split('\n');
      var item = {};
      lines.forEach(function(line) {
        var colonIdx = line.indexOf(':');
        if (colonIdx !== -1) {
          var key = line.substring(0, colonIdx).trim().toLowerCase();
          var val = line.substring(colonIdx + 1).trim();
          if (key && val) {
            if (key === 'rating') {
              item.rating = parseInt(val, 10) || 5;
            } else if (key === 'review') {
              item.text = val;
            } else {
              item[key] = val;
            }
          }
        }
      });
      if (item.name && item.text) {
        if (!item.id) item.id = 'rev_md_' + Math.random().toString(36).substr(2, 8);
        if (!item.status) item.status = 'approved';
        parsed.push(item);
      }
    });

    return parsed;
  }

  // Combine reviews from reviews.md database and localStorage submissions
  function loadAllReviews(callback) {
    var storedUserSubmissions = [];
    try {
      var stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        storedUserSubmissions = JSON.parse(stored);
      }
    } catch (e) {
      console.warn('localStorage read error:', e);
    }

    // Try fetching reviews.md text file database
    fetch('./reviews.md')
      .then(function(res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.text();
      })
      .then(function(mdText) {
        var baseReviews = parseReviewsMarkdown(mdText);
        if (baseReviews.length === 0) baseReviews = defaultReviews;
        
        // Merge user submissions from localStorage (avoiding ID duplicates)
        var baseIds = new Set(baseReviews.map(function(r) { return r.id; }));
        storedUserSubmissions.forEach(function(uRev) {
          if (!baseIds.has(uRev.id)) {
            baseReviews.push(uRev);
          }
        });
        
        callback(baseReviews);
      })
      .catch(function(err) {
        console.info('Using local fallback reviews array:', err);
        // Fallback to defaultReviews + user submissions
        var baseReviews = [].concat(defaultReviews);
        var baseIds = new Set(baseReviews.map(function(r) { return r.id; }));
        storedUserSubmissions.forEach(function(uRev) {
          if (!baseIds.has(uRev.id)) {
            baseReviews.push(uRev);
          }
        });
        callback(baseReviews);
      });
  }

  function saveUserSubmissions(reviews) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(reviews));
    } catch (e) {
      console.warn('localStorage write error:', e);
    }
  }

  var allReviews = [];
  var currentIndex = 0;
  var autoPlayInterval = null;

  function getApprovedReviews() {
    return allReviews.filter(function(r) { return r.status === 'approved'; });
  }

  function getPendingReviews() {
    return allReviews.filter(function(r) { return r.status === 'pending'; });
  }

  function getInitials(name) {
    if (!name) return 'TC';
    var parts = name.trim().split(' ');
    if (parts.length >= 2) {
      return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
    }
    return name.substring(0, 2).toUpperCase();
  }

  function renderStars(rating) {
    var stars = '';
    for (var i = 1; i <= 5; i++) {
      stars += i <= rating ? '★' : '☆';
    }
    return stars;
  }

  function getCardsPerView() {
    var w = window.innerWidth;
    if (w <= 640) return 1;
    if (w <= 900) return 2;
    return 3;
  }

  // Render Carousel (NO PHOTOS USED - Clean Text Avatar Initials)
  function renderCarousel() {
    var approved = getApprovedReviews();
    var dotsContainer = document.getElementById('reviewsDots');
    var section = document.getElementById('reviews');
    track.innerHTML = '';

    if (approved.length === 0) {
      if (section) section.style.display = 'none';
      if (dotsContainer) dotsContainer.innerHTML = '';
      return;
    }

    if (section) section.style.display = 'block';

    approved.forEach(function(rev) {
      var card = document.createElement('div');
      card.className = 'review-card';
      card.innerHTML = `
        <div>
          <div class="review-card-header">
            <div class="review-avatar-text">${getInitials(rev.name)}</div>
            <div class="review-meta">
              <div class="review-author-name">${escapeHTML(rev.name)}</div>
              <div class="review-author-role">${escapeHTML(rev.role)}</div>
            </div>
          </div>
          <div class="review-stars">${renderStars(rev.rating)}</div>
          <div class="review-body">"${escapeHTML(rev.text)}"</div>
        </div>
        <div class="review-footer">
          <span class="review-tag">${escapeHTML(rev.tag || 'Verified Review')}</span>
          <span class="review-date">${escapeHTML(rev.date || '2026')}</span>
        </div>
      `;
      track.appendChild(card);
    });

    renderDots();
    updateTrackPosition();
    updatePendingBadge();
  }

  function renderDots() {
    var approved = getApprovedReviews();
    var dotsContainer = document.getElementById('reviewsDots');
    if (!dotsContainer) return;
    dotsContainer.innerHTML = '';

    var cardsPerView = getCardsPerView();
    var totalPages = Math.max(1, approved.length - cardsPerView + 1);

    for (var i = 0; i < totalPages; i++) {
      (function(pageIndex) {
        var dot = document.createElement('button');
        dot.className = 'dot' + (pageIndex === currentIndex ? ' active' : '');
        dot.setAttribute('aria-label', 'Go to review slide ' + (pageIndex + 1));
        dot.addEventListener('click', function() {
          currentIndex = pageIndex;
          updateTrackPosition();
        });
        dotsContainer.appendChild(dot);
      })(i);
    }
  }

  function updateTrackPosition() {
    var approved = getApprovedReviews();
    var cardsPerView = getCardsPerView();
    var maxIndex = Math.max(0, approved.length - cardsPerView);

    if (currentIndex > maxIndex) currentIndex = maxIndex;
    if (currentIndex < 0) currentIndex = 0;

    var cardElem = track.querySelector('.review-card');
    if (cardElem) {
      var cardWidth = cardElem.offsetWidth;
      var gap = 24; // 1.5rem gap
      var offset = currentIndex * (cardWidth + gap);
      track.style.transform = 'translateX(-' + offset + 'px)';
    }

    var dots = document.querySelectorAll('#reviewsDots .dot');
    dots.forEach(function(dot, idx) {
      if (idx === currentIndex) dot.classList.add('active');
      else dot.classList.remove('active');
    });
  }

  function nextSlide() {
    var approved = getApprovedReviews();
    var cardsPerView = getCardsPerView();
    var maxIndex = Math.max(0, approved.length - cardsPerView);

    if (currentIndex >= maxIndex) {
      currentIndex = 0;
    } else {
      currentIndex++;
    }
    updateTrackPosition();
  }

  function prevSlide() {
    var approved = getApprovedReviews();
    var cardsPerView = getCardsPerView();
    var maxIndex = Math.max(0, approved.length - cardsPerView);

    if (currentIndex <= 0) {
      currentIndex = maxIndex;
    } else {
      currentIndex--;
    }
    updateTrackPosition();
  }

  function startAutoPlay() {
    stopAutoPlay();
    autoPlayInterval = setInterval(nextSlide, 5000);
  }

  function stopAutoPlay() {
    if (autoPlayInterval) clearInterval(autoPlayInterval);
  }

  // Navigation button listeners
  var prevBtn = document.getElementById('reviewsPrev');
  var nextBtn = document.getElementById('reviewsNext');
  if (prevBtn) prevBtn.addEventListener('click', function() { prevSlide(); startAutoPlay(); });
  if (nextBtn) nextBtn.addEventListener('click', function() { nextSlide(); startAutoPlay(); });

  // Pause autoplay on mouse enter / resume on leave
  var wrapper = document.querySelector('.reviews-carousel-wrapper');
  if (wrapper) {
    wrapper.addEventListener('mouseenter', stopAutoPlay);
    wrapper.addEventListener('mouseleave', startAutoPlay);
  }

  // Handle touch swipe gestures
  var touchStartX = 0;
  var touchEndX = 0;
  if (track) {
    track.addEventListener('touchstart', function(e) {
      touchStartX = e.changedTouches[0].screenX;
    }, { passive: true });

    track.addEventListener('touchend', function(e) {
      touchEndX = e.changedTouches[0].screenX;
      if (touchStartX - touchEndX > 40) {
        nextSlide();
        startAutoPlay();
      } else if (touchEndX - touchStartX > 40) {
        prevSlide();
        startAutoPlay();
      }
    }, { passive: true });
  }

  window.addEventListener('resize', function() {
    renderDots();
    updateTrackPosition();
  });

  // Helper escape HTML
  function escapeHTML(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Update Pending Badge
  function updatePendingBadge() {
    var pending = getPendingReviews();
    var badge = document.getElementById('pendingCount');
    if (badge) {
      badge.textContent = pending.length;
    }
  }

  // ==========================================================================
  // STAR RATING SELECTOR
  // ==========================================================================
  var starContainer = document.getElementById('starRatingSelect');
  var hiddenRatingInput = document.getElementById('reviewRating');
  if (starContainer && hiddenRatingInput) {
    var starBtns = starContainer.querySelectorAll('.star-btn');
    starBtns.forEach(function(starBtn) {
      starBtn.addEventListener('click', function() {
        var ratingVal = parseInt(starBtn.getAttribute('data-rating'), 10);
        hiddenRatingInput.value = ratingVal;
        starBtns.forEach(function(btn, i) {
          if (i < ratingVal) {
            btn.classList.add('active');
          } else {
            btn.classList.remove('active');
          }
        });
      });
    });
  }

  // ==========================================================================
  // MODALS & FORM SUBMISSION LOGIC
  // ==========================================================================
  var reviewModalBackdrop = document.getElementById('reviewModalBackdrop');
  var openReviewModalBtn = document.getElementById('openReviewModalBtn');
  var closeReviewModalBtn = document.getElementById('closeReviewModalBtn');
  var cancelReviewBtn = document.getElementById('cancelReviewBtn');
  var addReviewForm = document.getElementById('addReviewForm');
  var formStatus = document.getElementById('reviewFormStatus');

  function openReviewModal() {
    if (reviewModalBackdrop) {
      reviewModalBackdrop.classList.add('active');
      reviewModalBackdrop.setAttribute('aria-hidden', 'false');
      if (formStatus) formStatus.style.display = 'none';
    }
  }

  function closeReviewModal() {
    if (reviewModalBackdrop) {
      reviewModalBackdrop.classList.remove('active');
      reviewModalBackdrop.setAttribute('aria-hidden', 'true');
    }
  }

  if (openReviewModalBtn) openReviewModalBtn.addEventListener('click', openReviewModal);
  if (closeReviewModalBtn) closeReviewModalBtn.addEventListener('click', closeReviewModal);
  if (cancelReviewBtn) cancelReviewBtn.addEventListener('click', closeReviewModal);

  if (reviewModalBackdrop) {
    reviewModalBackdrop.addEventListener('click', function(e) {
      if (e.target === reviewModalBackdrop) closeReviewModal();
    });
  }

  // Add Review Form Submit
  if (addReviewForm) {
    addReviewForm.addEventListener('submit', function(e) {
      e.preventDefault();
      var nameInput = document.getElementById('reviewName');
      var roleInput = document.getElementById('reviewRole');
      var textInput = document.getElementById('reviewText');
      var ratingVal = parseInt(hiddenRatingInput.value, 10) || 5;

      if (!nameInput.value.trim() || !roleInput.value.trim() || !textInput.value.trim()) {
        if (formStatus) {
          formStatus.className = 'form-status error';
          formStatus.textContent = 'Please fill out all required fields.';
          formStatus.style.display = 'block';
        }
        return;
      }

      var newReview = {
        id: 'rev_' + Date.now(),
        name: nameInput.value.trim(),
        role: roleInput.value.trim(),
        rating: ratingVal,
        text: textInput.value.trim(),
        date: 'August 2026',
        tag: 'Community Review',
        status: 'pending' // Pending approval!
      };

      allReviews.push(newReview);
      saveUserSubmissions(allReviews);

      if (formStatus) {
        formStatus.className = 'form-status success';
        formStatus.innerHTML = '✅ <strong>Thank you!</strong> Your review has been submitted for moderation and will appear on the carousel once approved.';
        formStatus.style.display = 'block';
      }

      addReviewForm.reset();
      hiddenRatingInput.value = 5;
      var starBtns = starContainer.querySelectorAll('.star-btn');
      starBtns.forEach(function(b) { b.classList.add('active'); });

      updatePendingBadge();
      renderAdminPanel();

      setTimeout(function() {
        closeReviewModal();
      }, 2500);
    });
  }

  // ==========================================================================
  // MODERATION ADMIN PANEL LOGIC
  // ==========================================================================
  var adminModalBackdrop = document.getElementById('adminModalBackdrop');
  var toggleAdminBtn = document.getElementById('toggleAdminBtn');
  var closeAdminModalBtn = document.getElementById('closeAdminModalBtn');
  var pendingReviewsList = document.getElementById('pendingReviewsList');

  function openAdminModal() {
    renderAdminPanel();
    if (adminModalBackdrop) {
      adminModalBackdrop.classList.add('active');
      adminModalBackdrop.setAttribute('aria-hidden', 'false');
    }
  }

  function closeAdminModal() {
    if (adminModalBackdrop) {
      adminModalBackdrop.classList.remove('active');
      adminModalBackdrop.setAttribute('aria-hidden', 'true');
    }
  }

  if (toggleAdminBtn) toggleAdminBtn.addEventListener('click', openAdminModal);
  if (closeAdminModalBtn) closeAdminModalBtn.addEventListener('click', closeAdminModal);
  if (adminModalBackdrop) {
    adminModalBackdrop.addEventListener('click', function(e) {
      if (e.target === adminModalBackdrop) closeAdminModal();
    });
  }

  // Close modals on Escape key
  window.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      closeReviewModal();
      closeAdminModal();
    }
  });

  function renderAdminPanel() {
    if (!pendingReviewsList) return;
    var pending = getPendingReviews();
    pendingReviewsList.innerHTML = '';

    if (pending.length === 0) {
      pendingReviewsList.innerHTML = `
        <div class="empty-pending-msg">
          <p>🎉 <strong>No pending reviews awaiting moderation!</strong></p>
          <p style="font-size:0.8rem;margin-top:0.4rem;color:var(--text-muted);">When visitors submit new reviews via the "Add a Review" form, they will appear here for your approval.</p>
        </div>
      `;
      return;
    }

    pending.forEach(function(rev) {
      var item = document.createElement('div');
      item.className = 'pending-review-card';
      item.innerHTML = `
        <div class="pending-review-header">
          <div>
            <strong style="color:var(--text-primary);font-size:1.05rem;">${escapeHTML(rev.name)}</strong>
            <div style="font-family:var(--font-mono);font-size:0.75rem;color:var(--cyan);">${escapeHTML(rev.role)}</div>
          </div>
          <div style="color:var(--gold);">${renderStars(rev.rating)}</div>
        </div>
        <p style="font-size:0.9rem;color:var(--text-secondary);font-style:italic;line-height:1.6;">"${escapeHTML(rev.text)}"</p>
        <div class="pending-review-actions" style="margin-top:0.5rem;">
          <button class="btn btn-primary approve-btn" data-id="${rev.id}" style="font-size:0.72rem;padding:0.4rem 0.9rem;">
            ✅ Approve &amp; Publish
          </button>
          <button class="btn btn-outline reject-btn" data-id="${rev.id}" style="font-size:0.72rem;padding:0.4rem 0.9rem;color:var(--red);border-color:rgba(251,113,133,0.3);">
            🗑️ Reject &amp; Delete
          </button>
        </div>
      `;

      var approveBtn = item.querySelector('.approve-btn');
      var rejectBtn = item.querySelector('.reject-btn');

      approveBtn.addEventListener('click', function() {
        approveReview(rev.id);
      });

      rejectBtn.addEventListener('click', function() {
        rejectReview(rev.id);
      });

      pendingReviewsList.appendChild(item);
    });
  }

  function approveReview(id) {
    allReviews.forEach(function(rev) {
      if (rev.id === id) {
        rev.status = 'approved';
      }
    });
    saveUserSubmissions(allReviews);
    renderCarousel();
    renderAdminPanel();
  }

  function rejectReview(id) {
    allReviews = allReviews.filter(function(rev) {
      return rev.id !== id;
    });
    saveUserSubmissions(allReviews);
    renderCarousel();
    renderAdminPanel();
  }

  // Load reviews from reviews.md DB and start engine
  loadAllReviews(function(loadedReviews) {
    allReviews = loadedReviews;
    renderCarousel();
    startAutoPlay();
  });
}


