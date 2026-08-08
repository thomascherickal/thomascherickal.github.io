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
  });
})();
