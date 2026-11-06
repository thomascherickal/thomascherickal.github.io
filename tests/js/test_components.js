/**
 * test_components.js - Unit Tests for components.js.
 * Tests navigation highlighting, mobile navigation events, articles carousel,
 * email obfuscation, responsive tables, and code sample copy buttons.
 *
 * Strict Repo Compliance:
 * - Zero emojis.
 * - Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
 */

const { setupMockDOM } = require('./test_helpers.js');

class MockElement {
  constructor(tag = 'div', className = '') {
    this.tagName = tag.toUpperCase();
    this._className = '';
    this._classes = new Set();
    this.classList = {
      add: (cls) => { this._classes.add(cls); this._className = Array.from(this._classes).join(' '); },
      remove: (cls) => { this._classes.delete(cls); this._className = Array.from(this._classes).join(' '); },
      contains: (cls) => this._classes.has(cls),
      toggle: (cls) => {
        if (this._classes.has(cls)) {
          this._classes.delete(cls);
          this._className = Array.from(this._classes).join(' ');
          return false;
        } else {
          this._classes.add(cls);
          this._className = Array.from(this._classes).join(' ');
          return true;
        }
      }
    };
    this.className = className;
    this.attributes = {};
    this.style = {};
    this.children = [];
    this.parentElement = null;
    this.parentNode = null;
    this.listeners = {};
    this.textContent = '';
    this.innerText = '';
    this.checked = false;
    this.value = '';
    this.offsetWidth = 300;
    this.offsetLeft = 0;
    this.clientWidth = 900;
    this.scrollWidth = 1200;
    this.rows = [];
    this.cells = [];
  }

  get className() {
    return this._className;
  }

  set className(val) {
    this._className = val || '';
    this._classes = new Set(this._className ? this._className.split(/\s+/).filter(Boolean) : []);
  }

  getAttribute(name) {
    return this.attributes[name] !== undefined ? this.attributes[name] : null;
  }

  setAttribute(name, val) {
    this.attributes[name] = String(val);
  }

  removeAttribute(name) {
    delete this.attributes[name];
  }

  appendChild(child) {
    child.parentElement = this;
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  insertBefore(newChild, refChild) {
    newChild.parentElement = this;
    newChild.parentNode = this;
    const idx = this.children.indexOf(refChild);
    if (idx !== -1) {
      this.children.splice(idx, 0, newChild);
    } else {
      this.children.push(newChild);
    }
    return newChild;
  }

  removeChild(child) {
    const idx = this.children.indexOf(child);
    if (idx !== -1) {
      this.children.splice(idx, 1);
      child.parentElement = null;
      child.parentNode = null;
    }
    return child;
  }

  addEventListener(event, fn) {
    if (!this.listeners[event]) this.listeners[event] = [];
    this.listeners[event].push(fn);
  }

  trigger(event, evData = {}) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(fn => fn(Object.assign({ target: this }, evData)));
    }
  }

  querySelector(selector) {
    const all = this.querySelectorAll(selector);
    return all.length > 0 ? all[0] : null;
  }

  querySelectorAll(selector) {
    if (selector.includes(',')) {
      const parts = selector.split(',').map(s => s.trim());
      const res = [];
      for (const p of parts) {
        res.push(...this.querySelectorAll(p));
      }
      return res;
    }
    const parts = selector.trim().split(/\s+/);
    let currentSet = [this];
    for (const part of parts) {
      const nextSet = [];
      for (const node of currentSet) {
        function collect(n) {
          for (const child of n.children) {
            if (matchesSelector(child, part)) {
              nextSet.push(child);
            }
            collect(child);
          }
        }
        collect(node);
      }
      currentSet = nextSet;
    }
    return currentSet;
  }

  closest(selector) {
    let curr = this;
    while (curr) {
      if (matchesSelector(curr, selector)) return curr;
      curr = curr.parentElement;
    }
    return null;
  }

  scrollTo() {}
  select() {}
}

function matchesSelector(el, selector) {
  if (!el || !el.tagName) return false;
  const parts = selector.split(',').map(s => s.trim());
  return parts.some(sel => {
    if (sel.includes('.')) {
      const dotIdx = sel.indexOf('.');
      const tag = sel.slice(0, dotIdx);
      const classes = sel.slice(dotIdx + 1).split('.');
      if (tag && el.tagName.toLowerCase() !== tag.toLowerCase()) return false;
      return classes.every(c => el.classList.contains(c));
    }
    if (sel.startsWith('#')) {
      return el.getAttribute('id') === sel.slice(1);
    }
    if (sel.includes('[') && sel.includes(']')) {
      const m = sel.match(/\[([a-zA-Z0-9_-]+)([\*\^~|\$]?=)?['"]?([^'"]*)?['"]?\]/);
      if (m) {
        const attr = m[1];
        const op = m[2] || '';
        const val = m[3] || '';
        const attrVal = el.getAttribute(attr);
        if (attrVal === null) return false;
        if (!op) return true;
        if (op === '=' && attrVal === val) return true;
        if (op === '*=' && attrVal.includes(val)) return true;
        return false;
      }
    }
    return el.tagName.toLowerCase() === sel.toLowerCase();
  });
}

async function run(harness) {
  const { assert, setSection } = harness;
  setSection("Components Navigation & UI Tests");

  const components = require('../../components.js');

  // Test 1: Navigation Active Link Highlighting
  {
    const mockDoc = new MockElement('html');
    const body = new MockElement('body');
    mockDoc.appendChild(body);

    const navUl = new MockElement('ul', 'nav-links');
    const li1 = new MockElement('li');
    const link1 = new MockElement('a');
    link1.setAttribute('href', 'pricing.html');
    link1.textContent = 'Pricing';
    li1.appendChild(link1);

    const li2 = new MockElement('li');
    const link2 = new MockElement('a');
    link2.setAttribute('href', 'index.html');
    link2.textContent = 'Home';
    li2.appendChild(link2);

    navUl.appendChild(li1);
    navUl.appendChild(li2);
    body.appendChild(navUl);

    global.document = {
      querySelectorAll: (sel) => {
        if (sel.includes('.nav-links > li > a')) return [link1, link2];
        if (sel.includes('.dropdown-menu a')) return [];
        if (sel.includes('.nav-dropdown')) return [];
        return [];
      },
      querySelector: (sel) => null,
      documentElement: mockDoc,
      body: body
    };
    global.window = {
      location: { pathname: '/pricing.html' },
      innerWidth: 1200
    };

    components.syncActiveNav();

    assert(
      "Navigation Highlighting: Matches active page 'pricing.html'",
      link1.classList.contains('active') && link1.getAttribute('aria-current') === 'page',
      "link1 should have active class and aria-current=page"
    );
    assert(
      "Navigation Highlighting: Inactive page does not have active class",
      !link2.classList.contains('active'),
      "link2 should not have active class"
    );
  }

  // Test 2: Orchestration Category Toggle Highlighting
  {
    const toggle = new MockElement('button', 'nav-dropdown-toggle');
    toggle.textContent = 'Claude Guides';
    const dd = new MockElement('div', 'nav-dropdown');
    dd.appendChild(toggle);

    global.document = {
      querySelectorAll: (sel) => {
        if (sel.includes('.nav-dropdown')) return [dd];
        return [];
      },
      querySelector: () => null
    };
    global.window = {
      location: { pathname: '/claude-code.html' },
      innerWidth: 1200
    };

    components.syncActiveNav();
    assert(
      "Category Toggle Highlighting: Claude category toggle highlighted gold on claude-code.html",
      toggle.style.color === 'var(--gold)',
      `Expected style.color === 'var(--gold)', got ${toggle.style.color}`
    );
  }

  // Test 3: Email Obfuscation
  {
    const emailLink = new MockElement('a', 'email-reveal-link');
    emailLink.setAttribute('data-user', 'contact');
    emailLink.setAttribute('data-domain', 'example.com');
    const displaySpan = new MockElement('span', 'email-display');
    emailLink.appendChild(displaySpan);

    global.document = {
      querySelectorAll: (sel) => {
        if (sel.includes('.email-reveal-link')) return [emailLink];
        return [];
      }
    };

    components.initEmailObfuscation();
    assert(
      "Email Obfuscation: Reconstructs mailto link and display text",
      emailLink.getAttribute('href') === 'mailto:contact@example.com' && displaySpan.textContent === 'contact@example.com',
      `Expected mailto:contact@example.com, got href=${emailLink.getAttribute('href')}`
    );
  }

  // Test 4: Responsive Tables Label Injection
  {
    const table = new MockElement('table');
    const thead = new MockElement('thead');
    const trHead = new MockElement('tr');
    const th1 = new MockElement('th'); th1.textContent = 'Service';
    const th2 = new MockElement('th'); th2.textContent = 'Price';
    trHead.cells = [th1, th2];
    thead.appendChild(trHead);
    table.appendChild(thead);

    const tbody = new MockElement('tbody');
    const trBody = new MockElement('tr');
    const td1 = new MockElement('td'); td1.textContent = 'Training';
    const td2 = new MockElement('td'); td2.textContent = '$2,500';
    trBody.cells = [td1, td2];
    trBody.appendChild(td1);
    trBody.appendChild(td2);
    tbody.appendChild(trBody);
    table.appendChild(tbody);

    table.rows = [trHead, trBody];

    const main = new MockElement('main');
    main.appendChild(table);

    global.document = {
      querySelectorAll: (sel) => {
        if (sel.includes('main table')) return [table];
        return [];
      },
      createElement: (tag) => new MockElement(tag)
    };

    components.initResponsiveTables();
    assert(
      "Responsive Tables: Injects data-label attributes into table cells",
      td1.getAttribute('data-label') === 'Service' && td2.getAttribute('data-label') === 'Price',
      `Expected Service and Price, got td1=${td1.getAttribute('data-label')}, td2=${td2.getAttribute('data-label')}`
    );
  }

  // Test 5: Code Sample Copy Buttons & Container Wrap
  {
    const pre = new MockElement('pre');
    const code = new MockElement('code', 'language-python');
    code.textContent = 'print("hello")';
    code.innerText = 'print("hello")';
    pre.appendChild(code);
    const parent = new MockElement('div');
    parent.appendChild(pre);

    let docClick = null;
    let writtenText = '';
    Object.defineProperty(global.navigator, 'clipboard', {
      value: {
        writeText: (t) => {
          writtenText = t;
          return Promise.resolve();
        }
      },
      configurable: true
    });

    global.document = {
      querySelectorAll: (sel) => {
        if (sel === 'pre') return [pre];
        return [];
      },
      createElement: (tag) => new MockElement(tag),
      addEventListener: (evt, fn) => {
        if (evt === 'click') docClick = fn;
      },
      execCommand: () => true,
      body: new MockElement('body')
    };

    components.initCodeCopyButtons();

    const bar = pre.parentElement.children[0];
    const copyBtn = bar.children.find(c => c.className === 'copy');
    assert(
      "Code Copy Buttons: Creates header bar with dots and copy button",
      bar && bar.className === 'code-bar' && !!copyBtn,
      "pre should be wrapped in container with code-bar and copy button"
    );

    // Simulate clicking copy button
    if (docClick && copyBtn) {
      docClick({ target: copyBtn });
      await Promise.resolve();
      assert(
        "Code Copy Buttons: Copies text to clipboard and displays Copied!",
        writtenText === 'print("hello")' && copyBtn.textContent === 'Copied!',
        `Expected 'print("hello")', got text='${writtenText}', btn='${copyBtn.textContent}'`
      );
    }
  }

  // Test 6: Mobile Navigation Events & Toggle Interactions
  {
    const navToggle = new MockElement('input');
    navToggle.setAttribute('id', 'nav-toggle');
    navToggle.checked = true;

    const navLink = new MockElement('a');
    navLink.setAttribute('href', 'about.html');

    const ddParent = new MockElement('div', 'nav-dropdown');
    const ddToggle = new MockElement('button', 'nav-dropdown-toggle');
    const ddMenu = new MockElement('div', 'dropdown-menu mobile-hidden');
    const ddCaret = new MockElement('span', 'dropdown-caret');
    ddToggle.appendChild(ddCaret);
    ddParent.appendChild(ddToggle);
    ddParent.appendChild(ddMenu);

    const docListeners = {};
    const winListeners = {};

    global.document = {
      getElementById: (id) => (id === 'nav-toggle' ? navToggle : null),
      querySelectorAll: (sel) => {
        if (sel.includes('.nav-links a')) return [navLink];
        if (sel.includes('.nav-dropdown')) return [ddParent];
        return [];
      },
      addEventListener: (evt, fn) => { docListeners[evt] = fn; },
      body: new MockElement('body')
    };
    global.window = {
      innerWidth: 800,
      addEventListener: (evt, fn) => { winListeners[evt] = fn; }
    };

    components.bindNavEvents();

    // Toggle click on mobile opens dropdown
    ddToggle.trigger('click', { preventDefault() {}, stopPropagation() {} });
    assert(
      "Mobile Nav: Clicking dropdown toggle opens mobile dropdown",
      ddParent.classList.contains('open') && ddMenu.classList.contains('mobile-open'),
      "ddParent should have class 'open' and ddMenu should have 'mobile-open'"
    );

    // Escape key closes nav
    if (docListeners['keydown']) {
      docListeners['keydown']({ key: 'Escape' });
      assert(
        "Mobile Nav: Escape key closes nav toggle and dropdowns",
        !navToggle.checked && !ddParent.classList.contains('open'),
        "navToggle.checked should be false after Escape"
      );
    }

    // Resize to desktop (>1024) closes mobile dropdowns
    navToggle.checked = true;
    global.window.innerWidth = 1200;
    if (winListeners['resize']) {
      winListeners['resize']();
      assert(
        "Mobile Nav: Resizing to desktop closes mobile menu",
        !navToggle.checked,
        "navToggle should close on desktop resize"
      );
    }
  }

  // Test 7: Articles Carousel & Marquee
  {
    const marqueeContainer = new MockElement('div');
    const marqueeTrack = new MockElement('div');
    const articlesTrack = new MockElement('div');
    const card1 = new MockElement('div', 'article-carousel-card'); card1.offsetWidth = 320;
    const card2 = new MockElement('div', 'article-carousel-card'); card2.offsetWidth = 320; card2.offsetLeft = 344;
    const card3 = new MockElement('div', 'article-carousel-card'); card3.offsetWidth = 320; card3.offsetLeft = 688;
    articlesTrack.children = [card1, card2, card3];

    const prevBtn = new MockElement('button');
    const nextBtn = new MockElement('button');
    const dotsContainer = new MockElement('div');
    const progressText = new MockElement('span');
    const wrapper = new MockElement('div');

    global.document = {
      getElementById: (id) => {
        if (id === 'articlesMarqueeContainer') return marqueeContainer;
        if (id === 'articlesMarqueeTrack') return marqueeTrack;
        if (id === 'articlesTrack') return articlesTrack;
        if (id === 'articlesPrev') return prevBtn;
        if (id === 'articlesNext') return nextBtn;
        if (id === 'articlesDots') return dotsContainer;
        if (id === 'articlesProgress') return progressText;
        if (id === 'articlesCarouselWrapper') return wrapper;
        return null;
      },
      createElement: (tag) => new MockElement(tag)
    };
    global.window = {
      innerWidth: 1200,
      addEventListener: () => {}
    };

    components.initArticlesCarousel();

    assert(
      "Articles Carousel: Renders dots and initializes progress text",
      dotsContainer.children.length > 0 && progressText.textContent.includes('of 3'),
      `Expected dots > 0 and progress text, got dots=${dotsContainer.children.length}, text='${progressText.textContent}'`
    );

    // Test next button click
    nextBtn.trigger('click');
    assert(
      "Articles Carousel: Clicking Next slides track offset",
      articlesTrack.style.transform && articlesTrack.style.transform.includes('translateX(-'),
      `Expected translateX transform, got '${articlesTrack.style.transform}'`
    );

    // Test marquee touch pause/resume
    marqueeContainer.trigger('touchstart');
    assert(
      "Articles Marquee: touchstart pauses animation",
      marqueeTrack.style.animationPlayState === 'paused',
      "animationPlayState should be paused"
    );

    marqueeContainer.trigger('touchend');
    assert(
      "Articles Marquee: touchend resumes animation",
      marqueeTrack.style.animationPlayState === 'running',
      "animationPlayState should be running"
    );

    // Stop autoplay timer cleanly
    wrapper.trigger('mouseenter');
  }
}

module.exports = { run };
