class SiteUI {
  constructor() {
    this.searchToggle = document.getElementById('searchToggle');
    this.searchInput = document.getElementById('searchInput');
    this.searchResults = document.getElementById('searchResults');
    this.navToggle = document.getElementById('navToggle');
    this.mainNav = document.getElementById('mainNav');
    this.themeToggle = document.getElementById('themeToggle');
    this.posts = [];
    this.lightbox = null;
    this.init();
  }

  async init() {
    this.bindNav();
    this.bindTheme();
    this.bindSearchUI();
    await this.loadSearchIndex();
    this.bindSearchEvents();
    this.bindAnchors();
    this.initLightbox();
    this.bindPostEnhancements();
  }

  bindNav() {
    if (!this.navToggle || !this.mainNav) return;
    this.navToggle.addEventListener('click', () => {
      const isOpen = this.mainNav.classList.toggle('open');
      this.navToggle.setAttribute('aria-expanded', String(isOpen));
    });
  }

  bindTheme() {
    if (!this.themeToggle) return;
    const preferred = localStorage.getItem('theme-preference');
    if (preferred) {
      document.documentElement.setAttribute('data-theme', preferred);
    }
    this.themeToggle.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('theme-preference', next);
    });
  }

  async loadSearchIndex() {
    try {
      const response = await fetch('/search-index.json', { cache: 'no-store' });
      this.posts = response.ok ? await response.json() : [];
    } catch {
      this.posts = [];
    }
  }

  bindSearchUI() {
    if (!this.searchInput || !this.searchResults || !this.searchToggle) return;
    this.searchToggle.addEventListener('click', () => {
      this.searchInput.focus();
      this.search(this.searchInput.value);
    });
  }

  bindSearchEvents() {
    if (!this.searchInput || !this.searchResults) return;
    this.searchInput.addEventListener('input', (e) => this.search(e.target.value));
    document.addEventListener('click', (e) => {
      if (!this.searchResults.contains(e.target) && e.target !== this.searchInput) {
        this.searchResults.innerHTML = '';
      }
    });
  }

  search(query) {
    const q = (query || '').trim().toLowerCase();
    if (!q) {
      this.searchResults.innerHTML = '';
      return;
    }

    const results = this.posts.filter((post) => {
      const tags = Array.isArray(post.tags)
        ? post.tags.join(' ')
        : (post.tags ? String(post.tags) : '');
      const excerpt = post.excerpt ? String(post.excerpt) : '';
      const text = `${post.title || ''} ${post.content || ''} ${tags} ${excerpt}`.toLowerCase();
      return text.includes(q);
    }).slice(0, 8);

    if (results.length === 0) {
      this.searchResults.innerHTML = '<div class="search-result-item"><div class="search-result-title">无匹配结果</div></div>';
      return;
    }

    this.searchResults.innerHTML = results.map((post) => {
      const title = this.escapeHTML(post.title || 'Untitled');
      const excerptRaw = (post.excerpt || post.content || '').replace(/<[^>]*>/g, '').slice(0, 120);
      const excerpt = this.escapeHTML(excerptRaw);
      return `<a href="${post.url}" class="search-result-item"><div class="search-result-title">${title}</div><div class="search-result-excerpt">${excerpt}</div></a>`;
    }).join('');
  }

  escapeHTML(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  bindAnchors() {
    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
      anchor.addEventListener('click', function(e) {
        const target = document.querySelector(this.getAttribute('href'));
        if (!target) return;
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth' });
      });
    });
  }

  initLightbox() {
    const box = document.createElement('div');
    box.id = 'lightbox';
    box.innerHTML = '<div class="lightbox-content"><button class="lightbox-close" aria-label="Close">&times;</button><img class="lightbox-img" src="" alt=""></div>';
    document.body.appendChild(box);
    this.lightbox = box;

    box.addEventListener('click', (e) => {
      if (e.target === box || e.target.classList.contains('lightbox-close')) {
        this.closeLightbox();
      }
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') this.closeLightbox();
    });
  }

  openLightbox(src, alt) {
    if (!this.lightbox) return;
    const img = this.lightbox.querySelector('.lightbox-img');
    img.src = src;
    img.alt = alt || '';
    this.lightbox.classList.add('active');
    document.body.style.overflow = 'hidden';
  }

  closeLightbox() {
    if (!this.lightbox) return;
    this.lightbox.classList.remove('active');
    document.body.style.overflow = '';
  }

  bindPostEnhancements() {
    const postContent = document.getElementById('post-content');
    if (!postContent) return;

    postContent.querySelectorAll('img').forEach((img) => {
      img.style.cursor = 'zoom-in';
      img.addEventListener('click', () => this.openLightbox(img.src, img.alt));
    });

    postContent.querySelectorAll('pre').forEach((pre, index) => {
      if (pre.parentElement && pre.parentElement.classList.contains('code-block-body')) return;
      const wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';
      pre.parentNode.insertBefore(wrapper, pre);

      const toolbar = document.createElement('div');
      toolbar.className = 'code-block-toolbar';

      const meta = document.createElement('div');
      meta.className = 'code-block-meta';

      const actions = document.createElement('div');
      actions.className = 'code-block-actions';

      const code = pre.querySelector('code');
      const rawText = code ? code.textContent : pre.textContent;
      const normalizedText = String(rawText || '').replace(/\n+$/, '');
      const lineCount = normalizedText ? normalizedText.split(/\r?\n/).length : 1;
      const shouldCollapse = lineCount > 14;

      const body = document.createElement('div');
      body.className = 'code-block-body';
      const bodyId = `code-block-body-${index + 1}`;
      body.id = bodyId;
      body.setAttribute('role', 'region');
      body.setAttribute('aria-label', '代码内容');
      body.appendChild(pre);

      const toggleBtn = document.createElement('button');
      toggleBtn.type = 'button';
      toggleBtn.className = 'code-fold-button';
      toggleBtn.setAttribute('aria-controls', bodyId);

      const copyBtn = document.createElement('button');
      copyBtn.type = 'button';
      copyBtn.className = 'copy-button';
      copyBtn.textContent = '复制';

      function setCollapsed(collapsed) {
        wrapper.classList.toggle('is-collapsed', collapsed);
        toggleBtn.setAttribute('aria-expanded', String(!collapsed));
        toggleBtn.textContent = collapsed ? `展开代码 (${lineCount} 行)` : '收起代码';
      }

      toggleBtn.addEventListener('click', () => {
        setCollapsed(!wrapper.classList.contains('is-collapsed'));
      });

      copyBtn.addEventListener('click', async () => {
        const code = pre.querySelector('code');
        const text = code ? code.textContent : pre.textContent;
        try {
          await navigator.clipboard.writeText(text);
          copyBtn.textContent = '已复制';
        } catch {
          copyBtn.textContent = '失败';
        }
        setTimeout(() => { copyBtn.textContent = '复制'; }, 1600);
      });

      meta.textContent = `代码 ${lineCount} 行`;
      toggleBtn.textContent = shouldCollapse ? `展开代码 (${lineCount} 行)` : '收起代码';
      toggleBtn.setAttribute('aria-expanded', String(!shouldCollapse));
      setCollapsed(shouldCollapse);

      actions.appendChild(toggleBtn);
      actions.appendChild(copyBtn);
      toolbar.appendChild(meta);
      toolbar.appendChild(actions);
      wrapper.appendChild(toolbar);
      wrapper.appendChild(body);
    });
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new SiteUI();
});
