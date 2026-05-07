// Search functionality
class Search {
  constructor() {
    this.searchToggle = document.getElementById('searchToggle');
    this.searchContainer = document.getElementById('searchContainer');
    this.searchInput = document.getElementById('searchInput');
    this.searchResults = document.getElementById('searchResults');
    this.posts = [];
    this.isOpen = false;

    this.init();
  }

  async init() {
    // Load search index
    await this.loadSearchIndex();

    // Event listeners
    this.searchToggle.addEventListener('click', () => this.toggle());
    this.searchInput.addEventListener('input', (e) => this.search(e.target.value));
    this.searchInput.addEventListener('focus', () => this.search(this.searchInput.value));

    // Close on click outside
    document.addEventListener('click', (e) => {
      if (!this.searchContainer.contains(e.target) && !this.searchToggle.contains(e.target)) {
        this.close();
      }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        this.toggle();
      }
      if (e.key === 'Escape' && this.isOpen) {
        this.close();
      }
    });
  }

  async loadSearchIndex() {
    try {
      const response = await fetch('/search-index.json');
      if (response.ok) {
        this.posts = await response.json();
      }
    } catch (error) {
      console.warn('Search index not found. Build the site to enable search.');
      this.posts = [];
    }
  }

  toggle() {
    if (this.isOpen) {
      this.close();
    } else {
      this.open();
    }
  }

  open() {
    this.isOpen = true;
    this.searchContainer.classList.add('active');
    this.searchInput.focus();
  }

  close() {
    this.isOpen = false;
    this.searchContainer.classList.remove('active');
    this.searchInput.value = '';
    this.searchResults.innerHTML = '';
  }

  search(query) {
    if (!query.trim()) {
      this.searchResults.innerHTML = '';
      return;
    }

    const results = this.posts.filter(post => {
      const searchText = `${post.title} ${post.content} ${post.tags || ''}`.toLowerCase();
      return searchText.includes(query.toLowerCase());
    }).slice(0, 10);

    if (results.length === 0) {
      this.searchResults.innerHTML = '<div class="search-result-item"><div class="search-result-title">No results found</div></div>';
      return;
    }

    this.searchResults.innerHTML = results.map(post => `
      <a href="${post.url}" class="search-result-item">
        <div class="search-result-title">${this.highlight(query, post.title)}</div>
        <div class="search-result-excerpt">${this.highlight(query, post.excerpt || post.content.substring(0, 100))}</div>
      </a>
    `).join('');
  }

  highlight(query, text) {
    const regex = new RegExp(`(${this.escapeRegex(query)})`, 'gi');
    return text.replace(regex, '<mark>$1</mark>');
  }

  escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
}

// Initialize search
document.addEventListener('DOMContentLoaded', () => {
  new Search();
});

// Add smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth' });
    }
  });
});

// Lazy load images
if ('loading' in HTMLImageElement.prototype) {
  document.querySelectorAll('img[loading="lazy"]').forEach(img => {
    img.src = img.dataset.src;
  });
}

// Lightbox for images
class Lightbox {
  constructor() {
    this.lightbox = null;
    this.lightboxImg = null;
    this.lightboxClose = null;
    this.isOpen = false;
    this.init();
  }

  init() {
    // Create lightbox elements
    this.lightbox = document.createElement('div');
    this.lightbox.id = 'lightbox';
    this.lightbox.innerHTML = `
      <div class="lightbox-content">
        <button class="lightbox-close" aria-label="Close">&times;</button>
        <img class="lightbox-img" src="" alt="">
      </div>
    `;
    document.body.appendChild(this.lightbox);

    this.lightboxImg = this.lightbox.querySelector('.lightbox-img');
    this.lightboxClose = this.lightbox.querySelector('.lightbox-close');

    // Event listeners
    this.lightbox.addEventListener('click', (e) => {
      if (e.target === this.lightbox) this.close();
    });
    this.lightboxClose.addEventListener('click', () => this.close());
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isOpen) this.close();
    });
  }

  open(src, alt) {
    this.lightboxImg.src = src;
    this.lightboxImg.alt = alt || '';
    this.lightbox.classList.add('active');
    this.isOpen = true;
    document.body.style.overflow = 'hidden';
  }

  close() {
    this.lightbox.classList.remove('active');
    this.isOpen = false;
    document.body.style.overflow = '';
  }
}

const lightbox = new Lightbox();

// Add click handlers to all images in post content and footnotes
document.addEventListener('DOMContentLoaded', () => {
  const postContent = document.getElementById('post-content');
  if (postContent) {
    postContent.querySelectorAll('img').forEach(img => {
      img.style.cursor = 'zoom-in';
      img.addEventListener('click', () => {
        lightbox.open(img.src, img.alt);
      });
    });

    // Also handle images in footnotes
    postContent.querySelectorAll('aside.footnote img').forEach(img => {
      img.style.cursor = 'zoom-in';
      img.addEventListener('click', () => {
        lightbox.open(img.src, img.alt);
      });
    });
  }
});

// Copy button for code blocks
document.addEventListener('DOMContentLoaded', () => {
  const postContent = document.getElementById('post-content');
  if (postContent) {
    postContent.querySelectorAll('pre').forEach(pre => {
      const wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';
      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(pre);

      const copyBtn = document.createElement('button');
      copyBtn.className = 'copy-button';
      copyBtn.textContent = 'Copy';
      copyBtn.setAttribute('aria-label', 'Copy code');
      wrapper.appendChild(copyBtn);

      copyBtn.addEventListener('click', () => {
        const code = pre.querySelector('code');
        const text = code ? code.textContent : pre.textContent;
        navigator.clipboard.writeText(text).then(() => {
          copyBtn.textContent = 'Copied!';
          setTimeout(() => {
            copyBtn.textContent = 'Copy';
          }, 2000);
        }).catch(() => {
          copyBtn.textContent = 'Failed';
          setTimeout(() => {
            copyBtn.textContent = 'Copy';
          }, 2000);
        });
      });
    });
  }
});