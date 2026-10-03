(function (root) {
  'use strict';
  var names = new Intl.Collator('zh-CN', { numeric: true, sensitivity: 'base' });

  function timestamp(value, fallback) {
    var parsed = value ? new Date(value).getTime() : NaN;
    if (Number.isFinite(parsed)) return parsed;
    var backup = fallback ? new Date(fallback).getTime() : NaN;
    return Number.isFinite(backup) ? backup : 0;
  }

  function compare(a, b, mode) {
    var byName = names.compare(a.title, b.title) || names.compare(a.url, b.url);
    if (mode === 'name') return byName;
    var field = mode === 'created' ? 'created' : 'updated';
    return b[field] - a[field] || b.created - a.created || byName;
  }

  if (typeof module === 'object' && module.exports) {
    module.exports = { timestamp: timestamp, compare: compare };
    return;
  }
  var list = root.document.querySelector('[data-home-posts]');
  var control = root.document.getElementById('post-sort');
  if (!list || !control) return;
  var posts = Array.from(list.children).map(function (element) {
    return {
      element: element,
      title: element.querySelector('.post-list-title').textContent.trim(),
      url: element.querySelector('.post-list-title a').getAttribute('href'),
      created: Number(element.dataset.created),
      updated: Number(element.dataset.updated)
    };
  });
  control.value = 'updated';
  control.disabled = false;
  control.addEventListener('change', function () {
    posts.sort(function (a, b) { return compare(a, b, control.value); });
    posts.forEach(function (post) { list.appendChild(post.element); });
  });
})(typeof window === 'undefined' ? null : window);
