(function () {
  var root = document.querySelector('[data-stats-page]');
  if (!root) return;
  var number = new Intl.NumberFormat('zh-CN');
  var compactNumber = new Intl.NumberFormat('zh-CN', { notation: 'compact', maximumFractionDigits: 1 });
  var titles = Object.create(null);
  var titleTemplate = document.getElementById('stats-page-titles');

  function pathKey(path) {
    // Decode per segment so an encoded slash does not become a path separator.
    return String(path).split(/[?#]/)[0].split('/').map(function (part) {
      try { return encodeURIComponent(decodeURIComponent(part)); }
      catch (_) { return part; }
    }).join('/').replace(/\/+$/, '') || '/';
  }

  if (titleTemplate) {
    titleTemplate.content.querySelectorAll('a').forEach(function (link) {
      titles[pathKey(link.getAttribute('href'))] = link.textContent.trim();
    });
  }

  function count(value) {
    var result = Number(value);
    return Number.isFinite(result) && result >= 0 ? result : 0;
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, function (char) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' })[char];
    });
  }

  function setText(name, value) {
    var node = root.querySelector('[data-stat="' + name + '"]');
    if (node) node.textContent = value;
  }

  function readablePath(path) {
    return path.split('/').map(function (part) {
      try { return decodeURIComponent(part); }
      catch (_) { return part; }
    }).join('/');
  }

  function pageLink(item) {
    var path = String(item.path || '/');
    // Historical data is untrusted; only generate links to pages on this site.
    if (!path.startsWith('/') || path.startsWith('//') || /[\\\u0000-\u0020]/.test(path)) {
      return '<span class="stats-ranking-title">无效页面地址</span>';
    }
    var key = pathKey(path);
    var readable = readablePath(path);
    var label = titles[key];
    if (!label && item.title && pathKey(item.title) !== key && !String(item.title).startsWith('/')) {
      label = String(item.title);
    }
    if (!label) {
      // Removed or unindexed pages retain a readable leaf name and the original link.
      var segments = readable.split(/[?#]/)[0].split('/').filter(Boolean);
      label = segments.length ? segments[segments.length - 1] : '首页';
    }
    return '<a href="' + escapeHtml(path) + '" title="' + escapeHtml(label + '\n' + readable) +
      '"><span class="stats-ranking-title">' + escapeHtml(label) + '</span></a>';
  }

  function localFallback() {
    var value;
    try { value = JSON.parse(localStorage.getItem('site-local-stats-v1') || '{}') || {}; }
    catch (_) { value = {}; }
    var days = value.days || {};
    var pages = value.pages || {};
    var todayKey = new Date().toISOString().slice(0, 10);
    return {
      total: count(value.total),
      today: count(days[todayKey]),
      daily: Array.from({ length: 14 }, function (_, index) {
        var date = new Date();
        date.setUTCDate(date.getUTCDate() - (13 - index));
        var key = date.toISOString().slice(0, 10);
        return { date: key, count: count(days[key]) };
      }),
      pageCount: Object.keys(pages).length,
      pageViews: pages,
      topPages: Object.entries(pages).map(function (entry) {
        return { path: entry[0], count: count(entry[1]) };
      }).sort(function (a, b) { return b.count - a.count || a.path.localeCompare(b.path); }).slice(0, 10),
      updatedAt: value.updatedAt || null
    };
  }

  function render(data) {
    // The API already supplies today/daily; it does not return the raw days map.
    if (!data || !data.persistent) data = localFallback();
    var candidates = data.pageViews && typeof data.pageViews === 'object'
      ? Object.entries(data.pageViews).map(function (entry) { return { path: entry[0], count: count(entry[1]) }; })
      : (Array.isArray(data.topPages) ? data.topPages : []);
    // The build-time index contains only published article pages. Merge encoded
    // and decoded forms before sorting, and apply the limit after filtering.
    var articles = Object.create(null);
    candidates.forEach(function (item) {
      if (!item || !item.path) return;
      var key = pathKey(item.path);
      if (!titles[key] || count(item.count) === 0) return;
      if (!articles[key]) articles[key] = { path: item.path, count: 0 };
      articles[key].count += count(item.count);
    });
    var topPages = Object.values(articles).sort(function (a, b) {
      return b.count - a.count || a.path.localeCompare(b.path);
    }).slice(0, 10);
    setText('total', number.format(count(data.total)));
    setText('today', number.format(count(data.today)));
    setText('pages', number.format(count(data.pageCount == null ? topPages.length : data.pageCount)));

    var daily = Array.isArray(data.daily) ? data.daily.slice(-14) : [];
    var max = Math.max.apply(null, daily.map(function (item) { return count(item.count); }).concat([1]));
    root.querySelector('[data-stat="daily"]').innerHTML = daily.map(function (item) {
      var value = count(item.count);
      var height = Math.round((value / max) * 100);
      return '<div class="stats-bar-item" title="' + escapeHtml(item.date) + ': ' + number.format(value) + '">' +
        '<span class="stats-bar-value">' + compactNumber.format(value) + '</span>' +
        '<span class="stats-bar-track"><span class="stats-bar" style="height:' + height + '%"></span></span>' +
        '<span class="stats-bar-label">' + escapeHtml(String(item.date || '').slice(5)) + '</span></div>';
    }).join('');

    root.querySelector('[data-stat="topPages"]').innerHTML = topPages.length ? topPages.map(function (item) {
      return '<li>' + pageLink(item) + '<span class="stats-ranking-count">' +
        number.format(count(item.count)) + '</span></li>';
    }).join('') : '<li class="stats-empty">暂无文章访问记录</li>';
    var updated = data.updatedAt ? new Date(data.updatedAt) : null;
    setText('updated', updated && !Number.isNaN(updated.getTime()) ? '更新于 ' + updated.toLocaleString('zh-CN') : '');
  }

  fetch('/api/stats', { cache: 'no-store' })
    .then(function (response) { if (!response.ok) throw new Error('stats unavailable'); return response.json(); })
    .then(render)
    .catch(function () { render(null); });
})();
