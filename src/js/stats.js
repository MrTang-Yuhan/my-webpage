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

  function render(data) {
    // A browser cannot identify IPs or measure traffic from other visitors.
    // Never present localStorage counters as site-wide or IP-based statistics.
    if (!data || !data.persistent || data.version !== 2) {
      ['allTotal', 'allToday', 'uniqueTotal', 'uniqueToday', 'pages'].forEach(function (key) { setText(key, '—'); });
      setText('status', '全站统计暂不可用，请稍后刷新。');
      setText('period', '');
      root.querySelector('[data-stat="daily"]').innerHTML = '<p class="stats-empty">暂无可用的全站数据</p>';
      root.querySelector('[data-stat="topPages"]').innerHTML = '<li class="stats-empty">文章访问统计暂不可用</li>';
      setText('updated', '');
      return;
    }
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
    setText('allTotal', number.format(count(data.allTotal)));
    setText('allToday', number.format(count(data.allToday)));
    setText('uniqueTotal', number.format(count(data.uniqueTotal)));
    setText('uniqueToday', number.format(count(data.uniqueToday)));
    setText('pages', number.format(count(data.pageCount == null ? topPages.length : data.pageCount)));
    setText('status', data.missingIpViews ? '部分访问未能识别 IP；不同 IP 统计仅包含已识别的访问。' : '');
    var period = data.startedAt
      ? '统计起点：' + new Date(data.startedAt).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai' }) + '。今日按北京时间 00:00 划分。'
      : '等待新口径下的首次访问。今日按北京时间 00:00 划分。';
    if (data.legacy) period += ' 旧版每日去重记录已保留（' + number.format(count(data.legacy.total)) + ' 次），无法还原重复访问和跨日 IP 去重，未混入新统计。';
    setText('period', period);

    var daily = Array.isArray(data.daily) ? data.daily.slice(-14) : [];
    var max = Math.max.apply(null, daily.map(function (item) { return count(item.count); }).concat([1]));
    root.querySelector('[data-stat="daily"]').innerHTML = daily.map(function (item) {
      var value = count(item.count);
      var height = Math.round((value / max) * 100);
      return '<div class="stats-bar-item" title="' + escapeHtml(item.date) + '：所有 IP ' + number.format(value) + ' 次，不同 IP ' + number.format(count(item.unique)) + ' 个">' +
        '<span class="stats-bar-value">' + compactNumber.format(value) + '</span>' +
        '<span class="stats-bar-track"><span class="stats-bar" style="height:' + height + '%"></span></span>' +
        '<span class="stats-bar-label">' + escapeHtml(String(item.date || '').slice(5)) + '</span></div>';
    }).join('');

    root.querySelector('[data-stat="topPages"]').innerHTML = topPages.length ? topPages.map(function (item) {
      return '<li>' + pageLink(item) + '<span class="stats-ranking-count">' +
        number.format(count(item.count)) + '</span></li>';
    }).join('') : '<li class="stats-empty">暂无文章访问记录</li>';
    var updated = data.updatedAt ? new Date(data.updatedAt) : null;
    setText('updated', updated && !Number.isNaN(updated.getTime()) ? '更新于 ' + updated.toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai' }) : '暂无访问记录');
  }

  fetch('/api/stats', { cache: 'no-store' })
    .then(function (response) { if (!response.ok) throw new Error('stats unavailable'); return response.json(); })
    .then(render)
    .catch(function () { render(null); });
})();
