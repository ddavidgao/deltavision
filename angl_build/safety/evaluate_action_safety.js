function getDomainParts(url) {
  let hostname;
  try {
    hostname = new URL(url).hostname.toLowerCase();
  } catch (e) {
    hostname = String(url).toLowerCase();
  }
  const labels = hostname.split('.').filter(Boolean);
  const baseDomain = labels.length >= 2 ? labels.slice(-2).join('.') : hostname;
  return { hostname, baseDomain };
}

function evaluateActionSafety(action, policy, currentUrl, pageContext) {
  action = action || {};
  policy = policy || {};

  if (action.type === 'type' && policy.block_credential_entry) {
    const context = String(pageContext || '').toLowerCase();
    const patterns = [
      'password', 'ssn', 'social.security', 'credit.card',
      'card.number', 'cvv', 'cvc', 'bank.account', 'routing.number'
    ];
    for (const p of patterns) {
      const re = new RegExp(p);
      if (re.test(context)) {
        return {
          allowed: false,
          reason: `Blocked typing into potential credential field (matched: ${p})`,
          severity: 'block'
        };
      }
    }

    const text = String(action.text || '');
    const cleaned = text.replace(/[ -]/g, '');
    const isAllDigits = /^\d+$/.test(cleaned);
    if (isAllDigits) {
      const len = cleaned.length;
      const isCredential =
        len === 9 ||
        (len >= 13 && len <= 19) ||
        ((len === 3 || len === 4) && text.length <= 4);
      if (isCredential) {
        return {
          allowed: false,
          reason: 'Blocked: text looks like a credential (SSN, card number, etc.)',
          severity: 'block'
        };
      }
    }
  }

  if (currentUrl && currentUrl.length > 0) {
    const { hostname, baseDomain } = getDomainParts(currentUrl);
    const lowerUrl = currentUrl.toLowerCase();

    if (policy.block_url_shorteners) {
      const shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly'];
      for (const s of shorteners) {
        if (hostname.includes(s)) {
          return {
            allowed: false,
            reason: `Blocked: URL shortener detected (${s}). Cannot verify destination.`,
            severity: 'block'
          };
        }
      }
    }

    if (policy.allowed_domains !== null && policy.allowed_domains !== undefined) {
      const allowedList = policy.allowed_domains;
      const isAllowed = allowedList.includes(hostname) || allowedList.includes(baseDomain);
      if (!isAllowed) {
        return {
          allowed: false,
          reason: `Blocked: domain ${hostname} not in allowlist`,
          severity: 'warn'
        };
      }
    }

    const isEduOrGov = lowerUrl.includes('.edu') || lowerUrl.includes('.gov');
    const suspiciousPatterns = [];

    if (!isEduOrGov) {
      suspiciousPatterns.push(['.*login.*\\..*', /login.*\./]);
    }
    suspiciousPatterns.push(['.*signin.*', /signin/]);
    suspiciousPatterns.push(['.*account.*verify.*', /account.*verify/]);
    suspiciousPatterns.push(['.*password.*reset.*', /password.*reset/]);
    suspiciousPatterns.push(['.*\\.ru/.*', /\.ru\//]);
    suspiciousPatterns.push(['.*\\.cn/.*login.*', /\.cn\/.*login/]);

    for (const [patternStr, regex] of suspiciousPatterns) {
      if (regex.test(lowerUrl)) {
        return {
          allowed: false,
          reason: `Blocked: suspicious URL pattern (${patternStr})`,
          severity: 'warn'
        };
      }
    }
  }

  if (action.type === 'type') {
    const text = String(action.text || '');
    const maxLen = policy.max_type_length;
    if (text.length > maxLen) {
      return {
        allowed: false,
        reason: `Blocked: type action too long (${text.length} chars > ${maxLen})`,
        severity: 'warn'
      };
    }
  }

  if (action.type === 'click') {
    const x = action.x;
    const y = action.y;
    if (x !== null && x !== undefined && y !== null && y !== undefined) {
      if (x < 0 || y < 0) {
        return {
          allowed: false,
          reason: 'Blocked: negative click coordinates',
          severity: 'warn'
        };
      }
    }
  }

  return { allowed: true, reason: '', severity: 'info' };
}

function main() {
  let input = '';
  process.stdin.on('data', (chunk) => { input += chunk; });
  process.stdin.on('end', () => {
    try {
      const req = JSON.parse(input);
      const args = req.args || [];
      const action = args[0];
      const policy = args[1];
      const currentUrl = args[2];
      const pageContext = args[3];
      const value = evaluateActionSafety(action, policy, currentUrl, pageContext);
      process.stdout.write(JSON.stringify({ ok: true, value: value }));
    } catch (err) {
      process.stdout.write(JSON.stringify({ ok: false, error: String(err && err.message ? err.message : err) }));
    }
  });
}

main();
