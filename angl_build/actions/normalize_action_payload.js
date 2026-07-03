function isPlainObject(v) {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function coerceInt(v) {
  if (v === undefined || v === null) return null;
  if (typeof v === 'number') {
    return Number.isFinite(v) ? Math.trunc(v) : null;
  }
  if (typeof v === 'string') {
    if (v.trim() === '') return null;
    const n = Number(v);
    return Number.isFinite(n) ? Math.trunc(n) : null;
  }
  return null;
}

function orNull(v) {
  return v === undefined || v === null ? null : v;
}

const NATIVE_TYPES = ['click', 'type', 'scroll', 'key', 'wait', 'done'];

const UI_TARS_ACTION_MAP = {
  left_click: 'click',
  click: 'click',
  right_click: 'click',
  double_click: 'click',
  type: 'type',
  scroll: 'scroll',
  key: 'key',
  press: 'key',
  wait: 'wait',
  finished: 'done',
  done: 'done',
};

function normalizeNative(raw) {
  const type = raw.type;
  if (typeof type !== 'string' || !NATIVE_TYPES.includes(type)) return null;

  return {
    type,
    x: coerceInt(raw.x),
    y: coerceInt(raw.y),
    text: orNull(raw.text),
    direction: orNull(raw.direction),
    amount: coerceInt(raw.amount),
    key: orNull(raw.key),
    duration_ms: coerceInt(raw.duration_ms),
  };
}

function normalizeUiTars(raw) {
  const action = raw.action;
  if (typeof action !== 'string') return null;

  const type = UI_TARS_ACTION_MAP[action];
  if (!type) return null;

  let x = null;
  let y = null;

  if (raw.coordinate !== undefined && raw.coordinate !== null) {
    if (!Array.isArray(raw.coordinate)) return null;
    const [cx, cy] = raw.coordinate;

    if (cx !== undefined) {
      x = coerceInt(cx);
      if (x === null) return null;
    }
    if (cy !== undefined) {
      y = coerceInt(cy);
      if (y === null) return null;
    }
  }

  return {
    type,
    x,
    y,
    text: orNull(raw.text),
    direction: orNull(raw.direction),
    amount: coerceInt(raw.amount),
    key: orNull(raw.key),
    duration_ms: coerceInt(raw.duration_ms),
  };
}

function normalizeActionPayload(raw) {
  if (!isPlainObject(raw)) return null;

  if (raw.type !== undefined) {
    return normalizeNative(raw);
  }
  if (raw.action !== undefined) {
    return normalizeUiTars(raw);
  }
  return null;
}

function readStdin() {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => {
      data += chunk;
    });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

(async () => {
  try {
    const input = await readStdin();
    const req = JSON.parse(input);
    const rawAction = req.args[0];
    const value = normalizeActionPayload(rawAction);
    process.stdout.write(JSON.stringify({ ok: true, value }));
  } catch (err) {
    process.stdout.write(JSON.stringify({ ok: false, error: String(err && err.message ? err.message : err) }));
  }
})();
