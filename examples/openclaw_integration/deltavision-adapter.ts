/**
 * DeltaVision adapter for OpenClaw.
 *
 * OpenClaw routes every browser screenshot through
 *   src/agents/tools/common.ts::imageResultFromFile
 * before it hits the model. This file exports a drop-in replacement that
 * preprocesses the screenshot through a local DeltaVision HTTP sidecar.
 *
 * HOW TO INSTALL
 * --------------
 * 1. Run the DeltaVision sidecar on localhost:
 *      python /path/to/deltavision/server.py --port 9000
 * 2. Copy this file into your OpenClaw checkout at:
 *      src/agents/tools/deltavision-adapter.ts
 * 3. Replace the direct import in callsites:
 *      // before:
 *      import { imageResultFromFile } from "./common";
 *      // after:
 *      import { imageResultFromFile } from "./deltavision-adapter";
 *
 *    OR use the test-injection hook already exposed in
 *    extensions/browser/src/browser-tool.actions.ts:
 *      browserToolActionDeps.imageResultFromFile = dvImageResultFromFile;
 *
 * That's it. Every browser screenshot is now DeltaVision-gated.
 *
 * The sidecar holds ONE observer instance per process. Call POST /reset
 * at the start of each new agent run to drop the prior t0 anchor.
 */

import { imageResult, imageResultFromFile as originalImageResultFromFile } from "./common";
import type { AgentToolResult, ImageSanitizationLimits } from "./common";
import * as fs from "fs/promises";

const DV_ENDPOINT = process.env.DELTAVISION_URL ?? "http://127.0.0.1:9000";
const DV_FORMAT = process.env.DELTAVISION_FORMAT ?? "raw";
const DV_ENABLED = (process.env.DELTAVISION_ENABLED ?? "true") !== "false";

// One-time warmup check so we fail loudly if the sidecar isn't reachable.
let warmedUp = false;

async function warmup(): Promise<boolean> {
  if (warmedUp) return true;
  try {
    const r = await fetch(`${DV_ENDPOINT}/health`);
    const ok = r.ok;
    warmedUp = ok;
    return ok;
  } catch {
    return false;
  }
}

export async function dvResetSession(): Promise<void> {
  try {
    await fetch(`${DV_ENDPOINT}/reset`, { method: "POST" });
  } catch {
    // best-effort
  }
}

/**
 * Drop-in replacement for src/agents/tools/common.ts::imageResultFromFile.
 * Falls back to the unmodified path if the sidecar is down or DV is disabled.
 */
export async function imageResultFromFile(params: {
  label: string;
  path: string;
  extraText?: string;
  details?: Record<string, unknown>;
  imageSanitization?: ImageSanitizationLimits;
}): Promise<AgentToolResult<unknown>> {
  if (!DV_ENABLED || !(await warmup())) {
    return originalImageResultFromFile(params);
  }

  const buf = await fs.readFile(params.path);
  const b64 = buf.toString("base64");

  // Infer url + last_action from OpenClaw's details metadata if present.
  const url = (params.details?.url as string | undefined) ?? null;
  const lastAction = (params.details?.action as string | undefined) ?? null;

  let dvResult: any;
  try {
    const r = await fetch(`${DV_ENDPOINT}/observe`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        screenshot_b64: b64,
        url,
        last_action: lastAction,
        format: DV_FORMAT,
      }),
    });
    if (!r.ok) {
      // If DV returns an error, fall through to the original behavior.
      console.warn(`[deltavision] sidecar returned ${r.status}; passing through`);
      return originalImageResultFromFile(params);
    }
    dvResult = await r.json();
  } catch (e) {
    console.warn(`[deltavision] sidecar unreachable, passing through: ${e}`);
    return originalImageResultFromFile(params);
  }

  // On NEW_PAGE we send a full frame; on DELTA we send a composited image.
  // Both paths return a single base64 image we can feed to imageResult().
  const dvB64 =
    dvResult.payload?.frame_b64 ?? // format=raw full-frame
    dvResult.payload?.thumbnail_b64 ?? // format=raw delta (thumbnail only)
    null;

  if (!dvB64) {
    // Unexpected response shape — pass through.
    return originalImageResultFromFile(params);
  }

  // Emit rich telemetry so OpenClaw users can see DV's decisions in logs.
  const augmentedDetails = {
    ...params.details,
    deltavision: {
      obs_type: dvResult.obs_type,
      trigger: dvResult.trigger,
      diff_ratio: dvResult.diff_ratio,
      phash_distance: dvResult.phash_distance,
      anchor_score: dvResult.anchor_score,
      estimated_tokens: dvResult.estimated_tokens,
    },
  };

  return imageResult({
    label: params.label,
    path: params.path,
    base64: dvB64,
    mimeType: "image/png",
    extraText: params.extraText,
    details: augmentedDetails,
    imageSanitization: params.imageSanitization,
  });
}
