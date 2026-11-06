/*
 * The models that can read shared screens (docs: notes-and-inference › Shared screens): found at runtime from NEAR AI,
 * never sent anything with content. Two public lists are joined:
 *
 *  - NEAR AI's catalog (`${NEAR_AI_BASE_URL}/models`): each model's ID, name and `input_modalities`. Only models that
 *    take images (and aren't marked not ready) count.
 *  - NEAR AI's registry of the models it runs itself (NEAR_AI_ENDPOINTS_URL, https://completions.near.ai/endpoints):
 *    which host serves each one's "direct completions", `{"endpoints": [{"domain", "models": [id]}]}`. A model's
 *    endpoint is that domain, on the registry's own scheme (https://qwen3-vl-30b.completions.near.ai for
 *    Qwen/Qwen3-VL-30B-A3B-Instruct). The catalog gives no endpoint, and the host isn't derived from the ID by any rule
 *    NEAR AI documents (most drop the vendor and turn dots into dashes, as glm-5-3-flash for z-ai/glm-5.3-flash; many
 *    don't), so a model the registry doesn't list isn't offered: requests only ever go sealed to a model's own endpoint.
 *
 * Being listed isn't being trusted: every request to the model chosen is still sealed to the key its attestation
 * vouches for, checked as the notes model's is (evidence.ts). The notes model (NEAR_AI_NOTES_MODEL at
 * NEAR_AI_NOTES_URL) always counts as able to read screens; it's the choice "null". The joined list is kept for ten
 * minutes, and a failure to fetch it for 30 seconds. Log lines carry counts, statuses and model IDs.
 */
import type { VisionModel } from '@alcove/api';
import { z } from 'zod';
import type { Config } from '../config';
import { UnansweredError, UnverifiedError } from '../errors';
import { verified } from '../evidence';
import { inference, ready } from '../inference';
import { log } from '../log';

const mlog = log.child({ component: 'models' });

/** How long the joined list is used before it's fetched again. */
const TTL_MS = 10 * 60_000;
/** How long a failure to fetch it stands before the next ask tries again. */
const REFUSED_MS = 30_000;

/** A model that can read shared screens: its ID, its name, and its own endpoint. */
export interface VisionModelEntry { id: string; name: string; url: string }

/** One catalog entry, as much of it as is used; anything else in it is ignored, and an entry without an ID is skipped. */
const CatalogEntry = z.object({
  id: z.string().min(1),
  name: z.string().nullish().catch(null),
  input_modalities: z.array(z.string()).nullish().catch(null),
  architecture: z.object({ inputModalities: z.array(z.string()).nullish().catch(null) }).nullish().catch(null),
  is_ready: z.boolean().nullish().catch(null)
});
const Catalog = z.object({ data: z.array(z.unknown()) });
const Endpoint = z.object({ domain: z.string().regex(/^[A-Za-z0-9.-]+(?::\d+)?$/), models: z.array(z.string()) });
const Registry = z.object({ endpoints: z.array(z.unknown()) });

const kept = new Map<string, { until: number; models: VisionModelEntry[] }>();
const refused = new Map<string, { until: number; error: UnansweredError }>();
const running = new Map<string, Promise<VisionModelEntry[]>>();

async function fetchJson(url: string, apiKey: string | undefined, what: string): Promise<unknown> {
  const t0 = performance.now();
  let res: Response;
  try {
    res = await fetch(url, { headers: { accept: 'application/json', ...(apiKey ? { authorization: `Bearer ${apiKey}` } : {}) }, signal: AbortSignal.timeout(10_000) });
  } catch (e) {
    throw new UnansweredError('NEAR AI’s list of models can’t be reached right now.', { cause: e, context: { what, ms: Math.round(performance.now() - t0) } });
  }
  if (!res.ok) {
    await res.body?.cancel().catch(() => undefined);
    throw new UnansweredError('NEAR AI’s list of models can’t be reached right now.', { context: { what, status: res.status, ms: Math.round(performance.now() - t0) } });
  }
  return res.json().catch(() => null);
}

async function load(cfg: Config): Promise<VisionModelEntry[]> {
  const t0 = performance.now();
  const [catalog, registry] = await Promise.all([
    fetchJson(`${cfg.NEAR_AI_BASE_URL.replace(/\/$/, '')}/models`, cfg.NEAR_AI_API_KEY, 'catalog'),
    fetchJson(cfg.NEAR_AI_ENDPOINTS_URL, undefined, 'endpoints')
  ]);
  const c = Catalog.safeParse(catalog);
  const r = Registry.safeParse(registry);
  if (!c.success || !r.success) {
    throw new UnansweredError('NEAR AI’s list of models came back in a shape Alcove doesn’t know.', { context: { what: 'models', catalog: c.success, endpoints: r.success } });
  }
  const scheme = new URL(cfg.NEAR_AI_ENDPOINTS_URL).protocol;
  const hosts = new Map<string, string>();
  for (const x of r.data.endpoints) {
    const e = Endpoint.safeParse(x);
    if (e.success) for (const m of e.data.models) if (!hosts.has(m)) hosts.set(m, `${scheme}//${e.data.domain}`);
  }
  const entries = c.data.data.flatMap((x) => { const e = CatalogEntry.safeParse(x); return e.success ? [e.data] : []; });
  const named = (id: string) => entries.find((e) => e.id === id)?.name?.trim() || id;
  const others = entries
    .filter((e) => e.id !== cfg.NEAR_AI_NOTES_MODEL && e.is_ready !== false && [...(e.input_modalities ?? []), ...(e.architecture?.inputModalities ?? [])].includes('image'))
    .flatMap((e): VisionModelEntry[] => { const url = hosts.get(e.id); return url ? [{ id: e.id, name: named(e.id), url }] : []; })
    .sort((a, b) => a.name.localeCompare(b.name));
  const models = [{ id: cfg.NEAR_AI_NOTES_MODEL, name: named(cfg.NEAR_AI_NOTES_MODEL), url: cfg.NEAR_AI_NOTES_URL }, ...others];
  mlog.info('vision models listed', { catalog: entries.length, endpoints: hosts.size, vision: others.length, models: others.map((m) => m.id), ms: Math.round(performance.now() - t0) });
  return models;
}

/**
 * Every model that can read shared screens, the notes model first: from NEAR AI's catalog and its registry of
 * endpoints, kept for ten minutes. Without NEAR AI configured there are none. NEAR AI not answering is thrown.
 */
export async function visionModels(cfg: Config): Promise<VisionModelEntry[]> {
  if (!inference(cfg).configured) return [];
  const id = `${cfg.NEAR_AI_BASE_URL}|${cfg.NEAR_AI_ENDPOINTS_URL}|${cfg.NEAR_AI_NOTES_MODEL}`;
  const hit = kept.get(id);
  if (hit && hit.until > Date.now()) return hit.models;
  const no = refused.get(id);
  if (no && no.until > Date.now()) throw no.error;
  const inFlight = running.get(id);
  if (inFlight) return inFlight;
  const made = load(cfg)
    .then((models) => { kept.set(id, { until: Date.now() + TTL_MS, models }); refused.delete(id); return models; }, (e: unknown) => {
      if (e instanceof UnansweredError) {
        mlog.warn('vision models not listed', { err: e });
        refused.set(id, { until: Date.now() + REFUSED_MS, error: e });
      }
      throw e;
    })
    .finally(() => running.delete(id));
  running.set(id, made);
  return made;
}

/** The models someone can choose to read shared screens besides the notes model (the app offers that one as the default). */
export async function choices(cfg: Config): Promise<VisionModel[]> {
  return (await visionModels(cfg)).filter((m) => m.id !== cfg.NEAR_AI_NOTES_MODEL).map((m) => ({ id: m.id, name: m.name }));
}

/**
 * The model that reads a note's shared screens, and its endpoint: the one chosen, or the notes model for null. A model
 * chosen that NEAR AI no longer lists, or whose evidence doesn't check out, can't be sent anything, so the notes model
 * reads them instead (and the note's list of verified models says so): one model never holds the notes up. If the
 * notes model can't be verified either, its requests say so and the notes wait. NEAR AI's lists not answering is
 * thrown, so the notes wait for them.
 */
export async function screenModel(cfg: Config, chosen: string | null): Promise<{ model: string; url: string }> {
  const notes = { model: cfg.NEAR_AI_NOTES_MODEL, url: cfg.NEAR_AI_NOTES_URL };
  if (chosen === null || chosen === cfg.NEAR_AI_NOTES_MODEL || !ready(cfg)) return notes;
  const found = (await visionModels(cfg)).find((m) => m.id === chosen);
  if (!found) {
    mlog.warn('screen model unavailable', { model: chosen, instead: notes.model });
    return notes;
  }
  try {
    await verified(cfg, found.id, 'notes');
  } catch (e) {
    if (!(e instanceof UnverifiedError)) throw e;
    mlog.warn('screen model unverified', { model: found.id, instead: notes.model, err: e });
    return notes;
  }
  return { model: found.id, url: found.url };
}
