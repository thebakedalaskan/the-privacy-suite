/*
 * Choosing the model that reads shared screens for the notes: the organizer's in a call, a default in Settings, and
 * again when notes are written over. The notes model is always a choice (null on the wire); the others are the
 * verified private vision models the API lists (`useVisionModels`).
 */
import type { VisionModel } from '@alcove/api';
import type { SelectOption } from '@alcove/ui';

/** The notes model, as a `Select` value: model IDs are never empty. */
export const NOTES_MODEL = '';
export const NOTES_MODEL_LABEL = 'Standard (notes model)';

/** The choices, the notes model first. A chosen model the API no longer lists stays shown by its ID, so the choice never looks blank. */
export function screenModelOptions(models: readonly VisionModel[], current: string | null): SelectOption<string>[] {
  const listed = models.map((m) => ({ value: m.id, label: m.name }));
  const missing = current !== null && !models.some((m) => m.id === current) ? [{ value: current, label: current }] : [];
  return [{ value: NOTES_MODEL, label: NOTES_MODEL_LABEL }, ...listed, ...missing];
}
/** A `Select` value as the model to send: null for the notes model. */
export const screenModelOf = (value: string): string | null => (value === NOTES_MODEL ? null : value);
/** A model to send as a `Select` value. */
export const screenModelValue = (model: string | null): string => model ?? NOTES_MODEL;
