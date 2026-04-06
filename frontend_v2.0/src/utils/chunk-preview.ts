/**
 * Mirrors backend/services/ingestion.py chunk_text logic for live Chunker preview.
 */

export type ChunkStrategy = 'fixed' | 'overlapping';

function normalizeText(text: string): string {
  return text.replace(/\n{3,}/g, '\n\n').trim();
}

function fixedChunk(text: string, chunkSize: number): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  const chunks: string[] = [];
  let current: string[] = [];
  let currentLen = 0;

  for (const word of words) {
    const wordLen = word.length + (current.length ? 1 : 0);
    if (currentLen + wordLen > chunkSize && current.length) {
      chunks.push(current.join(' '));
      current = [word];
      currentLen = word.length;
    } else {
      current.push(word);
      currentLen += wordLen;
    }
  }
  if (current.length) chunks.push(current.join(' '));
  return chunks;
}

function overlappingChunk(text: string, chunkSize: number, overlap: number): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  const chunks: string[] = [];
  let start = 0;

  while (start < words.length) {
    let end = start;
    let currentLen = 0;
    while (end < words.length && currentLen + words[end].length + (end > start ? 1 : 0) <= chunkSize) {
      currentLen += words[end].length + (end > start ? 1 : 0);
      end++;
    }
    if (end === start) end = start + 1;

    chunks.push(words.slice(start, end).join(' '));

    if (end >= words.length) break;

    let overlapWords = 0;
    let overlapLen = 0;
    for (let i = end - 1; i >= start; i--) {
      if (overlapLen + words[i].length > overlap) break;
      overlapLen += words[i].length + 1;
      overlapWords++;
    }
    const oldStart = start;
    start = end - overlapWords;
    if (start <= oldStart) start = oldStart + 1;
  }

  return chunks;
}

export function previewChunks(
  rawText: string,
  strategy: ChunkStrategy,
  chunkSize: number,
  overlap: number,
): string[] {
  const text = normalizeText(rawText);
  if (!text) return [];
  const size = Math.max(50, Math.min(5000, chunkSize));
  const ov = Math.max(0, Math.min(500, overlap));

  if (strategy === 'fixed') {
    return fixedChunk(text, size);
  }
  return overlappingChunk(text, size, ov);
}

export function countBoundaryOverlapTokens(leftText: string, rightText: string): number {
  const left = leftText.split(/\s+/).filter(Boolean);
  const right = rightText.split(/\s+/).filter(Boolean);
  const maxK = Math.min(left.length, right.length);
  for (let k = maxK; k > 0; k--) {
    let all = true;
    for (let i = 0; i < k; i++) {
      if (left[left.length - k + i] !== right[i]) {
        all = false;
        break;
      }
    }
    if (all) return k;
  }
  return 0;
}
