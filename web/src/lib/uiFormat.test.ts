import { describe, expect, it } from 'vitest';
import { formatClock, groupByWord, progressPercent, withCurrent } from './uiFormat';

describe('progressPercent', () => {
  it('is 0 when there is nothing to do', () => {
    expect(progressPercent(0, 0)).toBe(0);
    expect(progressPercent(3, 0)).toBe(0);
  });

  it('rounds to a whole percent', () => {
    expect(progressPercent(1, 3)).toBe(33);
    expect(progressPercent(2, 3)).toBe(67);
  });

  it('stays between 0 and 100', () => {
    expect(progressPercent(12, 10)).toBe(100);
    expect(progressPercent(-1, 10)).toBe(0);
  });
});

describe('groupByWord', () => {
  it('groups takes by word in first-seen order, keeping take order', () => {
    const takes = [
      { id: 1, word: 'su' },
      { id: 2, word: 'ekmek' },
      { id: 3, word: 'su' },
    ];
    expect(groupByWord(takes)).toEqual([
      ['su', [takes[0], takes[2]]],
      ['ekmek', [takes[1]]],
    ]);
  });

  it('returns an empty list for no takes', () => {
    expect(groupByWord([])).toEqual([]);
  });
});

describe('formatClock', () => {
  it('formats HH:MM:SS with zero padding', () => {
    expect(formatClock(new Date(2026, 9, 13, 9, 5, 7))).toBe('09:05:07');
    expect(formatClock(new Date(2026, 9, 13, 23, 59, 0))).toBe('23:59:00');
  });
});

describe('withCurrent', () => {
  it('keeps the options when the current value is one of them', () => {
    expect(withCurrent(['Furkan', 'base'], 'base')).toEqual(['Furkan', 'base']);
  });

  it('adds a server value missing from the list so it shows as selected', () => {
    expect(withCurrent(['Furkan', 'base', 'large'], 'whisper')).toEqual(['Furkan', 'base', 'large', 'whisper']);
  });

  it('ignores an empty current value', () => {
    expect(withCurrent(['lora'], '')).toEqual(['lora']);
  });
});
