import { Meter, Tafila } from './types';

// The fundamental, repeating sequence of syllabic units.
// '//0' represents a watid majmu'
// '/0' represents a sabab khafif
export const ATOMIC_SEQUENCE = ['//0', '/0', '//0', '/0', '/0', '//0', '/0', '//0', '/0', '/0'];

// Maps a sequence of atomic units to its corresponding poetic foot (tafila).
export const TAFILA_MAP: Record<string, Tafila> = {
  '//0,/0': { unmerged: 'فعو لن', merged: 'فعولن' },
  '//0,/0,/0': { unmerged: 'مفا عي لن', merged: 'مفاعيلن' },
  '/0,//0,/0': { unmerged: 'فا علا تن', merged: 'فاعلاتن' },
  '/0,//0': { unmerged: 'فا علن', merged: 'فاعلن' },
  '/0,/0,//0': { unmerged: 'مس تف علن', merged: 'مستفعلن' },
};

// A utility function to parse the atomic sequence and generate the pattern for a given meter.
export const parseMeterPattern = (meter: Meter): Tafila[] => {
  const pattern: Tafila[] = [];
  let cursor = meter.startOffset;

  for (const groupSize of meter.parsingInstructions) {
    const atomicGroup = [];
    for (let i = 0; i < groupSize; i++) {
      atomicGroup.push(ATOMIC_SEQUENCE[(cursor + i) % ATOMIC_SEQUENCE.length]);
    }
    const key = atomicGroup.join(',');
    if (TAFILA_MAP[key]) {
      pattern.push(TAFILA_MAP[key]);
    }
    cursor += groupSize;
  }
  return pattern;
};


export const METERS: Meter[] = [
  {
    id: 'al-tawil',
    name: 'البحر الطويل',
    startOffset: 0,
    parsingInstructions: [2, 3, 2, 3], // [فعولن, مفاعيلن, فعولن, مفاعيلن]
    patternTransliteration: 'faʿūlun mafāʿīlun faʿūlun mafāʿīlun',
    description: 'One of the most common meters, often used for praise, satire, and themes of pride.',
  },
  {
    id: 'al-madid',
    name: 'البحر المديد',
    startOffset: 1,
    parsingInstructions: [3, 2, 3, 2], // [فاعلاتن, فاعلن, فاعلاتن, فاعلن]
    patternTransliteration: 'fāʿilātun fāʿilun fāʿilātun fāʿilun',
    description: 'A lighter meter, suitable for descriptive poetry and expressions of personal feeling. Often used in its shorter, 3-foot form.',
  },
  {
    id: 'al-basit',
    name: 'البحر البسيط',
    startOffset: 3,
    parsingInstructions: [3, 2, 3, 2], // [مستفعلن, فاعلن, مستفعلن, فاعلن]
    patternTransliteration: 'mustafʿilun fāʿilun mustafʿilun fāʿilun',
    description: 'A versatile and smooth-flowing meter, used for a wide range of narrative and descriptive topics.',
  },
];