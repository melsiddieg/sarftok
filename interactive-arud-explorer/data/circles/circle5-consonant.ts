import { Circle, Meter, PoetryExample } from '../../types';

// Circle 5: الدائرة المتداركة (al-Da'ira al-Mutadarika) - Consonant Circle  
// This circle contains meters with consonant-heavy, rapid rhythmic patterns

// Circle 5 atomic sequence - should total 20 letters when expanded
// Pattern: /0(2) /0(2) /0(2) /0(2) //0(3) //0(3) /0(2) /0(2) /0(2) = 20 letters
export const CIRCLE5_ATOMIC_SEQUENCE = ['0/', '0/', '0/', '0/', '0//', '0//', '0/', '0/', '0/'];

// Historical poetry examples for Circle 5 meters
const MUTAQARIB_EXAMPLES: PoetryExample[] = [
  {
    text: 'أفَدِ الحَيَوانَ حَدِيثاً ما لَهُ بَدُ * أو فَاصمُت فَالسَكوتُ لَكَ أَبَرُد',
    poet: 'أبو العلاء المعري',
    translation: 'Tell the living a necessary tale, or be silent, for silence is cooler for you',
    era: 'Abbasid'
  },
  {
    text: 'وَما نَيلُ الأَماني بِالتَمَنّي * وَلكِن تُؤخَذُ الدُنيا غِلاباً',
    poet: 'أحمد شوقي',
    translation: 'Hopes are not attained by wishing, but the world is taken by force',
    era: 'Modern'
  }
];

const MUTADARIK_EXAMPLES: PoetryExample[] = [
  {
    text: 'يا لَيلُ يا لَيلُ يا لَيلُ الصَبا * رُدَّ عَلى قَلبيَ المَهجُورِ صَباهُ',
    poet: 'فاروق جويدة',
    translation: 'O night, O night, O night of youth, return to my abandoned heart its youth',
    era: 'Contemporary'
  }
];

const SARI_EXAMPLES: PoetryExample[] = [
  {
    text: 'بَكَيتُ عَلى الشَبابِ بِدَمعَةٍ * فَلَم تُغنِ البُكا شَيئاً وَلَم تُرِد',
    poet: 'أبو العتاهية',
    translation: 'I wept for youth with a tear, but weeping availed nothing and brought nothing back',
    era: 'Abbasid'
  }
];

const CIRCLE5_METERS: Meter[] = [
  {
    id: 'al-mutaqarib',
    name: 'البحر المتقارب',
    nameTransliteration: 'al-Bahr al-Mutaqārib',
    circleId: 'circle5-consonant',
    startOffset: 0,
    parsingInstructions: [2, 2, 2, 2], // [فعولن, فعولن, فعولن, فعولن]
    patternTransliteration: 'faʿūlun faʿūlun faʿūlun faʿūlun',
    description: 'The "Approaching" meter, with closely spaced beats creating rapid, urgent rhythm.',
    historicalUsage: 'Perfect for dramatic monologues and emotional outbursts, popular in modern poetry.',
    famousExamples: MUTAQARIB_EXAMPLES
  },
  {
    id: 'al-mutadarik',
    name: 'البحر المتدارك',
    nameTransliteration: 'al-Bahr al-Mutadārik',
    circleId: 'circle5-consonant',
    startOffset: 2,
    parsingInstructions: [2, 2, 2, 2], // [فاعلن, فاعلن, فاعلن, فاعلن]
    patternTransliteration: 'fāʿilun fāʿilun fāʿilun fāʿilun',
    description: 'The "Compensating" meter, filling gaps left by other meters with its unique cadence.',
    historicalUsage: 'Relatively late addition to classical prosody, popular in folk songs and modern verse.',
    famousExamples: MUTADARIK_EXAMPLES
  },
  {
    id: 'al-sari',
    name: 'البحر السريع',
    nameTransliteration: 'al-Bahr al-Sarīʿ',
    circleId: 'circle5-consonant',
    startOffset: 4,
    parsingInstructions: [3, 2, 3, 2], // [مستفعلن, مستفعلن, مفعولات, مفعولات]
    patternTransliteration: 'mustafʿilun mustafʿilun mafʿūlātu mafʿūlātu',
    description: 'The "Swift" meter, moving with quick, decisive beats perfect for energetic themes.',
    historicalUsage: 'Used for narrative poetry requiring speed and momentum, especially in historical accounts.',
    famousExamples: SARI_EXAMPLES
  },
];

export const CIRCLE5_CONSONANT: Circle = {
  id: 'circle5-consonant',
  name: 'دائرة المتفق',
  nameTransliteration: 'al-Da\'ira al-Mutadarika',
  description: 'The Consonant Circle - containing meters with rapid, consonant-heavy rhythmic patterns',
  atomicSequence: CIRCLE5_ATOMIC_SEQUENCE,
  baseSequenceLength: 9,
  meters: CIRCLE5_METERS,
  visualTheme: {
    primaryColor: '#EF4444', // Red
    accentColor: '#DC2626',
    backgroundGradient: ['#FEE2E2', '#FCA5A5'],
    borderColor: '#B91C1C'
  },
  order: 5
};