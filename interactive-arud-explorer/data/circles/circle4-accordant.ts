import { Circle, Meter, PoetryExample } from '../../types';

// Circle 4: الدائرة المتفقة (al-Da'ira al-Muttafiqa) - Accordant Circle
// The most complex circle with 5 distinct meters showing harmonic agreement

// Circle 4 atomic sequence - should total 21 letters when expanded
// Pattern: /0(2) //0(3) //0(3) /0(2) /0(2) //0(3) /0(2) //0(3) /0(2) = 21 letters
export const CIRCLE4_ATOMIC_SEQUENCE = ['0/', '0//', '0//', '0/', '0/', '0//', '0/', '0//', '0/'];

// Historical poetry examples for Circle 4 meters
const MUNSARIH_EXAMPLES: PoetryExample[] = [
  {
    text: 'لَيس الغَريبُ غَريبَ الشَام وَاليَمَنِ * إِنَّ الغَريبَ غَريبُ اللَحدِ وَالكَفَنِ',
    poet: 'ابن الرومي',
    translation: 'The stranger is not the stranger of Syria and Yemen; the true stranger is the stranger of the grave and shroud',
    era: 'Abbasid'
  }
];

const KHAFIF_EXAMPLES: PoetryExample[] = [
  {
    text: 'في أَيِّ يَومٍ مِنَ الأَيّامِ نَدفِنُها * نُحسِنُ صُنعاً بِأَن نُحيي وَنُدفِنُها',
    poet: 'نزار قباني',
    translation: 'On which day shall we bury it? We do well to both revive it and bury it',
    era: 'Modern'
  }
];

const MUDARI_EXAMPLES: PoetryExample[] = [
  {
    text: 'أَينَ أَزمانُ الوَصلِ بِالأَندَلُس * لَم تَكُن أَحلامَاً وَلا أَطيافاً',
    poet: 'ابن زيدون',
    translation: 'Where are the times of union in Al-Andalus? They were not dreams nor phantoms',
    era: 'Andalusian'
  }
];

const MUQTADAB_EXAMPLES: PoetryExample[] = [
  {
    text: 'إِنَّ الحَياةَ طَريقُ أَوحَش * مَا بَينَ مَولِدِنا وَمَدفَنِنا',
    poet: 'ميخائيل نعيمة',
    translation: 'Life is a desolate path between our birth and our grave',
    era: 'Modern'
  }
];

const MUJTATH_EXAMPLES: PoetryExample[] = [
  {
    text: 'يا رَبِّ إِن عَظُمَت ذُنوبي كَثرَةً * فَلَقَد عَلِمتُ بِأَنَّ عَفوَكَ أَعظَمُ',
    poet: 'أبو نواس',
    translation: 'O Lord, if my sins have grown great in number, I know that Your forgiveness is greater',
    era: 'Abbasid'
  }
];

const CIRCLE4_METERS: Meter[] = [
  {
    id: 'al-munsarih',
    name: 'البحر المنسرح',
    nameTransliteration: 'al-Bahr al-Munsarih',
    circleId: 'circle4-accordant',
    startOffset: 0,
    parsingInstructions: [3, 2, 3, 2], // [مستفعلن, مفعولات, مستفعلن, مفعولات]
    patternTransliteration: 'mustafʿilun mafʿūlātu mustafʿilun mafʿūlātu',
    description: 'The "Flowing" meter, with an easy, unimpeded rhythm perfect for philosophical reflection.',
    historicalUsage: 'Used for contemplative and philosophical poetry, especially in the Abbasid period.',
    famousExamples: MUNSARIH_EXAMPLES
  },
  {
    id: 'al-khafif',
    name: 'البحر الخفيف',
    nameTransliteration: 'al-Bahr al-Khafif',
    circleId: 'circle4-accordant',
    startOffset: 1,
    parsingInstructions: [3, 2, 2, 3], // [فاعلاتن, مستفع, لن فاعلاتن]
    patternTransliteration: 'fāʿilātun mustafʿi lan fāʿilātun',
    description: 'The "Light" meter, creating a swift, airy quality ideal for delicate subjects.',
    historicalUsage: 'Popular for love poetry and nature descriptions, favored by court poets.',
    famousExamples: KHAFIF_EXAMPLES
  },
  {
    id: 'al-mudari',
    name: 'البحر المضارع',
    nameTransliteration: 'al-Bahr al-Mudāriʿ',
    circleId: 'circle4-accordant',
    startOffset: 3,
    parsingInstructions: [2, 3, 2, 3], // [مفاعيل, فاعلاتن, مفاعيل, فاعلاتن]
    patternTransliteration: 'mafāʿīlu fāʿilātun mafāʿīlu fāʿilātun',
    description: 'The "Resembling" meter, creating subtle variations and nuanced rhythmic patterns.',
    historicalUsage: 'Less common but used for sophisticated, nuanced poetry requiring rhythmic variety.',
    famousExamples: MUDARI_EXAMPLES
  },
  {
    id: 'al-muqtadab',
    name: 'البحر المقتضب',
    nameTransliteration: 'al-Bahr al-Muqtadab',
    circleId: 'circle4-accordant',
    startOffset: 5,
    parsingInstructions: [2, 3, 2], // [فاعلن, مفتعلن, فاعلن]
    patternTransliteration: 'fāʿilun muftaʿilun fāʿilun',
    description: 'The "Abbreviated" meter, concise and powerful, often used in its shortened forms.',
    historicalUsage: 'Used for epigrams and brief, impactful verses requiring conciseness.',
    famousExamples: MUQTADAB_EXAMPLES
  },
  {
    id: 'al-mujtath',
    name: 'البحر المجتث',
    nameTransliteration: 'al-Bahr al-Mujtath',
    circleId: 'circle4-accordant',
    startOffset: 6,
    parsingInstructions: [3, 2, 3], // [مستفعلن, فاعلاتن, مستفعلن]
    patternTransliteration: 'mustafʿilun fāʿilātun mustafʿilun',
    description: 'The "Uprooted" meter, creating a sense of urgency and emotional intensity.',
    historicalUsage: 'Often used for prayers, religious verse, and emotionally charged poetry.',
    famousExamples: MUJTATH_EXAMPLES
  },
];

export const CIRCLE4_ACCORDANT: Circle = {
  id: 'circle4-accordant',
  name: 'دائرة المشتبه',
  nameTransliteration: 'al-Da\'ira al-Muttafiqa',
  description: 'The Accordant Circle - the most complex circle with 5 harmonically related meters',
  atomicSequence: CIRCLE4_ATOMIC_SEQUENCE,
  baseSequenceLength: 9,
  meters: CIRCLE4_METERS,
  visualTheme: {
    primaryColor: '#8B5CF6', // Violet
    accentColor: '#7C3AED',
    backgroundGradient: ['#EDE9FE', '#C4B5FD'],
    borderColor: '#6D28D9'
  },
  order: 4
};