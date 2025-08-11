import { Circle, Meter, PoetryExample } from '../../types';

// Circle 2: الدائرة المجتلبة (al-Da'ira al-Mujtaliба) - Pure Circle  
// This circle contains meters with "pure" prosodic patterns

// Circle 2 atomic sequence - ensures both meters get uniform patterns
// الوافر (index 0): //0 ///0 | //0 ///0 | //0 ///0 = مفاعلتن مفاعلتن مفاعلتن
// الكامل (index 1): ///0 //0 | ///0 //0 | ///0 //0 = متفاعلن متفاعلن متفاعلن
export const CIRCLE2_ATOMIC_SEQUENCE = ['0//', '0///', '0//', '0///', '0//', '0///', '0//', '0///', '0//', '0///', '0//', '0///'];

// Historical poetry examples for Circle 2 meters
const WAFIR_EXAMPLES: PoetryExample[] = [
  {
    text: 'مُباركٌ الاسمُ أَغَرُّ اللَقَبْ * كَريمُ الجَرثومَةِ شَريفُ النَسَبْ',
    poet: 'أحمد شوقي',
    translation: 'Blessed is the name, glorious the title, noble the origin, honorable the lineage',
    era: 'Modern'
  },
  {
    text: 'يا مَن يَعِزُّ عَلَيْنَا أَنْ نُفَارِقَهُمْ * وِجْدَانُنَا كُلَّ شَيْءٍ بَعْدَكُمْ عَدَمُ',
    poet: 'ابن زيدون',
    translation: 'O those whom it grieves us to part from, our consciousness after you is nothingness',
    era: 'Andalusian'
  }
];

const KAMIL_EXAMPLES: PoetryExample[] = [
  {
    text: 'بانَتْ سُعادُ فَقَلبي اليَومَ مَتبولُ * مُتَيَّمٌ إِثرَها لَم يُفْدَ مَكبولُ',
    poet: 'كعب بن زهير',
    translation: 'Su\'ad has departed, and my heart today is afflicted, captivated by her, unfree and shackled',
    era: 'Early Islamic'
  },
  {
    text: 'أَلا هُبِّي بِصَحْنِكِ فَاصْبَحِينَا * وَلا تُبْقِي خُمُورَ الأَنْدَرِينَا',
    poet: 'الأعشى',
    translation: 'Come, arise with your cup and give us to drink, do not leave the wines of the Andarīn',
    era: 'Pre-Islamic'
  }
];

const CIRCLE2_METERS: Meter[] = [
  {
    id: 'al-wafir',
    name: 'البحر الوافر', 
    nameTransliteration: 'al-Bahr al-Wafir',
    circleId: 'circle2-pure',
    startOffset: 0,
    parsingInstructions: [2, 2, 2], // Three instances of مفاعلتن (//0 ///0)
    patternTransliteration: 'mufāʿilatun mufāʿilatun mufāʿilatun',
    description: 'A flowing meter with compound feet, creating a sense of abundance and completeness.',
    historicalUsage: 'Popular in classical Arabic poetry, especially for philosophical and contemplative themes.',
    famousExamples: WAFIR_EXAMPLES
  },
  {
    id: 'al-kamil',
    name: 'البحر الكامل',
    nameTransliteration: 'al-Bahr al-Kamil',
    circleId: 'circle2-pure',
    startOffset: 1,
    parsingInstructions: [2, 2, 2], // Three instances of متفاعلن (///0,//0 pattern, shifted by 3)
    patternTransliteration: 'mutafāʿilun mutafāʿilun mutafāʿilun',
    description: 'The "Complete" meter, known for its perfect symmetry and musical quality.',
    historicalUsage: 'Extremely popular across all periods, used for panegyric, elegiac, and narrative poetry.',
    famousExamples: KAMIL_EXAMPLES
  },
];

export const CIRCLE2_PURE: Circle = {
  id: 'circle2-pure',
  name: 'دائرة المؤتلف',
  nameTransliteration: 'al-Da\'ira al-Mujtaliба',
  description: 'The Pure Circle - containing meters with harmonious, complete prosodic patterns',
  atomicSequence: CIRCLE2_ATOMIC_SEQUENCE,
  baseSequenceLength: 9,
  meters: CIRCLE2_METERS,
  visualTheme: {
    primaryColor: '#3B82F6', // Blue
    accentColor: '#1D4ED8',
    backgroundGradient: ['#DBEAFE', '#93C5FD'],
    borderColor: '#1E40AF'
  },
  order: 2
};