
/**
 * Represents a single poetic foot (tafila) in both its
 * deconstructed (unmerged) and final (merged) forms.
 */
export interface Tafila {
  unmerged: string;
  merged: string;
}

export interface Meter {
  id: string;
  name: string;
  description: string;
  startOffset: number; // Index in the master atomic sequence
  parsingInstructions: number[]; // How many atomic units to group for each tafila
  patternTransliteration: string;
}
