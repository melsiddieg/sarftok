import { Circle } from '../../types';
import { CIRCLE1_MIXED } from './circle1-mixed';
import { CIRCLE2_PURE } from './circle2-pure';
import { CIRCLE3_CONTRACTED } from './circle3-contracted';
import { CIRCLE4_ACCORDANT } from './circle4-accordant';
import { CIRCLE5_CONSONANT } from './circle5-consonant';

// Export all individual circles
export { CIRCLE1_MIXED } from './circle1-mixed';
export { CIRCLE2_PURE } from './circle2-pure';
export { CIRCLE3_CONTRACTED } from './circle3-contracted';
export { CIRCLE4_ACCORDANT } from './circle4-accordant';
export { CIRCLE5_CONSONANT } from './circle5-consonant';

// Master array of all circles in display order
export const ALL_CIRCLES: Circle[] = [
  CIRCLE1_MIXED,
  CIRCLE2_PURE,
  CIRCLE3_CONTRACTED,
  CIRCLE4_ACCORDANT,
  CIRCLE5_CONSONANT,
].sort((a, b) => a.order - b.order);

// Utility functions for circle management
export const getCircleById = (id: string): Circle | undefined => {
  return ALL_CIRCLES.find(circle => circle.id === id);
};

export const getMeterById = (meterId: string): { circle: Circle; meter: any; meterIndex: number } | undefined => {
  for (const circle of ALL_CIRCLES) {
    const meterIndex = circle.meters.findIndex(meter => meter.id === meterId);
    if (meterIndex !== -1) {
      return { circle, meter: circle.meters[meterIndex], meterIndex };
    }
  }
  return undefined;
};

export const getTotalMeterCount = (): number => {
  return ALL_CIRCLES.reduce((total, circle) => total + circle.meters.length, 0);
};