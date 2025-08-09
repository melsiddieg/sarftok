
import React from 'react';
import { ChevronLeftIcon, ChevronRightIcon } from './Icons';

interface ControlsProps {
  onPrev: () => void;
  onNext: () => void;
}

const Controls: React.FC<ControlsProps> = ({ onPrev, onNext }) => {
  return (
    <div className="flex items-center justify-center gap-4 w-full">
      <button
        onClick={onNext}
        aria-label="Next Meter"
        className="p-4 bg-gray-700/80 hover:bg-amber-600 rounded-full text-white transition-all duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-amber-400 focus:ring-offset-2 focus:ring-offset-gray-900"
      >
        <ChevronLeftIcon />
      </button>
      <div className="text-gray-400 font-medium text-lg">Rotate</div>
      <button
        onClick={onPrev}
        aria-label="Previous Meter"
        className="p-4 bg-gray-700/80 hover:bg-amber-600 rounded-full text-white transition-all duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-amber-400 focus:ring-offset-2 focus:ring-offset-gray-900"
      >
        <ChevronRightIcon />
      </button>
    </div>
  );
};

export default Controls;
