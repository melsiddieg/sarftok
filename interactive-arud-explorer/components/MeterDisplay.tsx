
import React from 'react';
import { Meter, Tafila } from '../types';

interface MeterDisplayProps {
  activeMeter: Meter;
  activePattern: Tafila[];
}

const MeterDisplay: React.FC<MeterDisplayProps> = ({ activeMeter, activePattern }) => {
  return (
    <div key={activeMeter.id} className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-2xl p-6 w-full animate-fade-in" dir="rtl">
      <div className="flex justify-between items-start mb-4">
        <h2 className="text-3xl font-bold font-amiri text-amber-400">{activeMeter.name}</h2>
        <span className="text-sm font-mono text-gray-500 bg-gray-700/50 px-2 py-1 rounded" dir="ltr">OFFSET: {activeMeter.startOffset}</span>
      </div>
      <p className="text-gray-300 text-lg mb-6" dir="ltr">{activeMeter.description}</p>
      
      <div className="border-t border-gray-700/60 pt-4">
        <h3 className="text-lg font-semibold text-gray-400 mb-2 text-left" dir="ltr">Pattern (Taf'īlāt)</h3>
        <p className="font-amiri text-3xl text-amber-300 text-right tracking-wider" dir="rtl">
            {activePattern.map(t => t.merged).join(' ')}
        </p>
        <p className="font-mono text-sm text-gray-500 mt-1 text-left" dir="ltr">
            {activeMeter.patternTransliteration}
        </p>
      </div>

       <style>{`
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fade-in 0.5s ease-in-out;
        }
      `}</style>
    </div>
  );
};

export default MeterDisplay;
