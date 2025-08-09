
import React, { useState, useCallback } from 'react';
import { METERS, parseMeterPattern } from './constants';
import ArudBanner from './components/ArudCircle';
import MeterDisplay from './components/MeterDisplay';
import Controls from './components/Controls';
import InfoCard from './components/InfoCard';
import { Meter, Tafila } from './types';

const App: React.FC = () => {
  const [currentMeterIndex, setCurrentMeterIndex] = useState(0);

  const handleNext = useCallback(() => {
    setCurrentMeterIndex((prevIndex) => (prevIndex + 1) % METERS.length);
  }, []);

  const handlePrev = useCallback(() => {
    setCurrentMeterIndex((prevIndex) => (prevIndex - 1 + METERS.length) % METERS.length);
  }, []);

  const activeMeter: Meter = METERS[currentMeterIndex];
  const activePattern: Tafila[] = parseMeterPattern(activeMeter);

  return (
    <div className="min-h-screen w-full bg-gray-900 flex flex-col items-center justify-center p-4 overflow-hidden">
      <header className="text-center mb-6">
        <h1 className="text-4xl md:text-5xl font-bold text-amber-400 font-amiri">مستكشف دوائر العروض</h1>
        <p className="text-gray-400 mt-2 text-lg">Interactive Arud Explorer</p>
      </header>

      <InfoCard />
      
      <main className="flex flex-col items-center justify-center gap-8 w-full max-w-7xl">
        <div className="w-full flex items-center justify-center p-4 h-[160px]">
          <ArudBanner activeMeter={activeMeter} activePattern={activePattern} />
        </div>
        
        <div className="w-full lg:w-[700px] flex flex-col gap-8">
          <MeterDisplay activeMeter={activeMeter} activePattern={activePattern} />
          <Controls onPrev={handlePrev} onNext={handleNext} />
        </div>
      </main>

      <footer className="mt-8 text-center text-gray-500 text-sm">
        <p>Inspired by the work of Al-Khalil ibn Ahmad al-Farahidi.</p>
        <p>Slide the banner to explore the poetic meters of Arabic.</p>
      </footer>
    </div>
  );
};

export default App;
