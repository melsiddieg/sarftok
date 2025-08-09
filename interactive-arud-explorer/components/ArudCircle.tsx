import React, { useEffect, useState } from 'react';
import { ATOMIC_SEQUENCE } from '../constants';
import { Meter, Tafila } from '../types';

interface ArudBannerProps {
  activeMeter: Meter;
  activePattern: Tafila[];
}

const ArudBanner: React.FC<ArudBannerProps> = ({ activeMeter, activePattern }) => {
  // Create a much longer sequence for infinite scrolling effect
  const repeatedSequence = Array(10).fill(ATOMIC_SEQUENCE).flat();
  const unitWidth = 90; // pixels
  const sequenceLength = ATOMIC_SEQUENCE.length;

  const totalUnitsInPattern = activeMeter.parsingInstructions.reduce((sum, val) => sum + val, 0);
  const patternWidth = totalUnitsInPattern * unitWidth;
  
  // Calculate shift with modular arithmetic for circular behavior
  const normalizedOffset = ((activeMeter.startOffset % sequenceLength) + sequenceLength) % sequenceLength;
  const shift = normalizedOffset * unitWidth;

  // State for sliding window effect
  const [currentOffset, setCurrentOffset] = useState(0);
  const [showGroupings, setShowGroupings] = useState(true);
  const [isAnimating, setIsAnimating] = useState(false);

  // Sliding window animation sequence
  useEffect(() => {
    // Calculate the target offset for smooth sliding
    const targetOffset = normalizedOffset;
    
    // Phase 1: Hide groupings immediately, start sliding
    setShowGroupings(false);
    setIsAnimating(true);
    
    // Phase 2: Update position for smooth slide
    const slideTimer = setTimeout(() => {
      setCurrentOffset(targetOffset);
    }, 50); // Small delay to ensure groupings hide first
    
    // Phase 3: After slide completes, show new groupings
    const completeTimer = setTimeout(() => {
      setShowGroupings(true);
      setIsAnimating(false);
    }, 1200);
    
    return () => {
      clearTimeout(slideTimer);
      clearTimeout(completeTimer);
    };
  }, [activeMeter.id]);

  return (
    <div 
      className="relative h-[140px] bg-gray-800/50 rounded-2xl overflow-hidden border border-gray-700 shadow-lg"
      style={{ width: `${patternWidth}px` }}
    >
      {/* Film reel: Continuous sliding window */}
      <div
        className="absolute top-0 h-full transition-transform duration-[1000ms] ease-[cubic-bezier(0.4,0,0.2,1)]"
        style={{
          width: `${repeatedSequence.length * unitWidth}px`,
          transform: `translateX(${currentOffset * unitWidth}px)`,
          right: `-${sequenceLength * unitWidth}px`,
        }}
      >
        {repeatedSequence.map((unit, index) => (
          <div
            key={index}
            className="absolute top-0 h-full flex items-center justify-center"
            style={{
              right: `${index * unitWidth}px`,
              width: `${unitWidth}px`,
            }}
          >
            <span className="text-3xl font-mono select-none text-gray-500 transition-colors duration-300">
              {unit}
            </span>
          </div>
        ))}
      </div>

      {/* Pattern groupings - appear only after reel stops */}
      {showGroupings && (
        <div className="absolute top-0 w-full h-full transition-all duration-500 ease-out opacity-100">
          {(() => {
            let cursorInPatternUnits = 0;
            return activePattern.map((tafila, tafilaIndex) => {
              const groupSize = activeMeter.parsingInstructions[tafilaIndex];
              const width = groupSize * unitWidth;
              const rightPosition = cursorInPatternUnits * unitWidth;
              
              cursorInPatternUnits += groupSize;

              return (
                <div
                  key={`group-${tafilaIndex}-${activeMeter.id}`}
                  className="absolute top-0 h-full flex flex-col"
                  style={{
                    right: `${rightPosition}px`,
                    width: `${width}px`,
                  }}
                >
                  {/* Highlighting overlay for grouped atomic units - positioned to align with centered atomic units */}
                  <div 
                    className="absolute w-full bg-amber-500/15 border-2 border-amber-400 rounded-lg"
                    style={{
                      top: '40px',
                      height: '60px'
                    }}
                  />

                  
                  {/* Tafila text - positioned lower to give more space */}
                  <div 
                    className="absolute w-full flex items-center justify-center"
                    style={{
                      top: '110px',
                      height: '30px'
                    }}
                  >
                    <span className="font-amiri text-amber-200 text-xl md:text-2xl tracking-wide font-bold">
                      {tafila.merged}
                    </span>
                  </div>
                </div>
              );
            });
          })()}
        </div>
      )}
    </div>
  );
};

export default ArudBanner;