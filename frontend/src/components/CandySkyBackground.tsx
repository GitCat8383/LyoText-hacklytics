import React from 'react';
import { motion } from 'motion/react';

const SimpleCloud = ({ className, style }: { className?: string, style?: React.CSSProperties }) => (
    <div className={`absolute text-white ${className}`} style={style}>
        <svg viewBox="0 0 64 32" fill="currentColor" className="w-full h-full drop-shadow-sm">
            <path d="M16 12C16 7.58172 19.5817 4 24 4C27.0564 4 29.7149 5.71661 31.0625 8.24357C32.2226 7.46467 33.6062 7 35.1 7C39.5183 7 43.1 10.5817 43.1 15C43.1 15.25 43.0872 15.4969 43.0621 15.7396C45.2451 16.3274 47 18.2536 47 20.6C47 23.5823 44.5823 26 41.6 26H12C7.58172 26 4 22.4183 4 18C4 13.9926 6.94436 10.6726 10.7626 10.108C11.6661 6.64724 14.8093 4 18.5 4C19.3876 4 20.2411 4.15396 21.0375 4.43635C21.9469 4.15577 22.9103 4 23.9 4H24C28.4183 4 32 7.58172 32 12H16Z" />
        </svg>
    </div>
);

const Lollipop = ({ color, className, delay = 0 }: { color: string, className?: string, delay?: number }) => (
  <motion.div 
    initial={{ y: 200, opacity: 0 }}
    animate={{ y: 0, opacity: 1 }}
    transition={{ duration: 1.2, delay, type: "spring", bounce: 0.5 }}
    className={`absolute flex flex-col items-center ${className}`}
  >
    <div className={`w-24 h-24 md:w-32 md:h-32 rounded-full border-4 border-white shadow-lg relative overflow-hidden ${color}`}>
       {/* Swirl pattern simulation */}
       <div className="absolute inset-0 border-[8px] border-white/30 rounded-full scale-75"></div>
       <div className="absolute inset-0 border-[8px] border-white/30 rounded-full scale-50"></div>
       <div className="absolute inset-0 border-[8px] border-white/30 rounded-full scale-25"></div>
    </div>
    <div className="w-3 h-32 md:h-48 bg-white shadow-sm mt-[-2px] -z-10"></div>
  </motion.div>
);

const WrappedCandy = ({ color, className, rotate = 0 }: { color: string, className?: string, rotate?: number }) => (
    <div className={`absolute ${className}`} style={{ transform: `rotate(${rotate}deg)` }}>
        <div className={`w-12 h-8 rounded-lg ${color} shadow-md relative flex items-center justify-center`}>
             <div className="w-10 h-6 border border-white/20 rounded opacity-50"></div>
             {/* Wrapper ends */}
             <div className={`absolute -left-3 top-1/2 -translate-y-1/2 w-4 h-6 ${color} [clip-path:polygon(100%_0,0_20%,0_80%,100%_100%)]`}></div>
             <div className={`absolute -right-3 top-1/2 -translate-y-1/2 w-4 h-6 ${color} [clip-path:polygon(0_0,100%_20%,100%_80%,0_100%)]`}></div>
        </div>
    </div>
);

const ChocolateBar = ({ className, rotate = 0 }: { className?: string, rotate?: number }) => (
    <div className={`absolute ${className}`} style={{ transform: `rotate(${rotate}deg)` }}>
        <div className="w-16 h-20 bg-[#5D4037] rounded-sm shadow-lg grid grid-cols-2 gap-1 p-1 border-b-4 border-r-4 border-[#3E2723]">
            <div className="bg-[#795548] rounded-sm shadow-inner"></div>
            <div className="bg-[#795548] rounded-sm shadow-inner"></div>
            <div className="bg-[#795548] rounded-sm shadow-inner"></div>
            <div className="bg-[#795548] rounded-sm shadow-inner"></div>
            <div className="bg-[#795548] rounded-sm shadow-inner"></div>
            <div className="bg-[#795548] rounded-sm shadow-inner"></div>
        </div>
        {/* Bite mark */}
        <div className="absolute -top-2 -right-2 w-8 h-8 bg-[#9ecaff] rounded-full"></div>
    </div>
);

const FloatingCandy = ({ color, className, size = "w-4 h-4" }: { color: string, className?: string, size?: string }) => (
    <div className={`absolute rounded-full ${size} ${color} ${className} shadow-sm opacity-80`} />
);

export default function CandySkyBackground() {
  return (
    <div className="fixed inset-0 z-0 overflow-hidden bg-[#9ecaff]">
      {/* Sky Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-[#89c2ff] to-[#b8d8ff]" />

      {/* Floating Candies (Background) */}
      <FloatingCandy color="bg-yellow-300" className="top-[15%] left-[20%]" />
      <FloatingCandy color="bg-pink-300" className="top-[25%] right-[25%] w-6 h-6" />
      <FloatingCandy color="bg-blue-300" className="top-[40%] left-[10%] w-3 h-3" />
      <FloatingCandy color="bg-green-300" className="top-[10%] right-[40%]" />
      <FloatingCandy color="bg-purple-300" className="bottom-[30%] right-[15%] w-5 h-5" />

      {/* Chocolates & Wrapped Candies */}
      <ChocolateBar className="top-[15%] left-[5%] opacity-60" rotate={-15} />
      <ChocolateBar className="top-[30%] right-[5%] opacity-60 scale-75" rotate={20} />
      
      <WrappedCandy color="bg-orange-400" className="top-[40%] left-[8%]" rotate={45} />
      <WrappedCandy color="bg-teal-400" className="top-[10%] right-[15%]" rotate={-30} />
      <WrappedCandy color="bg-rose-400" className="bottom-[35%] left-[25%]" rotate={10} />

      {/* Clouds */}
      <SimpleCloud className="top-[10%] left-[5%] w-32 h-16 opacity-80" />
      <SimpleCloud className="top-[20%] right-[10%] w-48 h-24 opacity-90" />
      <SimpleCloud className="top-[5%] right-[30%] w-24 h-12 opacity-60" />
      <SimpleCloud className="bottom-[40%] left-[15%] w-40 h-20 opacity-70" />
      
      {/* Big Bottom Clouds */}
      <div className="absolute bottom-0 left-0 right-0 h-[30vh] pointer-events-none">
         <SimpleCloud className="bottom-[-20px] left-[-5%] w-[40vw] h-[20vw] text-white opacity-90" />
         <SimpleCloud className="bottom-[-40px] right-[-5%] w-[45vw] h-[22vw] text-white opacity-90" />
         <SimpleCloud className="bottom-[-60px] left-[30%] w-[50vw] h-[25vw] text-white/80" />
      </div>

      {/* Lollipops */}
      <div className="absolute bottom-[-40px] left-0 right-0 h-[300px] flex justify-center items-end gap-8 md:gap-32 pointer-events-none z-10">
        <Lollipop color="bg-red-400" className="mb-10" delay={0.2} />
        <Lollipop color="bg-pink-400" className="mb-0 scale-75" delay={0.4} />
        <Lollipop color="bg-green-400" className="mb-4 scale-90" delay={0.6} />
        <Lollipop color="bg-purple-400" className="mb-12" delay={0.8} />
      </div>
    </div>
  );
}
