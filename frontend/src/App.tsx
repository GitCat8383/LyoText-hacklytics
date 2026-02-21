/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */
import P300Grid from './components/P300Grid';
import CandySkyBackground from './components/CandySkyBackground';

export default function App() {
  return (
    <div className="min-h-screen text-white selection:bg-emerald-500/30 overflow-hidden relative">
      <CandySkyBackground />
      <div className="relative z-10 h-screen flex flex-col overflow-y-auto">
        <P300Grid />
      </div>
    </div>
  );
}
