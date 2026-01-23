import { Analyzer } from './components/Analyzer';

export function App() {
  return (
    <div className="page">
      <header className="hero">
        <div className="hero-text">
          <p className="eyebrow">Deception Indicator Analyzer</p>
          <h1>Audio-based stress and speech pattern analysis</h1>
          <p className="subhead">
            This is not lie detection. It surfaces indicators like stress, hesitation,
            and pitch variation to aid review.
          </p>
        </div>
      </header>
      <main className="content">
        <Analyzer />
      </main>
    </div>
  );
}
