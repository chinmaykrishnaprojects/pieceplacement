import { useState, useCallback, useEffect } from 'react';
import { ChessBoard } from './components/ChessBoard';
import { GameControls } from './components/GameControls';
import { EvaluationBar } from './components/EvaluationBar';
import { AnalysisPanel } from './components/AnalysisPanel';
import { FenDisplay } from './components/FenDisplay';
import { useChessGame } from './chess/useChessGame';
import { useStockfish } from './engine/useStockfish';
import { getRandomMiddlegamePosition } from './services/lichessService';
import type { AnalysisResult, PositionEvaluation } from './chess/types';
import './App.css';

function App() {
  const {
    gameState,
    loadPosition,
    selectSide,
    makeMove,
    resetPosition,
    canDragPiece,
  } = useChessGame();

  const {
    isLoading: isEngineLoading,
    isInitialized: isEngineReady,
    evaluatePosition,
    comparePositions,
    formatEvaluation,
  } = useStockfish();

  const [isLoadingPosition, setIsLoadingPosition] = useState(false);
  const [initialEval, setInitialEval] = useState<PositionEvaluation | null>(null);
  const [currentEval, setCurrentEval] = useState<PositionEvaluation | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Handle piece movement
  const handleMove = useCallback((from: string, to: string, promotion?: string): boolean => {
    const result = makeMove(from, to, promotion);
    return result.success;
  }, [makeMove]);

  // Load random middlegame position
  const handleLoadPosition = useCallback(async () => {
    setIsLoadingPosition(true);
    setError(null);
    setAnalysisResult(null);
    setInitialEval(null);
    setCurrentEval(null);

    try {
      const position = await getRandomMiddlegamePosition();
      loadPosition(position.fen);

      // Evaluate the initial position
      if (isEngineReady) {
        const evaluation = await evaluatePosition(position.fen);
        setInitialEval(evaluation);
        setCurrentEval(evaluation);
      }
    } catch (err) {
      console.error('Failed to load position:', err);
      setError('Failed to load position. Please try again.');
    } finally {
      setIsLoadingPosition(false);
    }
  }, [loadPosition, isEngineReady, evaluatePosition]);

  // Handle manual FEN load
  const handleLoadFen = useCallback((fen: string): boolean => {
    const success = loadPosition(fen);
    if (success) {
      setAnalysisResult(null);
      setInitialEval(null);
      setCurrentEval(null);

      // Evaluate the new position
      if (isEngineReady) {
        evaluatePosition(fen).then(setInitialEval).catch(console.error);
      }
    }
    return success;
  }, [loadPosition, isEngineReady, evaluatePosition]);

  // Handle side selection
  const handleSelectSide = useCallback((side: 'white' | 'black') => {
    selectSide(side);
  }, [selectSide]);

  // Handle reset
  const handleReset = useCallback(() => {
    resetPosition();
    setCurrentEval(initialEval);
    setAnalysisResult(null);
  }, [resetPosition, initialEval]);

  // Trigger analysis when moves are complete
  useEffect(() => {
    if (gameState.gamePhase === 'analyzing' && isEngineReady) {
      const runAnalysis = async () => {
        try {
          const result = await comparePositions(gameState.initialFen, gameState.fen);
          setAnalysisResult(result);
          setCurrentEval(result.finalEval);
        } catch (err) {
          console.error('Analysis failed:', err);
          setError('Analysis failed. Please try again.');
        }
      };
      runAnalysis();
    }
  }, [gameState.gamePhase, gameState.initialFen, gameState.fen, isEngineReady, comparePositions]);

  // Update current evaluation on moves
  useEffect(() => {
    if (gameState.gamePhase === 'playing' && isEngineReady && gameState.fen !== gameState.initialFen) {
      evaluatePosition(gameState.fen)
        .then(setCurrentEval)
        .catch(console.error);
    }
  }, [gameState.fen, gameState.gamePhase, gameState.initialFen, isEngineReady, evaluatePosition]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>♟️ Positional Chess Trainer</h1>
        <p className="subtitle">Improve your piece placement with three-move repositioning</p>
      </header>

      <main className="app-main">
        {/* Left Panel - Evaluation */}
        <div className="left-panel">
          <EvaluationBar
            evaluation={currentEval}
            label="Current Position"
            isLoading={isEngineLoading}
          />
        </div>

        {/* Center - Chessboard */}
        <div className="center-panel">
          <ChessBoard
            fen={gameState.fen}
            onMove={handleMove}
            canDragPiece={canDragPiece}
            isLocked={gameState.isLocked}
            selectedSide={gameState.selectedSide}
            moveHistory={gameState.moveHistory}
          />

          <FenDisplay
            fen={gameState.fen}
            onLoadFen={handleLoadFen}
            isLocked={gameState.isLocked}
          />

          {error && (
            <div className="error-message">
              ⚠️ {error}
              <button onClick={() => setError(null)}>✕</button>
            </div>
          )}
        </div>

        {/* Right Panel - Controls */}
        <div className="right-panel">
          <GameControls
            gamePhase={gameState.gamePhase}
            selectedSide={gameState.selectedSide}
            movesRemaining={gameState.movesRemaining}
            isLoadingPosition={isLoadingPosition}
            isAnalyzing={isEngineLoading}
            onLoadPosition={handleLoadPosition}
            onSelectSide={handleSelectSide}
            onReset={handleReset}
            onAnalyze={() => { }}
          />

          {/* Engine Status */}
          <div className="engine-status">
            <span className={`status-dot ${isEngineReady ? 'ready' : 'loading'}`}></span>
            {isEngineReady ? 'Stockfish ready' : 'Loading Stockfish...'}
          </div>
        </div>
      </main>

      {/* Analysis Panel - Full Width Below */}
      <AnalysisPanel
        result={analysisResult}
        isVisible={gameState.gamePhase === 'analyzing' || gameState.gamePhase === 'complete'}
        selectedSide={gameState.selectedSide}
      />

      <footer className="app-footer">
        <p>Focus on <strong>piece placement</strong>, <strong>coordination</strong>, and <strong>positional understanding</strong></p>
      </footer>
    </div>
  );
}

export default App;
