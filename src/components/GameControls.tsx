import React from 'react';

interface GameControlsProps {
    gamePhase: 'setup' | 'playing' | 'analyzing' | 'complete';
    selectedSide: 'white' | 'black' | null;
    movesRemaining: number;
    isLoadingPosition: boolean;
    isAnalyzing: boolean;
    onLoadPosition: () => void;
    onSelectSide: (side: 'white' | 'black') => void;
    onReset: () => void;
    onAnalyze: () => void;
}

export function GameControls({
    gamePhase,
    selectedSide,
    movesRemaining,
    isLoadingPosition,
    isAnalyzing,
    onLoadPosition,
    onSelectSide,
    onReset,
    onAnalyze,
}: GameControlsProps) {
    return (
        <div className="game-controls">
            {/* Load Position Button */}
            <button
                className="control-button primary"
                onClick={onLoadPosition}
                disabled={isLoadingPosition || isAnalyzing}
            >
                {isLoadingPosition ? (
                    <>
                        <span className="spinner"></span>
                        Loading...
                    </>
                ) : (
                    '♟️ Load Random Middlegame Position'
                )}
            </button>

            {/* Side Selection */}
            <div className="side-selection">
                <h3>Select Your Side</h3>
                <div className="side-buttons">
                    <button
                        className={`side-button white ${selectedSide === 'white' ? 'selected' : ''}`}
                        onClick={() => onSelectSide('white')}
                        disabled={gamePhase === 'analyzing' || gamePhase === 'complete'}
                    >
                        ♔ White
                    </button>
                    <button
                        className={`side-button black ${selectedSide === 'black' ? 'selected' : ''}`}
                        onClick={() => onSelectSide('black')}
                        disabled={gamePhase === 'analyzing' || gamePhase === 'complete'}
                    >
                        ♚ Black
                    </button>
                </div>
            </div>

            {/* Moves Remaining Indicator */}
            <div className="moves-indicator">
                <h3>Moves Remaining</h3>
                <div className="moves-display">
                    <div className={`move-dot ${movesRemaining >= 1 ? 'active' : 'used'}`}></div>
                    <div className={`move-dot ${movesRemaining >= 2 ? 'active' : 'used'}`}></div>
                    <div className={`move-dot ${movesRemaining >= 3 ? 'active' : 'used'}`}></div>
                </div>
                <span className="moves-count">{movesRemaining} / 3</span>
            </div>

            {/* Game Status */}
            <div className="game-status">
                {gamePhase === 'setup' && !selectedSide && (
                    <p className="status-message">Load a position and select your side to begin</p>
                )}
                {gamePhase === 'setup' && selectedSide && (
                    <p className="status-message">Ready! Make your first move</p>
                )}
                {gamePhase === 'playing' && (
                    <p className="status-message">
                        Make {movesRemaining} more move{movesRemaining !== 1 ? 's' : ''} to improve the position
                    </p>
                )}
                {gamePhase === 'analyzing' && (
                    <p className="status-message analyzing">
                        <span className="spinner"></span>
                        Analyzing positions...
                    </p>
                )}
                {gamePhase === 'complete' && (
                    <p className="status-message complete">Analysis complete! Review your results</p>
                )}
            </div>

            {/* Action Buttons */}
            <div className="action-buttons">
                <button
                    className="control-button secondary"
                    onClick={onReset}
                    disabled={isAnalyzing}
                >
                    🔄 Reset Position
                </button>

                {gamePhase === 'analyzing' && (
                    <button
                        className="control-button accent"
                        onClick={onAnalyze}
                        disabled={isAnalyzing}
                    >
                        {isAnalyzing ? (
                            <>
                                <span className="spinner"></span>
                                Analyzing...
                            </>
                        ) : (
                            '📊 View Analysis'
                        )}
                    </button>
                )}
            </div>
        </div>
    );
}
