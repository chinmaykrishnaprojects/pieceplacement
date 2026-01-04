import type { LichessPosition } from '../chess/types';

interface GameDetailsPanelProps {
    position: LichessPosition | null;
    isRevealed: boolean;
    onToggleReveal: () => void;
}

export function GameDetailsPanel({ position, isRevealed, onToggleReveal }: GameDetailsPanelProps) {
    if (!position) return null;

    const details = position.gameDetails;

    return (
        <div className="game-details-panel">
            <div className="game-details-header">
                <h3>🎯 Game Details</h3>
                <button
                    className="reveal-toggle"
                    onClick={onToggleReveal}
                >
                    {isRevealed ? '🙈 Hide' : '👁️ Reveal'}
                </button>
            </div>

            {isRevealed ? (
                <div className="game-details-content">
                    {details ? (
                        <>
                            <div className="players">
                                <div className="player white-player">
                                    <span className="piece-icon">♔</span>
                                    <span className="player-name">{details.white}</span>
                                </div>
                                <span className="vs">vs</span>
                                <div className="player black-player">
                                    <span className="piece-icon">♚</span>
                                    <span className="player-name">{details.black}</span>
                                </div>
                            </div>
                            <div className="game-meta">
                                <span className="event">{details.event}</span>
                                <span className="year">{details.year}</span>
                            </div>
                        </>
                    ) : (
                        <p className="no-details">Game details not available</p>
                    )}

                    {position.gameUrl && (
                        <a
                            href={position.gameUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="game-link"
                        >
                            🔗 View on Lichess
                        </a>
                    )}

                    <div className="position-info">
                        <span>Move {position.moveNumber}</span>
                        <span>•</span>
                        <span>{position.pieceCount} pieces</span>
                    </div>
                </div>
            ) : (
                <div className="game-details-hidden">
                    <p>Complete your 3 moves to reveal game details!</p>
                </div>
            )}
        </div>
    );
}
