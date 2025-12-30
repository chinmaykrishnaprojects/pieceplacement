import React, { useMemo } from 'react';
import { Chessboard } from 'react-chessboard';

interface ChessBoardProps {
    fen: string;
    onMove: (from: string, to: string, promotion?: string) => boolean;
    canDragPiece: (piece: string) => boolean;
    isLocked: boolean;
    selectedSide: 'white' | 'black' | null;
    moveHistory: string[];
}

export function ChessBoard({
    fen,
    onMove,
    canDragPiece,
    isLocked,
    selectedSide,
    moveHistory,
}: ChessBoardProps) {
    // Board orientation based on selected side
    const boardOrientation = selectedSide === 'black' ? 'black' : 'white';

    // Handle piece drop
    const handleDrop = (sourceSquare: string, targetSquare: string, piece: string): boolean => {
        if (isLocked) return false;

        // Check for pawn promotion
        const isPromotion =
            piece.toLowerCase().includes('p') &&
            ((piece[0] === 'w' && targetSquare[1] === '8') ||
                (piece[0] === 'b' && targetSquare[1] === '1'));

        return onMove(sourceSquare, targetSquare, isPromotion ? 'q' : undefined);
    };

    // Custom square styles for highlighting last move
    const customSquareStyles = useMemo(() => {
        const styles: Record<string, React.CSSProperties> = {};

        // Add subtle highlight for interactivity
        if (!isLocked && selectedSide) {
            // Could add drag preview styles here
        }

        return styles;
    }, [isLocked, selectedSide]);

    // Drag piece validation
    const isDraggablePiece = ({ piece }: { piece: string }): boolean => {
        return canDragPiece(piece);
    };

    return (
        <div className="chessboard-container">
            <Chessboard
                position={fen}
                onPieceDrop={handleDrop}
                isDraggablePiece={isDraggablePiece}
                boardOrientation={boardOrientation}
                customSquareStyles={customSquareStyles}
                animationDuration={200}
                boardWidth={560}
                customBoardStyle={{
                    borderRadius: '8px',
                    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
                }}
                customDarkSquareStyle={{
                    backgroundColor: '#779952',
                }}
                customLightSquareStyle={{
                    backgroundColor: '#edeed1',
                }}
            />

            {/* Move History Display */}
            {moveHistory.length > 0 && (
                <div className="move-history">
                    <span className="move-history-label">Moves:</span>
                    {moveHistory.map((move, index) => (
                        <span key={index} className="move-item">
                            {index + 1}. {move}
                        </span>
                    ))}
                </div>
            )}
        </div>
    );
}
