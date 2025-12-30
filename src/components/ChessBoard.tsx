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

    // Custom square styles
    const customSquareStyles = useMemo(() => {
        const styles: Record<string, React.CSSProperties> = {};
        return styles;
    }, []);

    // react-chessboard v5 options
    const boardOptions = useMemo(() => ({
        position: fen,
        boardOrientation: boardOrientation as 'white' | 'black',
        onPieceDrop: handleDrop,
        canDragPiece: (args: { piece: string }) => canDragPiece(args.piece),
        animationDuration: 0,
        customDarkSquareStyle: { backgroundColor: '#779952' },
        customLightSquareStyle: { backgroundColor: '#edeed1' },
        customSquareStyles,
    }), [fen, boardOrientation, isLocked, selectedSide, canDragPiece, customSquareStyles]);

    return (
        <div className="chessboard-container">
            <Chessboard options={boardOptions} />

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
