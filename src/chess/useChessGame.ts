import { useState, useCallback, useRef } from 'react';
import { Chess, Move } from 'chess.js';
import type { GameState, MoveResult } from './types';

const INITIAL_MOVES = 3;

export function useChessGame() {
    const [gameState, setGameState] = useState<GameState>({
        fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
        initialFen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
        moveHistory: [],
        movesRemaining: INITIAL_MOVES,
        selectedSide: null,
        isLocked: false,
        gamePhase: 'setup',
    });

    const chessRef = useRef<Chess>(new Chess());

    // Load a new position from FEN
    const loadPosition = useCallback((fen: string) => {
        try {
            const chess = new Chess(fen);
            chessRef.current = chess;
            setGameState({
                fen,
                initialFen: fen,
                moveHistory: [],
                movesRemaining: INITIAL_MOVES,
                selectedSide: null,
                isLocked: false,
                gamePhase: 'setup',
            });
            return true;
        } catch (error) {
            console.error('Invalid FEN:', error);
            return false;
        }
    }, []);

    // Select which side the user will play
    const selectSide = useCallback((side: 'white' | 'black') => {
        setGameState(prev => ({
            ...prev,
            selectedSide: side,
            gamePhase: 'playing',
        }));
    }, []);

    // Make a move on the board
    const makeMove = useCallback((from: string, to: string, promotion?: string): MoveResult => {
        const chess = chessRef.current;

        // Check if board is locked
        if (gameState.isLocked) {
            return { success: false, newFen: gameState.fen, move: '', error: 'Board is locked' };
        }

        // Check if a side has been selected
        if (!gameState.selectedSide) {
            return { success: false, newFen: gameState.fen, move: '', error: 'Please select a side first' };
        }

        // Check if it's the correct side's turn (we allow only the selected side to move)
        const turn = chess.turn();
        const isWhiteTurn = turn === 'w';
        const isCorrectSide = (gameState.selectedSide === 'white' && isWhiteTurn) ||
            (gameState.selectedSide === 'black' && !isWhiteTurn);

        if (!isCorrectSide) {
            // Skip opponent's turn - just make the player's move anyway
            // For this training mode, we allow consecutive moves for one side
        }

        try {
            // For three-move mode, we need to manipulate the turn
            // We'll temporarily modify the position to allow the selected side to move
            const currentFen = chess.fen();
            const fenParts = currentFen.split(' ');
            const desiredTurn = gameState.selectedSide === 'white' ? 'w' : 'b';

            if (fenParts[1] !== desiredTurn) {
                fenParts[1] = desiredTurn;
                const modifiedFen = fenParts.join(' ');
                chess.load(modifiedFen);
            }

            const move = chess.move({ from, to, promotion: promotion || 'q' });

            if (!move) {
                // Restore original position if move failed
                chess.load(currentFen);
                return { success: false, newFen: gameState.fen, move: '', error: 'Invalid move' };
            }

            const newFen = chess.fen();
            const newMovesRemaining = gameState.movesRemaining - 1;
            const isComplete = newMovesRemaining === 0;

            setGameState(prev => ({
                ...prev,
                fen: newFen,
                moveHistory: [...prev.moveHistory, move.san],
                movesRemaining: newMovesRemaining,
                isLocked: isComplete,
                gamePhase: isComplete ? 'analyzing' : 'playing',
            }));

            return { success: true, newFen, move: move.san };
        } catch (error) {
            console.error('Move error:', error);
            return { success: false, newFen: gameState.fen, move: '', error: 'Invalid move' };
        }
    }, [gameState]);

    // Reset to original position
    const resetPosition = useCallback(() => {
        chessRef.current = new Chess(gameState.initialFen);
        setGameState(prev => ({
            ...prev,
            fen: prev.initialFen,
            moveHistory: [],
            movesRemaining: INITIAL_MOVES,
            isLocked: false,
            gamePhase: prev.selectedSide ? 'playing' : 'setup',
        }));
    }, [gameState.initialFen]);

    // Complete analysis
    const completeAnalysis = useCallback(() => {
        setGameState(prev => ({
            ...prev,
            gamePhase: 'complete',
        }));
    }, []);

    // Get legal moves for a square
    const getLegalMoves = useCallback((square: string): string[] => {
        const chess = chessRef.current;

        // Modify turn if needed for selected side
        const currentFen = chess.fen();
        const fenParts = currentFen.split(' ');
        const desiredTurn = gameState.selectedSide === 'white' ? 'w' : 'b';

        if (fenParts[1] !== desiredTurn && gameState.selectedSide) {
            fenParts[1] = desiredTurn;
            const tempChess = new Chess(fenParts.join(' '));
            const moves = tempChess.moves({ square: square as any, verbose: true });
            return moves.map((m: Move) => m.to);
        }

        const moves = chess.moves({ square: square as any, verbose: true });
        return moves.map((m: Move) => m.to);
    }, [gameState.selectedSide]);

    // Check if a square has a piece of the selected side
    const canDragPiece = useCallback((piece: string): boolean => {
        if (gameState.isLocked) return false;
        if (!gameState.selectedSide) return false;

        const pieceColor = piece[0]; // 'w' or 'b'
        return (gameState.selectedSide === 'white' && pieceColor === 'w') ||
            (gameState.selectedSide === 'black' && pieceColor === 'b');
    }, [gameState.isLocked, gameState.selectedSide]);

    return {
        gameState,
        loadPosition,
        selectSide,
        makeMove,
        resetPosition,
        completeAnalysis,
        getLegalMoves,
        canDragPiece,
    };
}
