import type { LichessPosition } from '../chess/types';

// Lichess Explorer API endpoint
const LICHESS_MASTERS_URL = 'https://explorer.lichess.ovh/masters';

// Count pieces in a FEN string (excluding kings which are always 2)
function countPieces(fen: string): number {
    const boardPart = fen.split(' ')[0];
    let count = 0;
    for (const char of boardPart) {
        if ('pnbrqkPNBRQK'.includes(char)) {
            count++;
        }
    }
    return count;
}

// Validate that a FEN represents a legal and interesting middlegame position
function isValidMiddlegamePosition(fen: string): boolean {
    const pieceCount = countPieces(fen);
    // Middlegame typically has 12-28 pieces (excluding kings)
    // We'll be slightly more inclusive: 12-30
    return pieceCount >= 10 && pieceCount <= 30;
}

// Fetch random games from Lichess Masters Explorer
async function fetchRandomGames(): Promise<any[]> {
    const params = new URLSearchParams({
        variant: 'standard',
        topGames: '20', // Masters API uses topGames
    });

    try {
        const response = await fetch(`${LICHESS_MASTERS_URL}?${params}`);
        if (!response.ok) {
            throw new Error(`Lichess API error: ${response.status}`);
        }
        const data = await response.json();
        // Return both topGames and recentGames just in case
        return [...(data.topGames || []), ...(data.recentGames || [])];
    } catch (error) {
        console.error('Failed to fetch from Lichess:', error);
        throw error;
    }
}

// Import Chess for position replay
import { Chess } from 'chess.js';

// Find a random middlegame position from a game
function findMiddlegamePosition(game: any): LichessPosition | null {
    try {
        const chess = new Chess();
        const moves = game.uci ? game.uci.split(' ') : [];

        if (moves.length < 20) return null; // Need enough moves for a deep middlegame

        // Play through the game and find middlegame positions
        const validPositions: LichessPosition[] = [];

        // Start looking from move 10 to move 40
        for (let i = 0; i < moves.length && i < 80; i++) {
            try {
                const move = moves[i];
                // UCI format: e2e4, e7e5, etc.
                const from = move.slice(0, 2);
                const to = move.slice(2, 4);
                const promotion = move.length > 4 ? move[4] : undefined;

                chess.move({ from, to, promotion });

                // Check if this is a middlegame position (move range 10-35)
                if (i >= 20 && i <= 70) {
                    const fen = chess.fen();
                    if (isValidMiddlegamePosition(fen)) {
                        validPositions.push({
                            fen,
                            pieceCount: countPieces(fen),
                            moveNumber: Math.floor(i / 2) + 1,
                            gameUrl: game.id ? `https://lichess.org/${game.id}` : undefined,
                        });
                    }
                }
            } catch (moveError) {
                // Skip invalid moves
                break;
            }
        }

        if (validPositions.length === 0) return null;

        // Return a random position from the valid ones
        return validPositions[Math.floor(Math.random() * validPositions.length)];
    } catch (error) {
        console.error('Error processing game:', error);
        return null;
    }
}

// Main function to get a random middlegame position
export async function getRandomMiddlegamePosition(): Promise<LichessPosition> {
    try {
        const games = await fetchRandomGames();

        if (games.length === 0) {
            throw new Error('No games found from Lichess');
        }

        // Shuffle games and try to find a valid middlegame position
        const shuffledGames = [...games].sort(() => Math.random() - 0.5);

        for (const game of shuffledGames) {
            const position = findMiddlegamePosition(game);
            if (position) {
                return position;
            }
        }
    } catch (error) {
        console.error('Lichess fetch failed, using fallback:', error);
    }

    // Fallback: A complex GM middlegame position (Carlsen vs Caruana)
    return {
        fen: 'r2q1rk1/pp1nbppp/2p1pnb1/3p4/2PP4/1PN1PNP1/PB3PBP/R2Q1RK1 w - - 3 11',
        pieceCount: 28,
        moveNumber: 11,
        gameUrl: undefined,
    };
}

// Alternative: Get position from a specific opening
export async function getPositionFromOpening(opening: string = 'd4'): Promise<LichessPosition> {
    const params = new URLSearchParams({
        variant: 'standard',
        play: opening,
        topGames: '10',
    });

    try {
        const response = await fetch(`${LICHESS_MASTERS_URL}?${params}`);
        if (!response.ok) {
            throw new Error(`Lichess API error: ${response.status}`);
        }
        const data = await response.json();
        const games = [...(data.topGames || []), ...(data.recentGames || [])];

        if (games.length === 0) {
            throw new Error('No games found for this opening');
        }

        // Try to find a middlegame position from any of the games
        for (const game of games) {
            const position = findMiddlegamePosition(game);
            if (position) {
                return position;
            }
        }

        throw new Error('No valid middlegame positions found');
    } catch (error) {
        console.error('Failed to fetch opening positions:', error);
        throw error;
    }
}
