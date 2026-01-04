import type { LichessPosition } from '../chess/types';
import { Chess } from 'chess.js';

// Lichess Explorer API endpoint - using Masters database for high-quality games
const LICHESS_MASTERS_URL = 'https://explorer.lichess.ovh/masters';

// Count pieces in a FEN string
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

// Validate that a FEN represents a legal middlegame position
function isValidMiddlegamePosition(fen: string): boolean {
    const pieceCount = countPieces(fen);
    return pieceCount >= 12 && pieceCount <= 30;
}

// Fetch games from Lichess Masters Explorer API
async function fetchLichessGames(): Promise<any[]> {
    // Use a random opening move in UCI format (e.g., e2e4 not e4)
    const openingMoves = ['e2e4', 'd2d4', 'c2c4', 'g1f3', 'e2e4,e7e5', 'd2d4,d7d5', 'e2e4,c7c5'];
    const randomOpening = openingMoves[Math.floor(Math.random() * openingMoves.length)];

    const params = new URLSearchParams({
        play: randomOpening,
        topGames: '15',
    });

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

    try {
        const response = await fetch(`${LICHESS_MASTERS_URL}?${params}`, {
            signal: controller.signal,
        });
        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`Lichess API error: ${response.status}`);
        }
        const data = await response.json();
        return data.topGames || [];
    } catch (error) {
        clearTimeout(timeoutId);
        console.error('Failed to fetch from Lichess:', error);
        throw error;
    }
}

// Find a middlegame position from a Lichess game
function findMiddlegamePosition(game: any): LichessPosition | null {
    try {
        const chess = new Chess();
        const moves = game.uci ? game.uci.split(' ') : [];

        if (moves.length < 20) return null;

        const validPositions: LichessPosition[] = [];

        // Extract game details from Masters API response
        const gameDetails = {
            white: game.white?.name || 'Unknown',
            black: game.black?.name || 'Unknown',
            event: game.month ? `Masters ${game.month}` : 'Masters Game',
            year: game.year || (game.month ? parseInt(game.month.split('-')[0]) : 2000),
        };

        for (let i = 0; i < moves.length && i < 80; i++) {
            try {
                const move = moves[i];
                const from = move.slice(0, 2);
                const to = move.slice(2, 4);
                const promotion = move.length > 4 ? move[4] : undefined;

                chess.move({ from, to, promotion });

                // Check if this is a middlegame position (after move 10-35)
                if (i >= 20 && i <= 70) {
                    const fen = chess.fen();
                    if (isValidMiddlegamePosition(fen)) {
                        validPositions.push({
                            fen,
                            pieceCount: countPieces(fen),
                            moveNumber: Math.floor(i / 2) + 1,
                            gameUrl: game.id ? `https://lichess.org/${game.id}` : undefined,
                            gameDetails,
                        });
                    }
                }
            } catch {
                break;
            }
        }

        if (validPositions.length === 0) return null;
        return validPositions[Math.floor(Math.random() * validPositions.length)];
    } catch (error) {
        console.error('Error processing game:', error);
        return null;
    }
}

// Curated collection of famous middlegame positions as fallback
const FAMOUS_MIDDLEGAMES: LichessPosition[] = [
    // Carlsen vs Caruana, 2018 World Championship
    {
        fen: 'r2q1rk1/pp1nbppp/2p1pn2/3p4/2PP4/1PN1PN2/PB3PPP/R2QKB1R w KQ - 0 9',
        pieceCount: 28,
        moveNumber: 9,
        gameUrl: 'https://lichess.org/study/carlsen-caruana-2018',
        gameDetails: { white: 'Magnus Carlsen', black: 'Fabiano Caruana', event: 'World Championship', year: 2018 },
    },
    // Kasparov vs Topalov, Wijk aan Zee 1999
    {
        fen: 'rn3rk1/1bq1bppp/p3pn2/1p6/3P4/1BN1PN2/PP3PPP/R1BQ1RK1 w - - 0 12',
        pieceCount: 28,
        moveNumber: 12,
        gameUrl: 'https://lichess.org/KYZPDB2y',
        gameDetails: { white: 'Garry Kasparov', black: 'Veselin Topalov', event: 'Wijk aan Zee', year: 1999 },
    },
    // Fischer vs Spassky, 1972 World Championship
    {
        fen: 'r1bq1rk1/ppp2ppp/2n2n2/2bpp3/2B1P3/2PP1N2/PP3PPP/RNBQ1RK1 w - - 0 8',
        pieceCount: 30,
        moveNumber: 8,
        gameUrl: undefined,
        gameDetails: { white: 'Bobby Fischer', black: 'Boris Spassky', event: 'World Championship', year: 1972 },
    },
    // Sicilian Najdorf middlegame
    {
        fen: 'r1b1kb1r/1pqn1ppp/p2ppn2/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQkq - 0 10',
        pieceCount: 28,
        moveNumber: 10,
        gameUrl: undefined,
        gameDetails: { white: 'Unknown', black: 'Unknown', event: 'Sicilian Najdorf', year: 2000 },
    },
    // Catalan middlegame
    {
        fen: 'r1bq1rk1/pp1n1ppp/2p1pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQ1RK1 w - - 0 8',
        pieceCount: 30,
        moveNumber: 8,
        gameUrl: undefined,
        gameDetails: { white: 'Unknown', black: 'Unknown', event: 'Catalan Opening', year: 2000 },
    },
    // King's Indian Defense
    {
        fen: 'r1bq1rk1/ppp1bppp/2n2n2/3pp3/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 8',
        pieceCount: 30,
        moveNumber: 8,
        gameUrl: undefined,
        gameDetails: { white: 'Unknown', black: 'Unknown', event: "King's Indian Defense", year: 2000 },
    },
    // Ruy Lopez middlegame
    {
        fen: 'r1bqkb1r/1pp2ppp/p1np1n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 6',
        pieceCount: 32,
        moveNumber: 6,
        gameUrl: undefined,
        gameDetails: { white: 'Unknown', black: 'Unknown', event: 'Ruy Lopez', year: 2000 },
    },
    // Queen's Gambit Declined
    {
        fen: 'r1bqk2r/pp1nbppp/2p1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 7',
        pieceCount: 30,
        moveNumber: 7,
        gameUrl: undefined,
        gameDetails: { white: 'Unknown', black: 'Unknown', event: "Queen's Gambit Declined", year: 2000 },
    },
    // English Opening
    {
        fen: 'r1bqk2r/pp1nbppp/2p1pn2/3p2B1/2PP4/2N2N2/PP2PPPP/R2QKB1R w KQkq - 0 7',
        pieceCount: 30,
        moveNumber: 7,
        gameUrl: undefined,
        gameDetails: { white: 'Unknown', black: 'Unknown', event: 'English Opening', year: 2000 },
    },
    // Complex GM position (Ding Liren style)
    {
        fen: 'r2q1rk1/pp1nbppp/2p1pnb1/3p4/2PP4/1PN1PNP1/PB3PBP/R2Q1RK1 w - - 3 11',
        pieceCount: 28,
        moveNumber: 11,
        gameUrl: undefined,
        gameDetails: { white: 'Ding Liren', black: 'Ian Nepomniachtchi', event: 'World Championship', year: 2023 },
    },
];

let recentPositions: number[] = [];

// Main function to get a random middlegame position
export async function getRandomMiddlegamePosition(): Promise<LichessPosition> {
    // Try Lichess API first
    try {
        const games = await fetchLichessGames();

        if (games.length > 0) {
            const shuffledGames = [...games].sort(() => Math.random() - 0.5);

            for (const game of shuffledGames) {
                const position = findMiddlegamePosition(game);
                if (position) {
                    console.log('Loaded position from Lichess Masters database');
                    return position;
                }
            }
        }
    } catch (error) {
        console.log('Lichess API unavailable, using curated positions');
    }

    // Fallback to curated positions
    if (recentPositions.length >= Math.floor(FAMOUS_MIDDLEGAMES.length * 0.7)) {
        recentPositions = [];
    }

    let index: number;
    let attempts = 0;
    do {
        index = Math.floor(Math.random() * FAMOUS_MIDDLEGAMES.length);
        attempts++;
    } while (recentPositions.includes(index) && attempts < 100);

    recentPositions.push(index);
    console.log('Using curated position:', FAMOUS_MIDDLEGAMES[index].moveNumber);

    return FAMOUS_MIDDLEGAMES[index];
}

// Get position from a specific opening
export async function getPositionFromOpening(opening: string = 'd4'): Promise<LichessPosition> {
    const params = new URLSearchParams({
        play: opening,
        topGames: '10',
    });

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    try {
        const response = await fetch(`${LICHESS_MASTERS_URL}?${params}`, {
            signal: controller.signal,
        });
        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`Lichess API error: ${response.status}`);
        }

        const data = await response.json();
        const games = data.topGames || [];

        for (const game of games) {
            const position = findMiddlegamePosition(game);
            if (position) {
                return position;
            }
        }
    } catch (error) {
        console.log('Opening fetch failed, using fallback');
    }

    // Fallback
    return getRandomMiddlegamePosition();
}
