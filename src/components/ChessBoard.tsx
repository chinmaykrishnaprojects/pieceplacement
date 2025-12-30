import { useState, useCallback, useMemo, useEffect } from 'react'
import './ChessBoard.css'

// Lichess SVG piece sprites
const PIECE_SVGS: Record<string, string> = {
    'K': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22.5 11.63V6M20 8h5" stroke-linejoin="miter"/><path d="M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5" fill="#fff" stroke-linecap="butt" stroke-linejoin="miter"/><path d="M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V27v-3.5c-3.5-7.5-13-10.5-16-4-3 6 5 10 5 10V37z" fill="#fff"/><path d="M11.5 30c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0"/></g></svg>`,
    'Q': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill="#fff" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M8 12a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM24.5 7.5a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM41 12a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM16 8.5a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM33 9a2 2 0 1 1-4 0 2 2 0 1 1 4 0z"/><path d="M9 26c8.5-1.5 21-1.5 27 0l2-12-7 11V11l-5.5 13.5-3-15-3 15-5.5-14V25L7 14l2 12z" stroke-linecap="butt"/><path d="M9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 1.5-1 0-2.5 0 0 .5-1.5-1-2.5-.5-2.5-.5-2 .5-3.5 1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z" stroke-linecap="butt"/><path d="M11.5 30c3.5-1 18.5-1 22 0M12 33.5c6-1 15-1 21 0" fill="none"/></g></svg>`,
    'R': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill="#fff" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 39h27v-3H9v3zM12 36v-4h21v4H12zM11 14V9h4v2h5V9h5v2h5V9h4v5" stroke-linecap="butt"/><path d="M34 14l-3 3H14l-3-3"/><path d="M31 17v12.5H14V17" stroke-linecap="butt" stroke-linejoin="miter"/><path d="M31 29.5l1.5 2.5h-20l1.5-2.5"/><path d="M11 14h23" fill="none" stroke-linejoin="miter"/></g></svg>`,
    'B': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><g fill="#fff" stroke-linecap="butt"><path d="M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.354.49-2.323.47-3-.5 1.354-1.94 3-2 3-2z"/><path d="M15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2z"/><path d="M25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z"/></g><path d="M17.5 26h10M15 30h15m-7.5-14.5v5M20 18h5" stroke-linejoin="miter"/></g></svg>`,
    'N': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10c10.5 1 16.5 8 16 29H15c0-9 10-6.5 8-21" fill="#fff"/><path d="M24 18c.38 2.91-5.55 7.37-8 9-3 2-2.82 4.34-5 4-1.042-.94 1.41-3.04 0-3-1 0 .19 1.23-1 2-1 0-4.003 1-4-4 0-2 6-12 6-12s1.89-1.9 2-3.5c-.73-.994-.5-2-.5-3 1-1 3 2.5 3 2.5h2s.78-1.992 2.5-3c1 0 1 3 1 3" fill="#fff"/><path d="M9.5 25.5a.5.5 0 1 1-1 0 .5.5 0 1 1 1 0zm5.433-9.75a.5 1.5 30 1 1-.866-.5.5 1.5 30 1 1 .866.5z" fill="#000"/></g></svg>`,
    'P': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><path d="M22.5 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38C17.33 16.5 16 18.59 16 21c0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-2.78-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z" fill="#fff" stroke="#000" stroke-width="1.5" stroke-linecap="round"/></svg>`,
    'k': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22.5 11.63V6" stroke-linejoin="miter"/><path d="M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5" fill="#000" stroke-linecap="butt" stroke-linejoin="miter"/><path d="M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V27v-3.5c-3.5-7.5-13-10.5-16-4-3 6 5 10 5 10V37z" fill="#000"/><path d="M20 8h5" stroke-linejoin="miter"/><path d="M11.5 30c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0" stroke="#fff"/></g></svg>`,
    'q': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><g fill="#000" stroke="none"><circle cx="6" cy="12" r="2.75"/><circle cx="14" cy="9" r="2.75"/><circle cx="22.5" cy="8" r="2.75"/><circle cx="31" cy="9" r="2.75"/><circle cx="39" cy="12" r="2.75"/></g><path d="M9 26c8.5-1.5 21-1.5 27 0l2.5-12.5L31 25l-.3-14.1-5.2 13.6-3-14.5-3 14.5-5.2-13.6L14 25 6.5 13.5 9 26z" fill="#000" stroke-linecap="butt"/><path d="M9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 1.5-1 0-2.5 0 0 .5-1.5-1-2.5-.5-2.5-.5-2 .5-3.5 1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z" fill="#000" stroke-linecap="butt"/><path d="M11 38.5a35 35 1 0 0 23 0" fill="none" stroke-linecap="butt"/><path d="M11 29a35 35 1 0 1 23 0m-21.5 2.5h20m-21 3a35 35 1 0 0 22 0" fill="none" stroke="#fff"/></g></svg>`,
    'r': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 39h27v-3H9v3zM12.5 32l1.5-2.5h17l1.5 2.5h-20zM12 36v-4h21v4H12z" fill="#000" stroke-linecap="butt"/><path d="M14 29.5v-13h17v13H14z" fill="#000" stroke-linecap="butt" stroke-linejoin="miter"/><path d="M14 16.5L11 14h23l-3 2.5H14zM11 14V9h4v2h5V9h5v2h5V9h4v5H11z" fill="#000" stroke-linecap="butt"/><path d="M12 35.5h21m-20-4h19m-18-2h17m-17-13h17M11 14h23" fill="none" stroke="#fff" stroke-width="1" stroke-linejoin="miter"/></g></svg>`,
    'b': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><g fill="#000" stroke-linecap="butt"><path d="M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.354.49-2.323.47-3-.5 1.354-1.94 3-2 3-2z"/><path d="M15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2z"/><path d="M25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z"/></g><path d="M17.5 26h10M15 30h15m-7.5-14.5v5M20 18h5" stroke="#fff" stroke-linejoin="miter"/></g></svg>`,
    'n': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10c10.5 1 16.5 8 16 29H15c0-9 10-6.5 8-21" fill="#000"/><path d="M24 18c.38 2.91-5.55 7.37-8 9-3 2-2.82 4.34-5 4-1.042-.94 1.41-3.04 0-3-1 0 .19 1.23-1 2-1 0-4.003 1-4-4 0-2 6-12 6-12s1.89-1.9 2-3.5c-.73-.994-.5-2-.5-3 1-1 3 2.5 3 2.5h2s.78-1.992 2.5-3c1 0 1 3 1 3" fill="#000"/><path d="M9.5 25.5a.5.5 0 1 1-1 0 .5.5 0 1 1 1 0zm5.433-9.75a.5 1.5 30 1 1-.866-.5.5 1.5 30 1 1 .866.5z" fill="#fff" stroke="#fff"/></g></svg>`,
    'p': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><path d="M22.5 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38C17.33 16.5 16 18.59 16 21c0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-2.78-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z" fill="#000" stroke="#000" stroke-width="1.5" stroke-linecap="round"/></svg>`,
}

interface ChessBoardProps {
    fen: string
    onMove: (from: string, to: string, promotion?: string) => Promise<boolean>
    isFlipped: boolean
    playerColor: 'white' | 'black'
    disabled: boolean
    lastMove?: { from: string; to: string } | null  // Passed from parent for LLM moves
    allowBothSides?: boolean  // Allow moving pieces of either color (for Analysis mode)
}

export default function ChessBoard({ fen, onMove, isFlipped, playerColor, disabled, lastMove: externalLastMove, allowBothSides = false }: ChessBoardProps) {
    const [selectedSquare, setSelectedSquare] = useState<string | null>(null)
    const [legalMoves, setLegalMoves] = useState<string[]>([])
    const [showPromotion, setShowPromotion] = useState<{ from: string, to: string } | null>(null)

    // Clear selection when disabled changes (e.g., LLM is thinking)
    useEffect(() => {
        if (disabled) {
            setSelectedSquare(null)
            setLegalMoves([])
            setShowPromotion(null)
        }
    }, [disabled])

    // Parse FEN to get board position
    const board = useMemo(() => {
        const position = fen.split(' ')[0]
        const rows = position.split('/')
        const squares: Record<string, string> = {}

        rows.forEach((row, rowIndex) => {
            let colIndex = 0
            for (const char of row) {
                if ('12345678'.includes(char)) {
                    colIndex += parseInt(char)
                } else {
                    const file = 'abcdefgh'[colIndex]
                    const rank = 8 - rowIndex
                    squares[`${file}${rank}`] = char
                    colIndex++
                }
            }
        })
        return squares
    }, [fen])

    const turn = fen.split(' ')[1]
    const isPlayerTurn = (turn === 'w' && playerColor === 'white') || (turn === 'b' && playerColor === 'black')

    // Calculate legal moves for a piece (simplified - server validates)
    const calculateLegalMoves = useCallback((square: string): string[] => {
        const piece = board[square]
        if (!piece) return []

        const moves: string[] = []
        const pieceType = piece.toLowerCase()
        const isWhite = piece === piece.toUpperCase()
        const file = square[0]
        const rank = parseInt(square[1])
        const fileIdx = 'abcdefgh'.indexOf(file)

        if (pieceType === 'p') {
            const dir = isWhite ? 1 : -1
            const startRank = isWhite ? 2 : 7
            const fwd1 = `${file}${rank + dir}`
            if (rank + dir >= 1 && rank + dir <= 8 && !board[fwd1]) {
                moves.push(fwd1)
                if (rank === startRank) {
                    const fwd2 = `${file}${rank + 2 * dir}`
                    if (!board[fwd2]) moves.push(fwd2)
                }
            }
            // Captures
            if (fileIdx > 0 && rank + dir >= 1 && rank + dir <= 8) {
                const cap1 = `${'abcdefgh'[fileIdx - 1]}${rank + dir}`
                if (board[cap1] && (isWhite !== (board[cap1] === board[cap1].toUpperCase()))) moves.push(cap1)
            }
            if (fileIdx < 7 && rank + dir >= 1 && rank + dir <= 8) {
                const cap2 = `${'abcdefgh'[fileIdx + 1]}${rank + dir}`
                if (board[cap2] && (isWhite !== (board[cap2] === board[cap2].toUpperCase()))) moves.push(cap2)
            }
        } else if (pieceType === 'n') {
            const offsets = [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]]
            for (const [df, dr] of offsets) {
                const nf = fileIdx + df, nr = rank + dr
                if (nf >= 0 && nf < 8 && nr >= 1 && nr <= 8) {
                    const target = `${'abcdefgh'[nf]}${nr}`
                    const tp = board[target]
                    if (!tp || (isWhite !== (tp === tp.toUpperCase()))) moves.push(target)
                }
            }
        } else if (pieceType === 'k') {
            for (let df = -1; df <= 1; df++) {
                for (let dr = -1; dr <= 1; dr++) {
                    if (df === 0 && dr === 0) continue
                    const nf = fileIdx + df, nr = rank + dr
                    if (nf >= 0 && nf < 8 && nr >= 1 && nr <= 8) {
                        const target = `${'abcdefgh'[nf]}${nr}`
                        const tp = board[target]
                        if (!tp || (isWhite !== (tp === tp.toUpperCase()))) moves.push(target)
                    }
                }
            }
        } else {
            const dirs: [number, number][] = pieceType === 'r' ? [[0, 1], [0, -1], [1, 0], [-1, 0]]
                : pieceType === 'b' ? [[1, 1], [1, -1], [-1, 1], [-1, -1]]
                    : [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
            for (const [df, dr] of dirs) {
                for (let i = 1; i < 8; i++) {
                    const nf = fileIdx + df * i, nr = rank + dr * i
                    if (nf < 0 || nf > 7 || nr < 1 || nr > 8) break
                    const target = `${'abcdefgh'[nf]}${nr}`
                    const tp = board[target]
                    if (tp) {
                        if (isWhite !== (tp === tp.toUpperCase())) moves.push(target)
                        break
                    }
                    moves.push(target)
                }
            }
        }
        return moves
    }, [board])

    const handleSquareClick = useCallback(async (square: string) => {
        if (disabled) return
        const piece = board[square]

        if (selectedSquare) {
            if (square === selectedSquare) {
                setSelectedSquare(null)
                setLegalMoves([])
                return
            }

            const fromPiece = board[selectedSquare]
            if (fromPiece?.toLowerCase() === 'p') {
                const toRank = parseInt(square[1])
                // Check if pawn is promoting based on piece color, not playerColor
                const isWhitePawn = fromPiece === 'P'
                if ((isWhitePawn && toRank === 8) || (!isWhitePawn && toRank === 1)) {
                    setShowPromotion({ from: selectedSquare, to: square })
                    return
                }
            }

            // Attempt move - clear selection immediately
            setSelectedSquare(null)
            setLegalMoves([])
            await onMove(selectedSquare, square)
            return
        }

        // In allowBothSides mode: select any piece whose turn it is
        // In normal mode: only select player's own pieces
        if (piece) {
            const isWhitePiece = piece === piece.toUpperCase()
            const isWhiteTurn = turn === 'w'

            if (allowBothSides) {
                // Allow selecting any piece if it's that color's turn
                if ((isWhitePiece && isWhiteTurn) || (!isWhitePiece && !isWhiteTurn)) {
                    setSelectedSquare(square)
                    setLegalMoves(calculateLegalMoves(square))
                }
            } else if (isPlayerTurn) {
                // Normal mode: only allow selecting player's pieces
                if ((isWhitePiece && playerColor === 'white') || (!isWhitePiece && playerColor === 'black')) {
                    setSelectedSquare(square)
                    setLegalMoves(calculateLegalMoves(square))
                }
            }
        }
    }, [board, selectedSquare, disabled, playerColor, isPlayerTurn, onMove, calculateLegalMoves, allowBothSides, turn])

    const handlePromotion = async (promotionPiece: string) => {
        if (!showPromotion) return
        setShowPromotion(null)
        setSelectedSquare(null)
        setLegalMoves([])
        await onMove(showPromotion.from, showPromotion.to, promotionPiece)
    }

    const squares = []
    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
            const file = isFlipped ? 'abcdefgh'[7 - col] : 'abcdefgh'[col]
            const rank = isFlipped ? row + 1 : 8 - row
            const sq = `${file}${rank}`
            const piece = board[sq]
            const isLight = (row + col) % 2 === 0
            const isSelected = selectedSquare === sq
            const isLegalTarget = legalMoves.includes(sq)
            const isLastMoveSquare = externalLastMove && (externalLastMove.from === sq || externalLastMove.to === sq)

            squares.push(
                <div key={sq} className={`square ${isLight ? 'light' : 'dark'} ${isSelected ? 'selected' : ''} ${isLastMoveSquare ? 'last-move' : ''}`} onClick={() => handleSquareClick(sq)}>
                    {col === 0 && <span className="coord rank">{rank}</span>}
                    {row === 7 && <span className="coord file">{file}</span>}
                    {piece && PIECE_SVGS[piece] && <div className="piece" dangerouslySetInnerHTML={{ __html: PIECE_SVGS[piece] }} />}
                    {isLegalTarget && <div className={`legal-move-marker ${piece ? 'capture' : ''}`} />}
                </div>
            )
        }
    }

    const promoP = playerColor === 'white' ? ['Q', 'R', 'B', 'N'] : ['q', 'r', 'b', 'n']

    return (
        <div className="chess-board-wrapper">
            <div className="chess-board">{squares}</div>
            {showPromotion && (
                <div className="promotion-overlay">
                    <div className="promotion-modal">
                        {promoP.map(p => <button key={p} className="promotion-piece" onClick={() => handlePromotion(p.toLowerCase())}><div dangerouslySetInnerHTML={{ __html: PIECE_SVGS[p] }} /></button>)}
                    </div>
                </div>
            )}
        </div>
    )
}
