"""
Flask backend for Stockfish chess engine evaluation.
Communicates with the Stockfish binary via UCI protocol using the stockfish Python library.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from stockfish import Stockfish
import chess

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Stockfish path - using non-AVX2 version for compatibility
STOCKFISH_PATH = r"C:\Users\krish\positionalchess\public\stockfish\stockfish-windows-x86-64.exe"

# Lower depth for faster analysis
DEFAULT_DEPTH = 12

def create_engine():
    """Create a new Stockfish engine instance."""
    try:
        engine = Stockfish(
            path=STOCKFISH_PATH,
            depth=DEFAULT_DEPTH,
            parameters={
                "Threads": 2,
                "Hash": 128,
            }
        )
        print("Stockfish engine initialized successfully")
        return engine
    except Exception as e:
        print(f"Failed to initialize Stockfish: {e}")
        return None

# Create engine at startup
stockfish = create_engine()


def get_absolute_eval(engine, fen):
    """
    Get evaluation normalized to White's perspective (absolute eval).
    Returns centipawns where positive = White advantage.
    """
    engine.set_fen_position(fen)
    evaluation = engine.get_evaluation()
    
    # Determine whose turn it is
    is_white_turn = ' w ' in fen
    
    centipawns = 0
    mate = None
    
    if evaluation['type'] == 'cp':
        centipawns = evaluation['value']
        # If it's Black's turn, Stockfish reports from Black's perspective
        # So we need to negate to get White's perspective
        if not is_white_turn:
            centipawns = -centipawns
    elif evaluation['type'] == 'mate':
        mate = evaluation['value']
        if not is_white_turn:
            mate = -mate
        centipawns = 10000 if mate > 0 else -10000
    
    return centipawns, mate


def get_tempo_adjusted_eval(engine, fen):
    """
    Get tempo-adjusted evaluation by averaging eval for both sides to move.
    This eliminates tempo advantage and gives a more stable positional assessment.
    """
    # Get eval with current side to move
    cp1, mate1 = get_absolute_eval(engine, fen)
    
    if mate1 is not None:
        # Don't average if there's a forced mate
        return cp1, mate1
    
    # Swap the side to move and get eval
    parts = fen.split(' ')
    parts[1] = 'b' if parts[1] == 'w' else 'w'
    flipped_fen = ' '.join(parts)
    
    cp2, mate2 = get_absolute_eval(engine, flipped_fen)
    
    if mate2 is not None:
        # One side has mate - use original eval
        return cp1, mate1
    
    # Average the two evaluations
    avg_cp = (cp1 + cp2) // 2
    return avg_cp, None


def is_capture_move(fen, move_uci):
    """Check if a move is a capture."""
    board = chess.Board(fen)
    try:
        move = chess.Move.from_uci(move_uci)
        return board.is_capture(move)
    except:
        return False


def is_trap_move(engine, fen, move_uci, threshold=100):
    """
    Check if a move creates a 'trap' - where eval swings > threshold 
    depending on whose turn it is.
    """
    board = chess.Board(fen)
    try:
        move = chess.Move.from_uci(move_uci)
        board.push(move)
        new_fen = board.fen()
        
        # Get eval for both sides to move
        cp_current, _ = get_absolute_eval(engine, new_fen)
        
        # Flip turn
        parts = new_fen.split(' ')
        parts[1] = 'b' if parts[1] == 'w' else 'w'
        flipped_fen = ' '.join(parts)
        
        cp_flipped, _ = get_absolute_eval(engine, flipped_fen)
        
        # If difference is > threshold, it's a trap
        return abs(cp_current - cp_flipped) > threshold
    except Exception as e:
        print(f"Trap check error: {e}")
        return False


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    global stockfish
    engine_ready = stockfish is not None
    return jsonify({
        'status': 'ok' if engine_ready else 'error',
        'engine_ready': engine_ready
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyze a chess position with absolute (White-perspective) evaluation.
    
    Request JSON:
        - fen: FEN string of the position
        - depth: (optional) Analysis depth, default 12
        - tempoAdjusted: (optional) If true, average eval for both sides
    
    Response JSON:
        - centipawns: Evaluation in centipawns (positive = White advantage)
        - mate: Mate in N moves (null if no forced mate)
        - bestMove: Best move in UCI notation
        - depth: Analysis depth used
    """
    global stockfish
    
    if stockfish is None:
        stockfish = create_engine()
        if stockfish is None:
            return jsonify({'error': 'Engine not ready'}), 503
    
    data = request.get_json()
    if not data or 'fen' not in data:
        return jsonify({'error': 'Missing FEN in request'}), 400
    
    fen = data['fen']
    depth = data.get('depth', DEFAULT_DEPTH)
    tempo_adjusted = data.get('tempoAdjusted', False)
    
    try:
        # Validate FEN
        if not stockfish.is_fen_valid(fen):
            return jsonify({'error': 'Invalid FEN string'}), 400
        
        stockfish.set_depth(depth)
        
        # Get evaluation
        if tempo_adjusted:
            centipawns, mate = get_tempo_adjusted_eval(stockfish, fen)
        else:
            centipawns, mate = get_absolute_eval(stockfish, fen)
        
        # Get best move
        stockfish.set_fen_position(fen)
        best_move = stockfish.get_best_move()
        
        return jsonify({
            'centipawns': centipawns,
            'mate': mate,
            'bestMove': best_move,
            'depth': depth
        })
        
    except Exception as e:
        print(f"Analysis error: {e}")
        stockfish = create_engine()
        return jsonify({'error': str(e)}), 500


@app.route('/api/best-move', methods=['POST'])
def best_move():
    """
    Get the best non-capture, non-trap move for a position.
    
    Request JSON:
        - fen: FEN string of the position
        - depth: (optional) Analysis depth, default 12
        - allowCaptures: (optional) If false, filters out captures
        - trapThreshold: (optional) Max eval swing to allow (in centipawns)
    
    Response JSON:
        - bestMove: Best move in UCI notation
        - isCapture: Whether the move is a capture
    """
    global stockfish
    
    if stockfish is None:
        stockfish = create_engine()
        if stockfish is None:
            return jsonify({'error': 'Engine not ready'}), 503
    
    data = request.get_json()
    if not data or 'fen' not in data:
        return jsonify({'error': 'Missing FEN in request'}), 400
    
    fen = data['fen']
    depth = data.get('depth', DEFAULT_DEPTH)
    allow_captures = data.get('allowCaptures', False)
    trap_threshold = data.get('trapThreshold', 100)  # 1 pawn = 100cp
    
    try:
        if not stockfish.is_fen_valid(fen):
            return jsonify({'error': 'Invalid FEN string'}), 400
        
        stockfish.set_fen_position(fen)
        stockfish.set_depth(depth)
        
        # Get top N moves to filter
        top_moves = stockfish.get_top_moves(10)
        
        for move_info in top_moves:
            move_uci = move_info['Move']
            
            # Skip captures if not allowed
            if not allow_captures and is_capture_move(fen, move_uci):
                continue
            
            # Skip trap moves
            if is_trap_move(stockfish, fen, move_uci, trap_threshold):
                continue
            
            # Found a valid move
            return jsonify({
                'bestMove': move_uci,
                'isCapture': is_capture_move(fen, move_uci)
            })
        
        # Fallback: return best move even if it's a capture/trap
        best = stockfish.get_best_move()
        return jsonify({
            'bestMove': best,
            'isCapture': is_capture_move(fen, best),
            'warning': 'No non-capture non-trap move found, using best move'
        })
        
    except Exception as e:
        print(f"Best move error: {e}")
        stockfish = create_engine()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Stockfish backend server...")
    print(f"Engine status: {'Ready' if stockfish else 'NOT READY'}")
    print(f"Default depth: {DEFAULT_DEPTH}")
    app.run(host='127.0.0.1', port=5000, debug=True)
