"""
Flask backend for Stockfish chess engine evaluation.
Communicates with the Stockfish binary via UCI protocol using the stockfish Python library.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from stockfish import Stockfish

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Initialize Stockfish engine
# Using the local Stockfish executable
STOCKFISH_PATH = r"C:\Users\krish\positionalchess\public\stockfish-windows-x86-64-avx2.exe"
try:
    stockfish = Stockfish(path=STOCKFISH_PATH)
    stockfish.set_depth(18)  # Default analysis depth
    ENGINE_READY = True
except Exception as e:
    print(f"Failed to initialize Stockfish: {e}")
    print("Make sure Stockfish is installed and in your PATH.")
    ENGINE_READY = False
    stockfish = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok' if ENGINE_READY else 'error',
        'engine_ready': ENGINE_READY
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyze a chess position.
    
    Request JSON:
        - fen: FEN string of the position
        - depth: (optional) Analysis depth, default 18
    
    Response JSON:
        - centipawns: Evaluation in centipawns (100 = 1 pawn advantage)
        - mate: Mate in N moves (null if no forced mate)
        - bestMove: Best move in UCI notation (e.g., "e2e4")
        - depth: Analysis depth used
    """
    if not ENGINE_READY or stockfish is None:
        return jsonify({'error': 'Engine not ready'}), 503
    
    data = request.get_json()
    if not data or 'fen' not in data:
        return jsonify({'error': 'Missing FEN in request'}), 400
    
    fen = data['fen']
    depth = data.get('depth', 18)
    
    try:
        # Set the position
        stockfish.set_fen_position(fen)
        stockfish.set_depth(depth)
        
        # Get evaluation
        evaluation = stockfish.get_evaluation()
        
        # Get best move
        best_move = stockfish.get_best_move()
        
        # Parse evaluation
        centipawns = 0
        mate = None
        
        if evaluation['type'] == 'cp':
            centipawns = evaluation['value']
        elif evaluation['type'] == 'mate':
            mate = evaluation['value']
            # Convert mate to large centipawn value for UI consistency
            centipawns = 10000 if mate > 0 else -10000
        
        return jsonify({
            'centipawns': centipawns,
            'mate': mate,
            'bestMove': best_move,
            'depth': depth
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/best-move', methods=['POST'])
def best_move():
    """
    Get only the best move for a position (faster, no full eval).
    
    Request JSON:
        - fen: FEN string of the position
        - depth: (optional) Analysis depth, default 12
    
    Response JSON:
        - bestMove: Best move in UCI notation
    """
    if not ENGINE_READY or stockfish is None:
        return jsonify({'error': 'Engine not ready'}), 503
    
    data = request.get_json()
    if not data or 'fen' not in data:
        return jsonify({'error': 'Missing FEN in request'}), 400
    
    fen = data['fen']
    depth = data.get('depth', 12)
    
    try:
        stockfish.set_fen_position(fen)
        stockfish.set_depth(depth)
        best_move = stockfish.get_best_move()
        
        return jsonify({
            'bestMove': best_move
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Stockfish backend server...")
    print(f"Engine status: {'Ready' if ENGINE_READY else 'NOT READY'}")
    app.run(host='127.0.0.1', port=5000, debug=True)
