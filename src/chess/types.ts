// Chess game types and interfaces

export interface GameState {
  fen: string;
  initialFen: string;
  moveHistory: string[];
  movesRemaining: number;
  selectedSide: 'white' | 'black' | null;
  isLocked: boolean;
  gamePhase: 'setup' | 'playing' | 'analyzing' | 'complete';
}

export interface PositionEvaluation {
  centipawns: number;
  mate: number | null;
  depth: number;
  isLoading: boolean;
}

export interface AnalysisResult {
  initialEval: PositionEvaluation;
  finalEval: PositionEvaluation;
  difference: number;
  isImprovement: boolean;
}

export interface MoveResult {
  success: boolean;
  newFen: string;
  move: string;
  error?: string;
}

export interface GameDetails {
  white: string;
  black: string;
  event: string;
  year: number;
}

export interface LichessPosition {
  fen: string;
  pieceCount: number;
  moveNumber: number;
  gameUrl?: string;
  gameDetails?: GameDetails;
}
