import type { PositionEvaluation } from '../chess/types';

const BACKEND_URL = 'http://127.0.0.1:5000';

type EvalCallback = (evaluation: PositionEvaluation) => void;

class StockfishEngine {
    private isReady = false;
    private initPromise: Promise<void> | null = null;

    async init(): Promise<void> {
        if (this.initPromise) {
            return this.initPromise;
        }

        this.initPromise = new Promise(async (resolve, reject) => {
            try {
                const response = await fetch(`${BACKEND_URL}/health`);
                const data = await response.json();

                if (data.status === 'ok' && data.engine_ready) {
                    this.isReady = true;
                    resolve();
                } else {
                    reject(new Error('Stockfish backend not ready'));
                }
            } catch (error) {
                console.error('Failed to connect to Stockfish backend:', error);
                reject(new Error('Failed to connect to Stockfish backend. Is the server running?'));
            }
        });

        return this.initPromise;
    }

    async evaluate(fen: string, depth: number = 18): Promise<PositionEvaluation> {
        if (!this.isReady) {
            await this.init();
        }

        try {
            const response = await fetch(`${BACKEND_URL}/api/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ fen, depth }),
            });

            if (!response.ok) {
                throw new Error(`Backend error: ${response.status}`);
            }

            const data = await response.json();

            return {
                centipawns: data.centipawns || 0,
                mate: data.mate || null,
                depth: data.depth || depth,
                isLoading: false,
            };
        } catch (error) {
            console.error('Evaluation error:', error);
            throw error;
        }
    }

    async evaluateWithProgress(
        fen: string,
        depth: number = 18,
        onProgress: EvalCallback
    ): Promise<PositionEvaluation> {
        onProgress({
            centipawns: 0,
            mate: null,
            depth: 0,
            isLoading: true,
        });

        const result = await this.evaluate(fen, depth);
        onProgress(result);
        return result;
    }

    stop(): void {
        // HTTP backend requests complete on their own
    }

    destroy(): void {
        this.isReady = false;
        this.initPromise = null;
    }
}

let engineInstance: StockfishEngine | null = null;

export function getStockfishEngine(): StockfishEngine {
    if (!engineInstance) {
        engineInstance = new StockfishEngine();
    }
    return engineInstance;
}

export { StockfishEngine };
