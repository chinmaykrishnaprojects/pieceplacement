import { useState, useCallback, useEffect, useRef } from 'react';
import type { PositionEvaluation, AnalysisResult } from '../chess/types';
import { getStockfishEngine, StockfishEngine } from './stockfishWorker';

const DEFAULT_DEPTH = 18;

export function useStockfish() {
    const [isLoading, setIsLoading] = useState(false);
    const [isInitialized, setIsInitialized] = useState(false);
    const [currentEval, setCurrentEval] = useState<PositionEvaluation | null>(null);
    const [error, setError] = useState<string | null>(null);
    const engineRef = useRef<StockfishEngine | null>(null);

    // Initialize engine on mount
    useEffect(() => {
        const initEngine = async () => {
            try {
                const engine = getStockfishEngine();
                await engine.init();
                engineRef.current = engine;
                setIsInitialized(true);
                setError(null);
            } catch (err) {
                console.error('Failed to initialize Stockfish:', err);
                setError('Failed to initialize chess engine');
            }
        };

        initEngine();

        return () => {
            // Don't destroy the singleton, just stop any running analysis
            engineRef.current?.stop();
        };
    }, []);

    // Evaluate a single position
    const evaluatePosition = useCallback(async (
        fen: string,
        depth: number = DEFAULT_DEPTH
    ): Promise<PositionEvaluation> => {
        if (!engineRef.current) {
            throw new Error('Engine not initialized');
        }

        setIsLoading(true);
        setError(null);

        try {
            const evaluation = await engineRef.current.evaluateWithProgress(
                fen,
                depth,
                (progress) => {
                    setCurrentEval(progress);
                }
            );

            setCurrentEval(evaluation);
            setIsLoading(false);
            return evaluation;
        } catch (err) {
            console.error('Evaluation error:', err);
            setError('Failed to evaluate position');
            setIsLoading(false);
            throw err;
        }
    }, []);

    // Compare two positions
    const comparePositions = useCallback(async (
        initialFen: string,
        finalFen: string,
        depth: number = DEFAULT_DEPTH
    ): Promise<AnalysisResult> => {
        if (!engineRef.current) {
            throw new Error('Engine not initialized');
        }

        setIsLoading(true);
        setError(null);

        try {
            // Evaluate initial position
            const initialEval = await engineRef.current.evaluate(initialFen, depth);

            // Evaluate final position
            const finalEval = await engineRef.current.evaluate(finalFen, depth);

            // Calculate difference (positive = improvement for the side that moved)
            const difference = finalEval.centipawns - initialEval.centipawns;

            const result: AnalysisResult = {
                initialEval: { ...initialEval, isLoading: false },
                finalEval: { ...finalEval, isLoading: false },
                difference,
                isImprovement: difference > 0,
            };

            setIsLoading(false);
            return result;
        } catch (err) {
            console.error('Comparison error:', err);
            setError('Failed to compare positions');
            setIsLoading(false);
            throw err;
        }
    }, []);

    // Stop current analysis
    const stopAnalysis = useCallback(() => {
        engineRef.current?.stop();
        setIsLoading(false);
    }, []);

    // Format evaluation for display
    const formatEvaluation = useCallback((eval_: PositionEvaluation | null): string => {
        if (!eval_) return '-';

        if (eval_.mate !== null) {
            return eval_.mate > 0 ? `M${eval_.mate}` : `-M${Math.abs(eval_.mate)}`;
        }

        const score = eval_.centipawns / 100;
        const sign = score >= 0 ? '+' : '';
        return `${sign}${score.toFixed(2)}`;
    }, []);

    return {
        isLoading,
        isInitialized,
        currentEval,
        error,
        evaluatePosition,
        comparePositions,
        stopAnalysis,
        formatEvaluation,
    };
}
