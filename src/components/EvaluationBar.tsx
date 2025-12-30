import React from 'react';
import type { PositionEvaluation } from '../chess/types';

interface EvaluationBarProps {
    evaluation: PositionEvaluation | null;
    label?: string;
    isLoading?: boolean;
}

export function EvaluationBar({ evaluation, label, isLoading }: EvaluationBarProps) {
    // Calculate bar fill percentage (capped at -500 to +500 centipawns for display)
    const getBarPercentage = (): number => {
        if (!evaluation) return 50;

        if (evaluation.mate !== null) {
            return evaluation.mate > 0 ? 100 : 0;
        }

        // Clamp centipawns to -500 to +500 range
        const clampedScore = Math.max(-500, Math.min(500, evaluation.centipawns));
        // Convert to 0-100 percentage (50 is equal, 100 is +500cp, 0 is -500cp)
        return 50 + (clampedScore / 10);
    };

    // Format the evaluation for display
    const formatEval = (): string => {
        if (!evaluation) return '-';

        if (evaluation.mate !== null) {
            return evaluation.mate > 0 ? `M${evaluation.mate}` : `-M${Math.abs(evaluation.mate)}`;
        }

        const score = evaluation.centipawns / 100;
        const sign = score >= 0 ? '+' : '';
        return `${sign}${score.toFixed(2)}`;
    };

    // Get color class based on evaluation
    const getEvalClass = (): string => {
        if (!evaluation) return 'neutral';

        if (evaluation.mate !== null) {
            return evaluation.mate > 0 ? 'winning' : 'losing';
        }

        if (evaluation.centipawns > 100) return 'advantage';
        if (evaluation.centipawns < -100) return 'disadvantage';
        return 'neutral';
    };

    const percentage = getBarPercentage();

    return (
        <div className="evaluation-bar-container">
            {label && <div className="eval-label">{label}</div>}

            <div className="evaluation-bar">
                <div
                    className="eval-fill white-side"
                    style={{ height: `${percentage}%` }}
                />
                <div className="eval-line" />
            </div>

            <div className={`eval-score ${getEvalClass()}`}>
                {isLoading ? (
                    <span className="eval-loading">
                        <span className="spinner small"></span>
                        {evaluation && ` d${evaluation.depth}`}
                    </span>
                ) : (
                    <>
                        <span className="eval-value">{formatEval()}</span>
                        {evaluation && <span className="eval-depth">depth {evaluation.depth}</span>}
                    </>
                )}
            </div>
        </div>
    );
}
