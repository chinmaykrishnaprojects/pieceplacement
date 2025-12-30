import React from 'react';
import type { AnalysisResult } from '../chess/types';

interface AnalysisPanelProps {
    result: AnalysisResult | null;
    isVisible: boolean;
    selectedSide: 'white' | 'black' | null;
}

export function AnalysisPanel({ result, isVisible, selectedSide }: AnalysisPanelProps) {
    if (!isVisible || !result) return null;

    const formatEval = (centipawns: number, mate: number | null): string => {
        if (mate !== null) {
            return mate > 0 ? `M${mate}` : `-M${Math.abs(mate)}`;
        }
        const score = centipawns / 100;
        const sign = score >= 0 ? '+' : '';
        return `${sign}${score.toFixed(2)}`;
    };

    // Adjust difference based on selected side
    // If playing black, a negative change in eval (better for black) is good
    const adjustedDifference = selectedSide === 'black'
        ? -result.difference
        : result.difference;

    const isPositive = adjustedDifference > 0;
    const differenceClass = isPositive ? 'positive' : adjustedDifference < 0 ? 'negative' : 'neutral';

    return (
        <div className="analysis-panel">
            <h2>Position Analysis</h2>

            <div className="analysis-grid">
                {/* Initial Evaluation */}
                <div className="analysis-card initial">
                    <div className="card-header">
                        <span className="card-icon">📍</span>
                        <span className="card-title">Starting Position</span>
                    </div>
                    <div className="card-value">
                        {formatEval(result.initialEval.centipawns, result.initialEval.mate)}
                    </div>
                    <div className="card-meta">depth {result.initialEval.depth}</div>
                </div>

                {/* Arrow */}
                <div className="analysis-arrow">
                    <span className="arrow-icon">→</span>
                </div>

                {/* Final Evaluation */}
                <div className="analysis-card final">
                    <div className="card-header">
                        <span className="card-icon">🎯</span>
                        <span className="card-title">After Your Moves</span>
                    </div>
                    <div className="card-value">
                        {formatEval(result.finalEval.centipawns, result.finalEval.mate)}
                    </div>
                    <div className="card-meta">depth {result.finalEval.depth}</div>
                </div>
            </div>

            {/* Net Change */}
            <div className={`net-change ${differenceClass}`}>
                <div className="change-header">
                    <span className="change-icon">{isPositive ? '📈' : adjustedDifference < 0 ? '📉' : '➡️'}</span>
                    <span className="change-title">Net Change for {selectedSide}</span>
                </div>
                <div className="change-value">
                    {isPositive ? '+' : ''}{(adjustedDifference / 100).toFixed(2)} pawns
                </div>
                <div className="change-description">
                    {isPositive
                        ? 'Great job! Your piece placement improved the position!'
                        : adjustedDifference < 0
                            ? 'The position got worse. Try to find better squares for your pieces.'
                            : 'The position remained roughly equal.'}
                </div>
            </div>

            {/* Tips */}
            <div className="analysis-tips">
                <h3>Training Goal</h3>
                <p>
                    This training is about improving <strong>piece placement</strong> and <strong>coordination</strong>.
                    Focus on:
                </p>
                <ul>
                    <li>Centralizing your pieces</li>
                    <li>Creating harmony between pieces</li>
                    <li>Improving your worst-placed piece</li>
                    <li>Controlling key squares</li>
                </ul>
            </div>
        </div>
    );
}
