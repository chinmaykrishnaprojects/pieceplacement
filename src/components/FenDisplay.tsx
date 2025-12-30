import React, { useState } from 'react';

interface FenDisplayProps {
    fen: string;
    onLoadFen: (fen: string) => boolean;
    isLocked: boolean;
}

export function FenDisplay({ fen, onLoadFen, isLocked }: FenDisplayProps) {
    const [editMode, setEditMode] = useState(false);
    const [inputFen, setInputFen] = useState(fen);
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = () => {
        const success = onLoadFen(inputFen);
        if (success) {
            setEditMode(false);
            setError(null);
        } else {
            setError('Invalid FEN string');
        }
    };

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(fen);
            // Could add a toast notification here
        } catch (err) {
            console.error('Failed to copy FEN:', err);
        }
    };

    return (
        <div className="fen-display">
            <div className="fen-header">
                <h3>Position (FEN)</h3>
                <div className="fen-actions">
                    <button
                        className="fen-button"
                        onClick={handleCopy}
                        title="Copy FEN"
                    >
                        📋
                    </button>
                    {!isLocked && (
                        <button
                            className="fen-button"
                            onClick={() => {
                                setEditMode(!editMode);
                                setInputFen(fen);
                                setError(null);
                            }}
                            title={editMode ? 'Cancel' : 'Edit FEN'}
                        >
                            {editMode ? '✕' : '✏️'}
                        </button>
                    )}
                </div>
            </div>

            {editMode ? (
                <div className="fen-edit">
                    <input
                        type="text"
                        value={inputFen}
                        onChange={(e) => setInputFen(e.target.value)}
                        className={`fen-input ${error ? 'error' : ''}`}
                        placeholder="Enter FEN string..."
                    />
                    {error && <span className="fen-error">{error}</span>}
                    <button
                        className="fen-load-button"
                        onClick={handleSubmit}
                    >
                        Load Position
                    </button>
                </div>
            ) : (
                <code className="fen-string">{fen}</code>
            )}
        </div>
    );
}
