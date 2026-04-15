import './App.css'
import { useState, useEffect } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import sendMove from './move_to_flask.js'
import { resetBoard } from './move_to_flask.js';

//https://react-chessboard.vercel.app/?path=/docs/how-to-use-basic-examples--docs

function App() {
  const [game, setGame] = useState(new Chess());
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [gameStatus, setGameStatus] = useState(null);
  
  useEffect(() => {
    resetBoard()
  }, []);

  async function makeMove(move) {
    const gameCopy = new Chess(game.fen());
    const targetSquare = move.to;
    try {
      gameCopy.move(move);
      setGame(gameCopy);
      
      // Add option to play against self maybe, but right now default to sending move to backend
      // Assumes player wants to play against engine and expects response 
      move['color'] = game.turn()
      sendMove(move)

      setSelectedSquare(null);
    
      if (gameCopy.isCheckmate()) {
        setGameStatus(
          gameCopy.turn() === "w"
            ? "Black wins by checkmate!"
            : "White wins by checkmate!"
        );
      } else if (gameCopy.isDraw()) {
        setGameStatus("Game drawn.");
      } else {
        setGameStatus(null);
      }

      return true;

    } catch (error) {
      const piece = game.get(targetSquare)
      if (piece && piece.color == game.turn()) {
        setSelectedSquare(targetSquare);
      } else {
        setSelectedSquare(null)
      }
      console.error(error);
      return false;
    }
  }

  function onPieceDrop ({ sourceSquare, targetSquare }) {
      return makeMove({
        from: sourceSquare,
        to: targetSquare,
        promotion: "q",
      });
  }

  function onSquareClick ({square}) {
    if (!selectedSquare) {
      const piece = game.get(square);
      if (piece && piece.color === game.turn()) {
        setSelectedSquare(square);
      }
      return;
    }
    
    makeMove({
      from: selectedSquare,
      to: square,
      promotion: "q",
    });
  }

  const chessboardOptions = {
    position: game.fen(),
    arePiecesDraggable: true,
    onPieceDrop,
    onSquareClick,
    /** Highlight selected square */
    squareStyles: selectedSquare ? {[selectedSquare]: {
      backgroundColor: "rgba(255, 77, 0, 0.4)"
    }} : {},
  };

  
  return (
      <div style={{ textAlign: "center" }}>
        {gameStatus && (
          <div style={styles.status}>
            {gameStatus}
          </div>
        )}

        <Chessboard options={chessboardOptions} />
      </div>
  );
};

const styles = {
  status: {
    marginBottom: "12px",
    fontSize: "1.5rem",
    fontWeight: "bold",
    color: "#d32f2f",
  },
};

export default App
